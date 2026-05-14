from pathlib import Path
from typing import Literal

import torch
from torch import nn
from transformers import BatchEncoding, BlipForConditionalGeneration, BlipProcessor


DEFAULT_BLIP_CHECKPOINT = "Salesforce/blip-image-captioning-base"
DEFAULT_CAPTION_PREFIX = "In exactly one word, this is an image of"


class BLIPWrapper(nn.Module):
    def __init__(
        self,
        checkpoint: str = DEFAULT_BLIP_CHECKPOINT,
        device: str | None = None,
        extract_vision_embeddings: bool = False,
        force_dropout_prob: float | None = None,
        lora_adapter: str | None = None,
        bayesian_lora_factors: str | Path | None = None,
        bayesian_lora_prior_var: float = 1.0,
        bayesian_lora_top_k: int = 5,
        bayesian_lora_n_lora: int | None = 32,
        bayesian_lora_n_kfac: int | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extract_vision_embeddings = extract_vision_embeddings
        self.lora_adapter = lora_adapter
        self.bayesian_lora_prior_var = bayesian_lora_prior_var
        self.bayesian_lora_top_k = bayesian_lora_top_k
        self.bayesian_lora_n_lora = bayesian_lora_n_lora
        self.bayesian_lora_n_kfac = bayesian_lora_n_kfac
        self.processor = BlipProcessor.from_pretrained(checkpoint)
        self.model = self.load_model(checkpoint, lora_adapter).to(self.device)
        self.bayesian_lora_factors = self.load_bayesian_lora_factors(bayesian_lora_factors)
        if force_dropout_prob is not None:
            self.set_dropout_probability(force_dropout_prob)
        self.model.eval()

    def load_model(
        self,
        checkpoint: str,
        lora_adapter: str | None = None,
    ) -> nn.Module:
        model_kwargs = {}
        if lora_adapter is not None:
            model_kwargs["torch_dtype"] = (
                torch.float16 if self.device.type == "cuda" else torch.float32
            )

        model = BlipForConditionalGeneration.from_pretrained(checkpoint, **model_kwargs)
        if lora_adapter is None:
            return model

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter, is_trainable=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora" in name.lower()
        return model

    def load_bayesian_lora_factors(
        self,
        factors_path: str | Path | None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]] | None:
        if factors_path is None:
            return None
        if self.lora_adapter is None:
            raise ValueError("bayesian_lora_factors requires lora_adapter to be set.")

        factors_blob = torch.load(Path(factors_path), map_location=self.device)
        factors = (
            factors_blob.get("factors", factors_blob)
            if isinstance(factors_blob, dict)
            else factors_blob
        )
        if not isinstance(factors, dict):
            raise ValueError("Bayesian-LoRA factors file must contain a factors dictionary.")
        return factors

    @property
    def uses_bayesian_lora(self) -> bool:
        return self.bayesian_lora_factors is not None

    def infer_lora_rank(self) -> int:
        if self.bayesian_lora_n_lora is not None:
            return self.bayesian_lora_n_lora

        peft_config = getattr(self.model, "peft_config", None)
        if peft_config:
            first_config = next(iter(peft_config.values()))
            rank = getattr(first_config, "r", None)
            if rank is not None:
                return int(rank)

        for name, parameter in self.model.named_parameters():
            if "lora" in name.lower() and parameter.ndim == 2:
                return int(min(parameter.shape))
        raise ValueError("Could not infer LoRA rank. Pass bayesian_lora_n_lora explicitly.")

    def infer_kfac_rank(
        self,
        factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
        n_lora: int,
    ) -> int:
        if self.bayesian_lora_n_kfac is not None:
            return self.bayesian_lora_n_kfac

        for factor_pair in factors.values():
            for factor in factor_pair:
                if factor.ndim >= 2 and factor.shape[-2:] != (n_lora, n_lora):
                    return int(factor.shape[-1])
        raise ValueError("Could not infer K-FAC rank. Pass bayesian_lora_n_kfac explicitly.")

    def set_dropout_probability(self, probability: float = 0.1) -> int:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {probability}.")

        count = 0
        for module in self.dropout_modules():
            module.p = probability
            count += 1
        return count

    def dropout_modules(self) -> list[nn.Module]:
        dropout_types = (
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.AlphaDropout,
            nn.FeatureAlphaDropout,
        )
        return [module for module in self.model.modules() if isinstance(module, dropout_types)]

    def set_dropout_train_mode(self, enabled: bool = True) -> int:
        count = 0
        for module in self.dropout_modules():
            module.train(enabled)
            count += 1
        return count

    def generation_inputs(self, pixel_values: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        pixel_values = pixel_values.to(self.device)
        inputs = {"pixel_values": pixel_values}
        if prefix:
            text_inputs = self.processor.tokenizer(
                [prefix] * pixel_values.shape[0],
                return_tensors="pt",
                padding=True,
            )
            inputs.update({key: value.to(self.device) for key, value in text_inputs.items()})
        return inputs

    def generate_captions(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 30,
        prefix: str = DEFAULT_CAPTION_PREFIX,
    ) -> list[str]:
        generation_inputs = self.generation_inputs(pixel_values, prefix)

        with torch.no_grad():
            output_ids = self.model.generate(
                **generation_inputs,
                max_new_tokens=max_new_tokens,
            )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)

    def laplace_lora_topk_logits(
        self,
        pixel_values: torch.Tensor,
        prefix: str = DEFAULT_CAPTION_PREFIX,
    ) -> dict[str, list]:
        if self.bayesian_lora_factors is None:
            raise RuntimeError("Bayesian-LoRA factors are not loaded.")
        if self.bayesian_lora_top_k < 1:
            raise ValueError("bayesian_lora_top_k must be at least 1.")

        try:
            from bayesian_lora import variance
            from bayesian_lora.main import jacobian_mean
        except ImportError as error:
            raise RuntimeError("Install bayesian-lora to use Laplace-LoRA uncertainty.") from error

        self.model.eval()
        batch_inputs = BatchEncoding(self.generation_inputs(pixel_values, prefix))
        if "input_ids" not in batch_inputs:
            raise ValueError("Laplace-LoRA uncertainty requires a non-empty text prefix.")

        with torch.no_grad():
            logits = self.model(**batch_inputs).logits[:, -1, :].float()
            topk = torch.topk(
                logits,
                k=min(self.bayesian_lora_top_k, logits.shape[-1]),
                dim=-1,
            )
            token_ids = topk.indices.detach()

        def output_callback(outputs: object) -> torch.Tensor:
            return outputs.logits[:, -1, :].float().gather(1, token_ids)

        with torch.enable_grad():
            jacobian, mu = jacobian_mean(
                self.model,
                batch_inputs,
                output_callback=output_callback,
            )
            n_lora = self.infer_lora_rank()
            n_kfac = self.infer_kfac_rank(self.bayesian_lora_factors, n_lora)
            prior_var = torch.tensor(
                self.bayesian_lora_prior_var,
                device=self.device,
                dtype=mu.dtype,
            )
            covariance = variance(
                batch_inputs,
                jacobian,
                self.bayesian_lora_factors,
                prior_var,
                token_ids.shape[1],
                n_lora,
                n_kfac,
                str(self.device),
            )

        sigma = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(0.0))
        token_ids_cpu = token_ids.detach().cpu()
        tokens = [
            self.processor.tokenizer.convert_ids_to_tokens(row)
            for row in token_ids_cpu.tolist()
        ]
        sigma_cpu = sigma.detach().cpu()

        return {
            "token_ids": token_ids_cpu.tolist(),
            "tokens": tokens,
            "mu": mu.detach().cpu().tolist(),
            "sigma": sigma_cpu.tolist(),
            "uncertainty": sigma_cpu.mean(dim=1).tolist(),
        }

    def mc_dropout_predictive_entropy(
        self,
        pixel_values: torch.Tensor,
        num_samples: int,
        max_new_tokens: int = 2,
        prefix: str = DEFAULT_CAPTION_PREFIX,
    ) -> dict[str, torch.Tensor | list[str]]:
        if num_samples < 1:
            raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

        self.model.eval()
        dropout_count = self.set_dropout_train_mode(True)
        if dropout_count == 0:
            raise RuntimeError("No Dropout modules found for MC-dropout inference.")

        probability_sum = None
        sample_captions = []
        generated_steps = None

        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model.generate(
                    **self.generation_inputs(pixel_values, prefix),
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                probabilities = torch.stack(
                    [torch.softmax(score.float(), dim=-1) for score in outputs.scores],
                    dim=1,
                )
                probability_sum = (
                    probabilities
                    if probability_sum is None
                    else probability_sum + probabilities
                )
                generated_steps = probabilities.shape[1]
                sample_captions.append(
                    self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
                )

        self.model.eval()

        mean_probabilities = probability_sum / num_samples
        token_entropy = -torch.sum(
            mean_probabilities * torch.log(mean_probabilities.clamp_min(1e-12)),
            dim=-1,
        )
        caption_uncertainty = token_entropy.mean(dim=1)

        return {
            "caption_uncertainty": caption_uncertainty.detach().cpu(),
            "token_entropy": token_entropy.detach().cpu(),
            "sample_captions": sample_captions,
            "generated_steps": torch.tensor(generated_steps),
        }

    def vision_outputs(self, pixel_values: torch.Tensor) -> object:
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            return self.model.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )

    def vision_embedding(
        self,
        pixel_values: torch.Tensor,
        output: Literal["last_hidden_state", "pooler_output"] = "last_hidden_state",
    ) -> torch.Tensor:
        outputs = self.vision_outputs(pixel_values)
        embedding = getattr(outputs, output)
        return embedding.detach().cpu()
