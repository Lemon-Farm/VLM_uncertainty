from pathlib import Path
from types import SimpleNamespace
import warnings

import torch
from transformers import BatchEncoding
from PIL import Image
from torch import nn
from transformers import BlipForQuestionAnswering, BlipProcessor


class BLIPVQAWrapper(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        lora_adapter: str | None = None,
        bayesian_lora_factors: str | Path | None = None,
        bayesian_lora_prior_var: float = 1.0,
        bayesian_lora_top_k: int = 5,
        bayesian_lora_batch_size: int = 2,
        bayesian_lora_token_step: int = 1,
        bayesian_lora_target_keywords: tuple[str, ...] = (
            "text_decoder&query.lora",
            "text_decoder&value.lora",
        ),
        bayesian_lora_n_lora: int | None = None,
        bayesian_lora_n_kfac: int | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lora_adapter = lora_adapter
        self.bayesian_lora_prior_var = bayesian_lora_prior_var
        self.bayesian_lora_top_k = bayesian_lora_top_k
        self.bayesian_lora_batch_size = bayesian_lora_batch_size
        self.bayesian_lora_token_step = bayesian_lora_token_step
        self.bayesian_lora_target_keywords = bayesian_lora_target_keywords
        self.bayesian_lora_n_lora = bayesian_lora_n_lora
        self.bayesian_lora_n_kfac = bayesian_lora_n_kfac
        self.processor = BlipProcessor.from_pretrained(checkpoint)
        self.model = self.load_model(checkpoint, lora_adapter).to(self.device)
        self.bayesian_lora_factors = self.load_bayesian_lora_factors(bayesian_lora_factors)
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

        model = BlipForQuestionAnswering.from_pretrained(checkpoint, **model_kwargs)
        if lora_adapter is None:
            return model

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter, is_trainable=True)
        trainable_lora_count = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = self.is_target_lora_parameter(name)
            trainable_lora_count += int(parameter.requires_grad)
        if trainable_lora_count == 0:
            raise ValueError(
                "No LoRA parameters matched bayesian_lora_target_keywords="
                f"{self.bayesian_lora_target_keywords}."
            )
        return model

    def is_target_lora_parameter(self, name: str) -> bool:
        name = name.lower()
        return "lora" in name and self.matches_lora_target(name)

    def matches_lora_target(self, name: str) -> bool:
        name = name.lower()
        for pattern in self.bayesian_lora_target_keywords:
            parts = [part.strip().lower() for part in pattern.split("&") if part.strip()]
            if parts and all(part in name for part in parts):
                return True
        return False

    def target_lora_module_names(self) -> list[str]:
        return [
            name
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear) and self.matches_lora_target(name)
        ]

    def load_bayesian_lora_factors(
        self,
        factors_path: str | Path | None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]] | None:
        if factors_path is None:
            return None
        if self.lora_adapter is None:
            raise ValueError("bayesian_lora_factors requires lora_adapter to be set.")

        factors_blob = torch.load(Path(factors_path), map_location=self.device)
        if isinstance(factors_blob, dict):
            factor_token_step = factors_blob.get("token_step")
            if factor_token_step is None:
                warnings.warn(
                    "Bayesian-LoRA factors do not include token_step metadata. "
                    "Recompute factors after changing bayesian_lora_token_step.",
                    stacklevel=2,
                )
            elif int(factor_token_step) != self.bayesian_lora_token_step:
                raise ValueError(
                    "Bayesian-LoRA factors were prepared for "
                    f"token_step={factor_token_step}, but inference uses "
                    f"token_step={self.bayesian_lora_token_step}."
                )
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

    def decoder_start_token_id(self) -> int:
        token_id = getattr(self.model.config.text_config, "bos_token_id", None)
        if token_id is None:
            token_id = self.processor.tokenizer.bos_token_id
        if token_id is None:
            token_id = self.processor.tokenizer.cls_token_id
        if token_id is None:
            raise ValueError("Could not infer decoder start token id.")
        return int(token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )
        question_outputs = model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
        answer_outputs = model.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True,
            reduction="mean",
        )
        return SimpleNamespace(logits=answer_outputs.logits)

    def question_inputs(
        self,
        pixel_values: torch.Tensor,
        question: str,
    ) -> dict[str, torch.Tensor]:
        text_inputs = self.processor.tokenizer(
            [question] * pixel_values.shape[0],
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            "pixel_values": pixel_values.to(self.device),
            **{key: value.to(self.device) for key, value in text_inputs.items()},
        }
        return inputs

    def inputs_for_token_step(
        self,
        pixel_values: torch.Tensor,
        question: str,
        token_step: int,
    ) -> BatchEncoding:
        if token_step < 1:
            raise ValueError(f"token_step must be at least 1, got {token_step}.")

        inputs = BatchEncoding(self.question_inputs(pixel_values, question))
        decoder_input_ids = torch.full(
            (pixel_values.shape[0], 1),
            self.decoder_start_token_id(),
            dtype=torch.long,
            device=self.device,
        )
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        inputs["decoder_input_ids"] = decoder_input_ids
        inputs["decoder_attention_mask"] = decoder_attention_mask

        for _ in range(token_step - 1):
            with torch.no_grad():
                next_token = self(**inputs).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            inputs["decoder_input_ids"] = torch.cat([inputs["decoder_input_ids"], next_token], dim=1)
            next_attention = torch.ones_like(next_token)
            inputs["decoder_attention_mask"] = torch.cat(
                [inputs["decoder_attention_mask"], next_attention],
                dim=1,
            )

        return inputs

    def image_question_inputs(
        self,
        image_path: str | Path,
        question: str,
    ) -> dict[str, torch.Tensor]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, question, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    def answer(
        self,
        pixel_values: torch.Tensor,
        question: str,
        max_new_tokens: int = 10,
    ) -> list[str]:
        with torch.no_grad():
            output_ids = self.model.generate(
                **self.question_inputs(pixel_values, question),
                max_new_tokens=max_new_tokens,
            )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)

    def answer_image(
        self,
        image_path: str | Path,
        question: str,
        max_new_tokens: int = 10,
    ) -> str:
        with torch.no_grad():
            output_ids = self.model.generate(
                **self.image_question_inputs(image_path, question),
                max_new_tokens=max_new_tokens,
            )
        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def softmax_entropy(
        self,
        pixel_values: torch.Tensor,
        question: str,
        max_new_tokens: int = 10,
    ) -> dict[str, torch.Tensor | list[str]]:
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}.")

        return self.softmax_entropy_from_inputs(
            self.question_inputs(pixel_values, question),
            max_new_tokens,
        )

    def softmax_entropy_from_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        max_new_tokens: int,
    ) -> dict[str, torch.Tensor | list[str]]:
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
            probabilities = torch.stack(
                [torch.softmax(score.float(), dim=-1) for score in outputs.scores],
                dim=1,
            )
            token_entropy = -torch.sum(
                probabilities * torch.log(probabilities.clamp_min(1e-12)),
                dim=-1,
            )
            answer_uncertainty = token_entropy.mean(dim=1)
            answers = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)

        return {
            "answer_uncertainty": answer_uncertainty.detach().cpu(),
            "token_entropy": token_entropy.detach().cpu(),
            "answers": answers,
            "generated_steps": torch.tensor(probabilities.shape[1]),
        }

    def softmax_entropy_image(
        self,
        image_path: str | Path,
        question: str,
        max_new_tokens: int = 10,
    ) -> dict[str, torch.Tensor | list[str]]:
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}.")

        return self.softmax_entropy_from_inputs(
            self.image_question_inputs(image_path, question),
            max_new_tokens,
        )

    def laplace_lora_topk_logits(
        self,
        pixel_values: torch.Tensor,
        question: str,
    ) -> dict[str, list]:
        if self.bayesian_lora_batch_size < 2:
            raise ValueError("bayesian_lora_batch_size must be at least 2 for VQA Laplace-LoRA.")
        if pixel_values.shape[0] > self.bayesian_lora_batch_size:
            merged = {
                "token_ids": [],
                "tokens": [],
                "mu": [],
                "sigma": [],
                "uncertainty": [],
            }
            for start in range(0, pixel_values.shape[0], self.bayesian_lora_batch_size):
                end = start + self.bayesian_lora_batch_size
                chunk = self.laplace_lora_topk_logits(pixel_values[start:end], question)
                for key, value in chunk.items():
                    merged[key].extend(value)
            return merged
        if pixel_values.shape[0] == 1:
            duplicated = torch.cat([pixel_values, pixel_values], dim=0)
            duplicated_output = self.laplace_lora_topk_logits(duplicated, question)
            return {key: value[:1] for key, value in duplicated_output.items()}

        if self.bayesian_lora_factors is None:
            raise RuntimeError("Bayesian-LoRA factors are not loaded.")
        if self.bayesian_lora_top_k < 2:
            raise ValueError("bayesian_lora_top_k must be at least 2 for Laplace-LoRA.")

        try:
            from bayesian_lora import variance
            from bayesian_lora.main import jacobian_mean
        except ImportError as error:
            raise RuntimeError("Install bayesian-lora to use Laplace-LoRA uncertainty.") from error

        self.model.eval()
        batch_inputs = self.inputs_for_token_step(
            pixel_values,
            question,
            self.bayesian_lora_token_step,
        )

        with torch.no_grad():
            logits = self(**batch_inputs).logits[:, -1, :].float()
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
                self,
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

        variance_diag = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(0.0)
        sigma = torch.sqrt(variance_diag)
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
            "uncertainty": sigma_cpu[:, 0].tolist(),
        }
