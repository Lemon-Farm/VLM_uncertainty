from typing import Literal

import torch
from torch import nn
from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_BLIP_CHECKPOINT = "Salesforce/blip-image-captioning-base"
DEFAULT_CAPTION_PREFIX = "Answer with exactly one word. This is an image of"


class BLIPWrapper(nn.Module):
    def __init__(
        self,
        checkpoint: str = DEFAULT_BLIP_CHECKPOINT,
        device: str | None = None,
        extract_vision_embeddings: bool = False,
        force_dropout_prob: float | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extract_vision_embeddings = extract_vision_embeddings
        self.processor = BlipProcessor.from_pretrained(checkpoint)
        self.model = BlipForConditionalGeneration.from_pretrained(checkpoint).to(self.device)
        if force_dropout_prob is not None:
            self.set_dropout_probability(force_dropout_prob)
        self.model.eval()

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
