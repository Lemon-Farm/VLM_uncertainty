from typing import Literal

import torch
from torch import nn
from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_BLIP_CHECKPOINT = "Salesforce/blip-image-captioning-base"


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

        dropout_types = (
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.AlphaDropout,
            nn.FeatureAlphaDropout,
        )
        count = 0
        for module in self.model.modules():
            if isinstance(module, dropout_types):
                module.p = probability
                count += 1
        return count

    def generate_captions(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 30,
    ) -> list[str]:
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
            )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)

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
