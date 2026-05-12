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
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = BlipProcessor.from_pretrained(checkpoint)
        self.model = BlipForConditionalGeneration.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

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
