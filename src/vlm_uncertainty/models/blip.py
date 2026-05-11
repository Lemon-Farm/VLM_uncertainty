from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_BLIP_CHECKPOINT = "Salesforce/blip-image-captioning-base"


class BlipCaptioner:
    def __init__(
        self,
        checkpoint: str = DEFAULT_BLIP_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = BlipProcessor.from_pretrained(checkpoint)
        self.model = BlipForConditionalGeneration.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    def caption(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        max_new_tokens: int = 30,
    ) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return self.processor.decode(output_ids[0], skip_special_tokens=True)
