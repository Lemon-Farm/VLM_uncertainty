from typing import Any

import torch
from datasets import Dataset
from PIL import Image
from transformers import BlipProcessor


class BlipImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        processor: BlipProcessor,
        prompt: str | None = None,
        image_key: str = "image",
    ) -> None:
        self.dataset = dataset
        self.processor = processor
        self.prompt = prompt
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = sample[self.image_key].convert("RGB")
        inputs = self._process_image(image)

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "index": index,
        }

    def _process_image(self, image: Image.Image) -> dict[str, torch.Tensor]:
        if self.prompt is None:
            return self.processor(image, return_tensors="pt")
        return self.processor(image, self.prompt, return_tensors="pt")


def build_blip_image_dataset(
    dataset: Dataset,
    checkpoint: str = "Salesforce/blip-image-captioning-base",
    prompt: str | None = None,
    image_key: str = "image",
) -> BlipImageDataset:
    processor = BlipProcessor.from_pretrained(checkpoint)
    return BlipImageDataset(
        dataset=dataset,
        processor=processor,
        prompt=prompt,
        image_key=image_key,
    )


def collate_blip_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "indices": torch.tensor([sample["index"] for sample in samples], dtype=torch.long),
    }
