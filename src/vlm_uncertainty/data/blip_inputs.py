from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BlipProcessor


class BlipImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        processor: BlipProcessor,
        image_key: str = "image",
    ) -> None:
        self.dataset = dataset
        self.processor = processor
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = sample[self.image_key].convert("RGB")
        inputs = self.processor(image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "index": index,
        }


def build_blip_image_dataset(
    dataset: Dataset,
    checkpoint: str = "Salesforce/blip-image-captioning-base",
    image_key: str = "image",
) -> BlipImageDataset:
    processor = BlipProcessor.from_pretrained(checkpoint)
    return BlipImageDataset(
        dataset=dataset,
        processor=processor,
        image_key=image_key,
    )


def collate_blip_images(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "indices": torch.tensor([sample["index"] for sample in samples], dtype=torch.long),
    }


def build_blip_image_dataloader(
    dataset: Dataset,
    checkpoint: str = "Salesforce/blip-image-captioning-base",
    image_key: str = "image",
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    blip_dataset = build_blip_image_dataset(
        dataset=dataset,
        checkpoint=checkpoint,
        image_key=image_key,
    )
    return DataLoader(
        blip_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_blip_images,
    )
