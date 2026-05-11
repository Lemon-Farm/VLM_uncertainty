from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, IterableDataset, load_dataset
from PIL import Image
from transformers import BlipProcessor


IMAGENET_DATASET_ID = "timm/mini-imagenet"


def load_imagenet(
    split: str = "validation",
    cache_dir: str | Path | None = None,
    token: str | None = None,
    streaming: bool = False,
) -> Dataset | IterableDataset:
    return load_dataset(
        IMAGENET_DATASET_ID,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
        streaming=streaming,
    )


class ImageNetBlipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        processor: BlipProcessor,
        prompt: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.processor = processor
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = _to_rgb(sample["image"])
        inputs = self._process_image(image)

        label = int(sample["label"])
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": label,
            "label_name": self.label_name(label),
            "index": index,
        }

    def label_name(self, label: int) -> str:
        if label < 0:
            return ""
        return self.dataset.features["label"].int2str(label)

    def _process_image(self, image: Image.Image) -> dict[str, torch.Tensor]:
        if self.prompt is None:
            return self.processor(image, return_tensors="pt")
        return self.processor(image, self.prompt, return_tensors="pt")


def build_imagenet_blip_dataset(
    checkpoint: str = "Salesforce/blip-image-captioning-base",
    split: str = "validation",
    cache_dir: str | Path | None = None,
    token: str | None = None,
    max_samples: int | None = None,
    prompt: str | None = None,
) -> ImageNetBlipDataset:
    dataset = load_imagenet(split=split, cache_dir=cache_dir, token=token)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    processor = BlipProcessor.from_pretrained(checkpoint)
    return ImageNetBlipDataset(dataset=dataset, processor=processor, prompt=prompt)


def collate_blip_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "label_names": [sample["label_name"] for sample in samples],
        "indices": torch.tensor([sample["index"] for sample in samples], dtype=torch.long),
    }


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")
