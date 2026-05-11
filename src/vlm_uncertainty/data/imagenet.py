from pathlib import Path

from datasets import Dataset, Features, Image, IterableDataset, Value, load_dataset, load_from_disk


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


def download_imagenet(
    split: str = "validation",
    cache_dir: str | Path | None = None,
    token: str | None = None,
    max_samples: int | None = None,
) -> Dataset:
    dataset = load_imagenet(split=split, cache_dir=cache_dir, token=token)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def download_imagenet_subset(
    split: str = "validation",
    max_samples: int = 100,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Dataset:
    dataset = load_imagenet(
        split=split,
        cache_dir=cache_dir,
        token=token,
        streaming=True,
    )
    samples = [
        {
            "image": sample["image"],
            "source_index": index,
        }
        for index, sample in enumerate(dataset.take(max_samples))
    ]
    features = Features(
        {
            "image": Image(),
            "source_index": Value("int64"),
        }
    )
    return Dataset.from_list(samples, features=features)


def load_downloaded_imagenet(dataset_dir: str | Path) -> Dataset:
    return load_from_disk(str(dataset_dir))
