from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Image, IterableDataset, Value
from datasets import load_dataset, load_from_disk


STL10_DATASET_ID = "galilai-group/stl10"
STL10_CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def load_stl10(
    split: str = "train",
    cache_dir: str | Path | None = None,
    token: str | None = None,
    streaming: bool = False,
) -> Dataset | IterableDataset:
    return load_dataset(
        STL10_DATASET_ID,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
        streaming=streaming,
    )


def download_stl10(
    split: str = "train",
    cache_dir: str | Path | None = None,
    token: str | None = None,
    max_samples: int | None = None,
) -> Dataset:
    try:
        dataset = load_stl10(
            split=split,
            cache_dir=cache_dir,
            token=token,
        )
    except RuntimeError as error:
        if "Dataset scripts are no longer supported" not in str(error):
            raise
        dataset = download_stl10_from_torchvision(
            split=split,
            root=Path(cache_dir) if cache_dir is not None else Path("data/raw/stl10"),
        )

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def download_stl10_from_torchvision(
    split: str = "train",
    root: str | Path = "data/raw/stl10",
) -> Dataset:
    from torchvision.datasets import STL10

    torchvision_dataset = STL10(
        root=str(root),
        split=split,
        download=True,
    )
    samples = [
        {
            "image": image,
            "label": int(label),
        }
        for image, label in torchvision_dataset
    ]
    label_feature = (
        Value("int64")
        if split == "unlabeled"
        else ClassLabel(names=STL10_CLASS_NAMES)
    )
    features = Features(
        {
            "image": Image(),
            "label": label_feature,
        }
    )
    return Dataset.from_list(samples, features=features)


def load_downloaded_stl10(dataset_dir: str | Path) -> Dataset:
    return load_from_disk(str(dataset_dir))
