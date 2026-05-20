from pathlib import Path

from datasets import Dataset, Features, Image, IterableDataset, Value, load_dataset, load_from_disk


IMAGENET_DATASET_ID = "timm/mini-imagenet"
EASY_OBJECT_SYNSETS = {
    "n01532829",
    "n01558993",
    "n02099601",
    "n02108089",
    "n02108915",
    "n02110341",
    "n02113712",
    "n02795169",
    "n02823428",
    "n02971356",
    "n03127925",
    "n03272010",
    "n03400231",
    "n03417042",
    "n03544143",
    "n03676483",
    "n04146614",
    "n04509417",
    "n04522168",
    "n04596742",
    "n06794110",
    "n07697537",
    "n07747607",
    "n11939491",
    "n12057211",
}
CLASS_GROUPS = {
    "easy_objects": EASY_OBJECT_SYNSETS,
}


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
    class_group: str | None = None,
) -> Dataset:
    dataset = load_imagenet(split=split, cache_dir=cache_dir, token=token)
    if class_group is not None:
        dataset = filter_imagenet_class_group(dataset, class_group)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def filter_imagenet_class_group(dataset: Dataset, class_group: str) -> Dataset:
    if class_group not in CLASS_GROUPS:
        raise ValueError(f"Unknown class_group '{class_group}'. Available: {sorted(CLASS_GROUPS)}")

    label_feature = dataset.features.get("label")
    label_names = getattr(label_feature, "names", None)
    if not label_names:
        raise ValueError("ImageNet class-group filtering requires ClassLabel names.")

    target_synsets = CLASS_GROUPS[class_group]
    target_label_ids = {index for index, name in enumerate(label_names) if name in target_synsets}
    if not target_label_ids:
        raise ValueError(f"No labels found for class_group '{class_group}'.")

    return dataset.filter(lambda sample: sample["label"] in target_label_ids)


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
