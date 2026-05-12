import argparse
from pathlib import Path

from datasets import Dataset, Features, Image, Value


RAW_DIR = Path("data/raw/NINCO/NINCO/NINCO_OOD_classes")
OUTPUT_DIR = Path("data/output/ninco/")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NINCO Textures as a local Arrow dataset, including subdirectories."
    )
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args()


def texture_label(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return ""
    return "_".join(parts[2:-1])


def image_paths(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths(raw_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under: {raw_dir}")

    rows = [
        {
            "image": str(path),
            "source_path": str(path),
            "relative_path": str(path.relative_to(raw_dir)),
            "filename": path.name,
            "label": texture_label(path),
            "source_index": index,
        }
        for index, path in enumerate(paths)
    ]

    features = Features(
        {
            "image": Image(),
            "source_path": Value("string"),
            "relative_path": Value("string"),
            "filename": Value("string"),
            "label": Value("string"),
            "source_index": Value("int64"),
        }
    )
    dataset = Dataset.from_list(rows, features=features)
    dataset.save_to_disk(str(output_dir))

    print(f"saved dataset: {output_dir}")
    print(f"num images: {len(dataset)}")


if __name__ == "__main__":
    main()
