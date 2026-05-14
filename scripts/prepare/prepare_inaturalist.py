import argparse
import mimetypes
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import Dataset, Features, Image, Value, load_dataset
from tqdm import tqdm


DATASET_ID = "ba188/inaturalist"
RAW_IMAGE_DIR = Path("data/raw/inaturalist")
OUTPUT_DIR = Path("data/output/inaturalist")
MAX_IMAGES = 3000
PHOTO_COLUMN = "photos"
USER_AGENT = "vlm-uncertainty/0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare iNaturalist images from URL metadata.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env var.")
    return parser.parse_args()


def medium_image_url(url: str) -> str:
    return url.replace("square", "medium")


def image_suffix(url: str, content_type: str | None) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix
    if suffix:
        return suffix
    if content_type:
        return mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) or ".jpg"
    return ".jpg"


def download_image(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        output_path.write_bytes(response.read())


def main() -> None:
    args = parse_args()
    RAW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = load_dataset(
        DATASET_ID,
        split=args.split,
        cache_dir=args.cache_dir,
        token=args.hf_token or os.getenv("HF_TOKEN"),
        streaming=True,
    )

    rows = []
    progress = tqdm(total=MAX_IMAGES, desc="Downloading iNaturalist images")
    for source_index, sample in enumerate(source):
        if source_index >= MAX_IMAGES:
            break

        original_url = sample[PHOTO_COLUMN]
        url = medium_image_url(original_url)
        temp_path = RAW_IMAGE_DIR / f"{len(rows):06d}.download"

        try:
            download_image(url, temp_path)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            temp_path.unlink(missing_ok=True)
            tqdm.write(f"skip source_index={source_index}: {url} ({error})")
            progress.update(1)
            continue

        suffix = image_suffix(url, None)
        image_path = RAW_IMAGE_DIR / f"{len(rows):06d}{suffix}"
        temp_path.replace(image_path)

        rows.append(
            {
                "image": str(image_path),
                "source_index": source_index,
                "original_url": original_url,
                "image_url": url,
                "quality_grade": sample.get("quality_grade"),
                "species_guess": sample.get("species_guess"),
                "taxon": sample.get("taxon"),
            }
        )
        progress.update(1)

    progress.close()

    features = Features(
        {
            "image": Image(),
            "source_index": Value("int64"),
            "original_url": Value("string"),
            "image_url": Value("string"),
            "quality_grade": Value("string"),
            "species_guess": Value("string"),
            "taxon": Value("string"),
        }
    )
    dataset = Dataset.from_list(rows, features=features)
    dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"saved raw images: {RAW_IMAGE_DIR}")
    print(f"saved dataset: {OUTPUT_DIR}")
    print(f"num images: {len(dataset)}")


if __name__ == "__main__":
    main()
