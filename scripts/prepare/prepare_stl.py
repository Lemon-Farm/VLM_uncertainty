import argparse
import os
from pathlib import Path

from vlm_uncertainty.data.stl import STL10_DATASET_ID, download_stl10


DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = "data/output/stl10"
DEFAULT_MAX_SAMPLES = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download STL-10 and save it locally as Arrow.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or args.split
    output_dir = Path(args.output_dir) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = download_stl10(
        split=args.split,
        cache_dir=args.cache_dir,
        token=args.hf_token or os.getenv("HF_TOKEN"),
        max_samples=args.max_samples,
    )
    dataset.save_to_disk(str(output_dir))
    print(f"dataset: {STL10_DATASET_ID}")
    print(f"saved {len(dataset)} rows to {output_dir}")


if __name__ == "__main__":
    main()
