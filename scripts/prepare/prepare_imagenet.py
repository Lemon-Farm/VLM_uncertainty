import argparse
import os
from pathlib import Path

from vlm_uncertainty.data.imagenet import download_imagenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ImageNet and save the raw dataset locally.")
    parser.add_argument("--split", default="train", help="Dataset split: train, validation, or test.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face dataset cache directory.")
    parser.add_argument("--output-dir", default="data/output/imagenet")
    parser.add_argument("--output-name", default="train_3K")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--class-group", default="easy_objects", choices=["easy_objects"])
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env var.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or (
        f"{args.split}_{args.class_group}" if args.class_group else args.split
    )
    output_dir = Path(args.output_dir) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = download_imagenet(
        split=args.split,
        cache_dir=args.cache_dir,
        token=args.hf_token or os.getenv("HF_TOKEN"),
        max_samples=args.max_samples,
        class_group=args.class_group,
    )
    dataset.save_to_disk(str(output_dir))
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
