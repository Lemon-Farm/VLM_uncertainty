import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vlm_uncertainty.data.imagenet import build_imagenet_blip_dataset, collate_blip_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ImageNet and save BLIP-ready tensor shards.")
    parser.add_argument("--split", default="validation", help="Dataset split: train, validation, or test.")
    parser.add_argument("--checkpoint", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face dataset/model cache directory.")
    parser.add_argument("--output-dir", default="data/processed/imagenet-blip")
    parser.add_argument("--prompt", default=None, help="Optional BLIP prompt for conditional captioning.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env var.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_imagenet_blip_dataset(
        checkpoint=args.checkpoint,
        split=args.split,
        cache_dir=args.cache_dir,
        token=args.hf_token or os.getenv("HF_TOKEN"),
        max_samples=args.max_samples,
        prompt=args.prompt,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_blip_batch,
    )

    for shard_index, batch in enumerate(loader):
        shard_path = output_dir / f"{args.split}_{shard_index:06d}.pt"
        torch.save(batch, shard_path)
        print(f"saved {shard_path}")


if __name__ == "__main__":
    main()
