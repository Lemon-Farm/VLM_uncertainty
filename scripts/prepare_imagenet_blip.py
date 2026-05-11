import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vlm_uncertainty.data.blip_inputs import build_blip_image_dataset, collate_blip_batch
from vlm_uncertainty.data.imagenet import load_downloaded_imagenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert downloaded ImageNet images to BLIP-ready tensors.")
    parser.add_argument("--split", default="validation", help="Dataset split name used in shard filenames.")
    parser.add_argument("--dataset-dir", default="data/interim/imagenet/validation")
    parser.add_argument("--checkpoint", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--output-dir", default="data/processed/imagenet-blip")
    parser.add_argument("--prompt", default=None, help="Optional BLIP prompt for conditional captioning.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_dataset = load_downloaded_imagenet(args.dataset_dir)
    dataset = build_blip_image_dataset(
        dataset=downloaded_dataset,
        checkpoint=args.checkpoint,
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
