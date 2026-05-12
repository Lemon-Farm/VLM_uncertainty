import argparse

from vlm_uncertainty.data.blip_inputs import build_blip_image_dataloader
from vlm_uncertainty.data.imagenet import load_downloaded_imagenet
from vlm_uncertainty.evaluation.runner import run_caption_inference
from vlm_uncertainty.models.blip import BLIPWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLIP captioning on a saved ImageNet dataset.")
    parser.add_argument(
        "--dataset-dir",
        default="data/interim/imagenet/train",
        help="Path to a dataset saved by scripts/prepare_imagenet.py.",
    )
    parser.add_argument("--output", default="outputs/captions.jsonl")
    parser.add_argument("--checkpoint", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--device", default=None, help="Device to use, e.g. cpu or cuda.")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_downloaded_imagenet(args.dataset_dir)
    dataloader = build_blip_image_dataloader(
        dataset=dataset,
        checkpoint=args.checkpoint,
        image_key=args.image_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = BLIPWrapper(checkpoint=args.checkpoint, device=args.device)
    run_caption_inference(
        model=model,
        dataloader=dataloader,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
