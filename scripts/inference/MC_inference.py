import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from vlm_uncertainty.data.blip_inputs import build_blip_image_dataloader
from vlm_uncertainty.data.imagenet import load_downloaded_imagenet
from vlm_uncertainty.models.blip import BLIPWrapper, DEFAULT_CAPTION_PREFIX


DATASET_DIR = Path("data/output/ninco/textures")
OUTPUT_PATH = Path("outputs/mc_dropout/mc_dropout_uncertainty.csv")
CHECKPOINT = "Salesforce/blip-image-captioning-base"
PREFIX = DEFAULT_CAPTION_PREFIX
MAX_NEW_TOKENS = 2
DROPOUT_PROB = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLIP MC-dropout uncertainty inference.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument(
        "--lora-adapter",
        default=None,
        help="Optional LoRA adapter path or Hugging Face repo id.",
    )
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_downloaded_imagenet(args.dataset_dir)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataloader = build_blip_image_dataloader(
        dataset=dataset,
        checkpoint=args.checkpoint,
        image_key=args.image_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = BLIPWrapper(
        checkpoint=args.checkpoint,
        device=args.device,
        force_dropout_prob=DROPOUT_PROB,
        lora_adapter=args.lora_adapter,
    )

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["index", "caption_uncertainty", "token_entropy_0", "token_entropy_1"])

        for batch in tqdm(dataloader, desc="MC-dropout inference", total=len(dataloader)):
            uncertainty = model.mc_dropout_predictive_entropy(
                pixel_values=batch["pixel_values"],
                num_samples=args.mc_samples,
                max_new_tokens=MAX_NEW_TOKENS,
                prefix=PREFIX,
            )
            caption_uncertainty = uncertainty["caption_uncertainty"].tolist()
            token_entropy = uncertainty["token_entropy"].tolist()

            for index, sample_uncertainty, sample_token_entropy in zip(
                batch["indices"].tolist(),
                caption_uncertainty,
                token_entropy,
            ):
                writer.writerow([index, sample_uncertainty, *sample_token_entropy])

    print(f"prefix: {PREFIX}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"dropout probability: {DROPOUT_PROB}")
    print(f"mc samples: {args.mc_samples}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
