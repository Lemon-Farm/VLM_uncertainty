import argparse

from vlm_uncertainty.data.blip_inputs import build_blip_image_dataloader
from vlm_uncertainty.data.imagenet import load_downloaded_imagenet
from vlm_uncertainty.evaluation.runner import run_caption_inference
from vlm_uncertainty.models.blip import BLIPWrapper, DEFAULT_CAPTION_PREFIX

DATA_PATH = "data/output/ninco/textures"


def zero_one_to_bool(value: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise argparse.ArgumentTypeError("Expected 0 or 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLIP captioning on a saved ImageNet dataset.")
    parser.add_argument(
        "--dataset-dir",
        default=DATA_PATH,
        help="Path to a dataset saved by scripts/prepare/prepare_imagenet.py.",
    )
    parser.add_argument("--output", default="outputs/captions.jsonl")
    parser.add_argument("--checkpoint", default="Salesforce/blip-image-captioning-base")
    parser.add_argument(
        "--lora-adapter",
        default="diyanigam/lora-blip-finetuned",
        help="Optional LoRA adapter path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--bayesian-lora-factors",
        default="outputs/bayesian_lora/kronecker_factors_ID.pt",
        help="Optional path to saved Bayesian-LoRA Kronecker factors.",
    )
    parser.add_argument("--bayesian-lora-prior-var", type=float, default=1.0)
    parser.add_argument("--bayesian-lora-top-k", type=int, default=5)
    parser.add_argument("--bayesian-lora-n-lora", type=int, default=None)
    parser.add_argument("--bayesian-lora-n-kfac", type=int, default=None)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--device", default=None, help="Device to use, e.g. cpu or cuda.")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--prefix", default=DEFAULT_CAPTION_PREFIX)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--extract-vision-embeddings",
        type=zero_one_to_bool,
        default=False,
        help="Set 1 to save BLIP vision embeddings to one CSV file, 0 otherwise.",
    )
    parser.add_argument(
        "--force-dropout-prob",
        type=float,
        default=0.1,
        help="If set, force every Dropout module in BLIP to this probability, e.g. 0.1.",
    )
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
    model = BLIPWrapper(
        checkpoint=args.checkpoint,
        device=args.device,
        extract_vision_embeddings=args.extract_vision_embeddings,
        force_dropout_prob=args.force_dropout_prob,
        lora_adapter=args.lora_adapter,
        bayesian_lora_factors=args.bayesian_lora_factors,
        bayesian_lora_prior_var=args.bayesian_lora_prior_var,
        bayesian_lora_top_k=args.bayesian_lora_top_k,
        bayesian_lora_n_lora=args.bayesian_lora_n_lora,
        bayesian_lora_n_kfac=args.bayesian_lora_n_kfac,
    )
    run_caption_inference(
        model=model,
        dataloader=dataloader,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
