import argparse
from pathlib import Path

import torch
from transformers import BatchEncoding

from vlm_uncertainty.data.blip_inputs import build_blip_image_dataloader
from vlm_uncertainty.data.imagenet import load_downloaded_imagenet
from vlm_uncertainty.models.blip import BLIPWrapper, DEFAULT_CAPTION_PREFIX


DEFAULT_DATASET_DIR = Path("data/output/imagenet/validation")
DEFAULT_OUTPUT = Path("outputs/bayesian_lora/kronecker_factors.pt")
DEFAULT_CHECKPOINT = "Salesforce/blip-image-captioning-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Bayesian-LoRA K-FAC factors.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lora-adapter", default="diyanigam/lora-blip-finetuned")
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prefix", default=DEFAULT_CAPTION_PREFIX)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-kfac", type=int, default=10)
    parser.add_argument("--lr-threshold", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

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
    wrapper = BLIPWrapper(
        checkpoint=args.checkpoint,
        device=args.device,
        lora_adapter=args.lora_adapter,
    )

    try:
        from bayesian_lora import calculate_kronecker_factors
        from bayesian_lora import cholesky_decompose_small_factors
    except ImportError as error:
        raise RuntimeError("Install bayesian-lora to prepare K-FAC factors.") from error

    def forward_call(model: torch.nn.Module, batch: dict) -> torch.Tensor:
        inputs = BatchEncoding(
            wrapper.generation_inputs(
                pixel_values=batch["pixel_values"],
                prefix=args.prefix,
            )
        )
        if "input_ids" not in inputs:
            raise ValueError("Bayesian-LoRA factor preparation requires a non-empty prefix.")

        logits = model(**inputs).logits[:, -1, :].float()
        topk_logits = torch.topk(
            logits,
            k=min(args.top_k, logits.shape[-1]),
            dim=-1,
        ).values
        return topk_logits.softmax(dim=-1)

    factors = calculate_kronecker_factors(
        wrapper.model,
        forward_call,
        dataloader,
        n_kfac=args.n_kfac,
        lr_threshold=args.lr_threshold,
        target_module_keywords=["lora"],
        use_tqdm=True,
    )
    factors = cholesky_decompose_small_factors(
        factors,
        args.lr_threshold,
        str(wrapper.device),
        torch.float32,
    )
    torch.save(
        {
            "factors": factors,
            "n_kfac": args.n_kfac,
            "lr_threshold": args.lr_threshold,
            "top_k": args.top_k,
            "checkpoint": args.checkpoint,
            "lora_adapter": args.lora_adapter,
        },
        output_path,
    )
    print(f"saved Bayesian-LoRA factors: {output_path}")


if __name__ == "__main__":
    main()
