import argparse
from pathlib import Path

import torch

from vlm_uncertainty.data.blip_inputs import build_blip_image_dataloader
from vlm_uncertainty.data.imagenet import load_downloaded_imagenet
from vlm_uncertainty.models.blip_vqa import BLIPVQAWrapper


DEFAULT_DATASET_DIR = Path("data/output/imagenet/train_3K")
DEFAULT_OUTPUT = Path("outputs/bayesian_lora_vqa/kronecker_factors.pt")
DEFAULT_CHECKPOINT = "Salesforce/blip-vqa-base"
DEFAULT_LORA_ADAPTER = "sohith18/blip-lora-vqa"
DEFAULT_QUESTION = "What is in the image?"
DEFAULT_TARGET_MODULE_KEYWORDS = ("text_decoder&query.lora", "text_decoder&value.lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BLIP VQA Bayesian-LoRA K-FAC factors.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lora-adapter", default=DEFAULT_LORA_ADAPTER)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--token-step", type=int, default=1)
    parser.add_argument("--target-module-keywords", default=",".join(DEFAULT_TARGET_MODULE_KEYWORDS))
    parser.add_argument("--n-kfac", type=int, default=10)
    parser.add_argument("--lr-threshold", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2 for Bayesian-LoRA K-FAC factors.")
    target_module_patterns = tuple(
        keyword.strip() for keyword in args.target_module_keywords.split(",") if keyword.strip()
    )
    if not target_module_patterns:
        raise ValueError("--target-module-keywords must contain at least one keyword.")

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
    wrapper = BLIPVQAWrapper(
        checkpoint=args.checkpoint,
        device=args.device,
        lora_adapter=args.lora_adapter,
        bayesian_lora_target_keywords=target_module_patterns,
    )
    target_module_keywords = wrapper.target_lora_module_names()
    if not target_module_keywords:
        raise ValueError(
            "No LoRA modules matched --target-module-keywords="
            f"{target_module_patterns}."
        )

    try:
        from bayesian_lora import calculate_kronecker_factors
        from bayesian_lora import cholesky_decompose_small_factors
    except ImportError as error:
        raise RuntimeError("Install bayesian-lora to prepare K-FAC factors.") from error

    def forward_call(model: torch.nn.Module, batch: dict) -> torch.Tensor:
        inputs = wrapper.inputs_for_token_step(
            pixel_values=batch["pixel_values"],
            question=args.question,
            token_step=args.token_step,
        )
        logits = model(**inputs).logits[:, -1, :].float()
        topk_logits = torch.topk(
            logits,
            k=min(args.top_k, logits.shape[-1]),
            dim=-1,
        ).values
        return topk_logits.softmax(dim=-1)

    factors = calculate_kronecker_factors(
        wrapper,
        forward_call,
        dataloader,
        n_kfac=args.n_kfac,
        lr_threshold=args.lr_threshold,
        target_module_keywords=list(target_module_keywords),
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
            "token_step": args.token_step,
            "checkpoint": args.checkpoint,
            "lora_adapter": args.lora_adapter,
            "question": args.question,
            "target_module_patterns": list(target_module_patterns),
            "target_module_keywords": list(target_module_keywords),
        },
        output_path,
    )
    print(f"saved BLIP VQA Bayesian-LoRA factors: {output_path}")


if __name__ == "__main__":
    main()
