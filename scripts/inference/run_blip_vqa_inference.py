import argparse
import json
from pathlib import Path

from tqdm import tqdm

from vlm_uncertainty.data.blip_inputs import build_blip_image_dataloader
from vlm_uncertainty.data.imagenet import load_downloaded_imagenet
from vlm_uncertainty.models.blip_vqa import BLIPVQAWrapper


dataset = ("data/output/imagenet/train_3K", "data/output/ninco/textures")

# Base
DEFAULT_DATA_PATH = dataset[0]
DEFAULT_OUTPUT_PATH = "outputs/blip_vqa_answers.jsonl"
DEFAULT_CHECKPOINT = "Salesforce/blip-vqa-base"
DEFAULT_LORA_ADAPTER = "sohith18/blip-lora-vqa"
DEFAULT_IMAGE_KEY = "image"
DEFAULT_QUESTION = "What is in the image?"
DEFAULT_MAX_NEW_TOKENS = 1
DEFAULT_DEVICE = None
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0

# Softmax entropy
DEFAULT_COMPUTE_SOFTMAX_ENTROPY = False

# Laplace-LoRA
DEFAULT_BAYESIAN_LORA_FACTORS = "outputs/bayesian_lora_vqa/kronecker_factors.pt"
DEFAULT_BAYESIAN_LORA_PRIOR_VAR = 1.0
DEFAULT_BAYESIAN_LORA_TOP_K = 2
DEFAULT_BAYESIAN_LORA_TOKEN_STEP = 1
DEFAULT_BAYESIAN_LORA_BATCH_SIZE = 16
DEFAULT_BAYESIAN_LORA_TARGET_KEYWORDS = "text_decoder&query.lora,text_decoder&value.lora"
DEFAULT_BAYESIAN_LORA_N_LORA = None
DEFAULT_BAYESIAN_LORA_N_KFAC = None


def zero_one_to_bool(value: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise argparse.ArgumentTypeError("Expected 0 or 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BLIP VQA inference on a saved ImageNet-style dataset."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lora-adapter", default=DEFAULT_LORA_ADAPTER)
    parser.add_argument("--image-key", default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--compute-softmax-entropy",
        type=zero_one_to_bool,
        default=DEFAULT_COMPUTE_SOFTMAX_ENTROPY,
    )
    parser.add_argument("--bayesian-lora-factors", default=DEFAULT_BAYESIAN_LORA_FACTORS)
    parser.add_argument("--bayesian-lora-prior-var", type=float, default=DEFAULT_BAYESIAN_LORA_PRIOR_VAR)
    parser.add_argument("--bayesian-lora-top-k", type=int, default=DEFAULT_BAYESIAN_LORA_TOP_K)
    parser.add_argument("--bayesian-lora-token-step", type=int, default=DEFAULT_BAYESIAN_LORA_TOKEN_STEP)
    parser.add_argument("--bayesian-lora-batch-size", type=int, default=DEFAULT_BAYESIAN_LORA_BATCH_SIZE)
    parser.add_argument("--bayesian-lora-target-keywords", default=DEFAULT_BAYESIAN_LORA_TARGET_KEYWORDS)
    parser.add_argument("--bayesian-lora-n-lora", type=int, default=DEFAULT_BAYESIAN_LORA_N_LORA)
    parser.add_argument("--bayesian-lora-n-kfac", type=int, default=DEFAULT_BAYESIAN_LORA_N_KFAC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bayesian_lora_target_keywords = tuple(
        keyword.strip() for keyword in args.bayesian_lora_target_keywords.split(",") if keyword.strip()
    )
    if not bayesian_lora_target_keywords:
        raise ValueError("--bayesian-lora-target-keywords must contain at least one keyword.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_downloaded_imagenet(args.dataset_dir)
    dataloader = build_blip_image_dataloader(
        dataset=dataset,
        checkpoint=args.checkpoint,
        image_key=args.image_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = BLIPVQAWrapper(
        checkpoint=args.checkpoint,
        device=args.device,
        lora_adapter=args.lora_adapter,
        bayesian_lora_factors=args.bayesian_lora_factors,
        bayesian_lora_prior_var=args.bayesian_lora_prior_var,
        bayesian_lora_top_k=args.bayesian_lora_top_k,
        bayesian_lora_token_step=args.bayesian_lora_token_step,
        bayesian_lora_batch_size=args.bayesian_lora_batch_size,
        bayesian_lora_target_keywords=bayesian_lora_target_keywords,
        bayesian_lora_n_lora=args.bayesian_lora_n_lora,
        bayesian_lora_n_kfac=args.bayesian_lora_n_kfac,
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for batch in tqdm(dataloader, desc="BLIP VQA inference", total=len(dataloader)):
            softmax_entropy = None
            if args.compute_softmax_entropy:
                softmax_entropy = model.softmax_entropy(
                    pixel_values=batch["pixel_values"],
                    question=args.question,
                    max_new_tokens=args.max_new_tokens,
                )
                answers = softmax_entropy["answers"]
            else:
                answers = model.answer(
                    pixel_values=batch["pixel_values"],
                    question=args.question,
                    max_new_tokens=args.max_new_tokens,
                )
            laplace_lora = None
            if model.uses_bayesian_lora:
                laplace_lora = model.laplace_lora_topk_logits(
                    pixel_values=batch["pixel_values"],
                    question=args.question,
                )

            for row, (index, answer) in enumerate(zip(batch["indices"].tolist(), answers)):
                record = {
                    "index": index,
                    "question": args.question,
                    "answer": answer,
                }
                if softmax_entropy is not None:
                    record["softmax_entropy"] = {
                        "token_entropy": softmax_entropy["token_entropy"][row].tolist(),
                        "uncertainty": softmax_entropy["answer_uncertainty"][row].item(),
                        "generated_steps": softmax_entropy["generated_steps"].item(),
                    }
                if laplace_lora is not None:
                    record["laplace_lora"] = {
                        "token_ids": laplace_lora["token_ids"][row],
                        "tokens": laplace_lora["tokens"][row],
                        "mu": laplace_lora["mu"][row],
                        "sigma": laplace_lora["sigma"][row],
                        "uncertainty": laplace_lora["uncertainty"][row],
                    }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"question: {args.question}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
