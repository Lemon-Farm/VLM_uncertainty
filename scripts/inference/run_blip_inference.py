import argparse

from vlm_uncertainty.data.blip_inputs import build_blip_image_dataloader
from vlm_uncertainty.data.imagenet import load_downloaded_imagenet
from vlm_uncertainty.evaluation.runner import run_caption_inference
from vlm_uncertainty.models.blip import BLIPWrapper, DEFAULT_CAPTION_PREFIX

dataset = ("data/output/imagenet/validation", "data/output/ninco/textures")

# Base
DEFAULT_DATA_PATH = dataset[1]
DEFAULT_OUTPUT_PATH = "outputs/captions.jsonl"
DEFAULT_CHECKPOINT = "Salesforce/blip-image-captioning-base"
DEFAULT_IMAGE_KEY = "image"
DEFAULT_MAX_NEW_TOKENS = 3
DEFAULT_DEVICE = None
DEFAULT_PREFIX = DEFAULT_CAPTION_PREFIX
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0
DEFAULT_COMPUTE_SOFTMAX_ENTROPY = True
DEFAULT_EXCLUDE_SOFTMAX_ENTROPY_STOPWORDS = False

# Bayesian-LoRA
DEFAULT_LORA_ADAPTER = None #"diyanigam/lora-blip-finetuned"
DEFAULT_BAYESIAN_LORA_FACTORS = None #"outputs/bayesian_lora/kronecker_factors.pt"
DEFAULT_BAYESIAN_LORA_PRIOR_VAR = 1.0
DEFAULT_BAYESIAN_LORA_TOP_K = 2
DEFAULT_BAYESIAN_LORA_TOKEN_STEP = 2
DEFAULT_BAYESIAN_LORA_BATCH_SIZE = 2
DEFAULT_BAYESIAN_LORA_N_LORA = None
DEFAULT_BAYESIAN_LORA_N_KFAC = None

# Mahalanobis
DEFAULT_EXTRACT_VISION_EMBEDDINGS = False

# MC-dropout
DEFAULT_FORCE_DROPOUT_PROB = None # 0.1


def zero_one_to_bool(value: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise argparse.ArgumentTypeError("Expected 0 or 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLIP captioning on a saved ImageNet dataset.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lora-adapter", default=DEFAULT_LORA_ADAPTER)
    parser.add_argument("--bayesian-lora-factors", default=DEFAULT_BAYESIAN_LORA_FACTORS)
    parser.add_argument("--bayesian-lora-prior-var", type=float, default=DEFAULT_BAYESIAN_LORA_PRIOR_VAR)
    parser.add_argument("--bayesian-lora-top-k", type=int, default=DEFAULT_BAYESIAN_LORA_TOP_K)
    parser.add_argument("--bayesian-lora-token-step", type=int, default=DEFAULT_BAYESIAN_LORA_TOKEN_STEP)
    parser.add_argument("--bayesian-lora-batch-size", type=int, default=DEFAULT_BAYESIAN_LORA_BATCH_SIZE)
    parser.add_argument("--bayesian-lora-n-lora", type=int, default=DEFAULT_BAYESIAN_LORA_N_LORA)
    parser.add_argument("--bayesian-lora-n-kfac", type=int, default=DEFAULT_BAYESIAN_LORA_N_KFAC)
    parser.add_argument("--image-key", default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--compute-softmax-entropy", type=zero_one_to_bool, default=DEFAULT_COMPUTE_SOFTMAX_ENTROPY)
    parser.add_argument("--exclude-softmax-entropy-stopwords", type=zero_one_to_bool, default=DEFAULT_EXCLUDE_SOFTMAX_ENTROPY_STOPWORDS)
    parser.add_argument("--extract-vision-embeddings", type=zero_one_to_bool, default=DEFAULT_EXTRACT_VISION_EMBEDDINGS)
    parser.add_argument("--force-dropout-prob", type=float, default=DEFAULT_FORCE_DROPOUT_PROB)
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
        bayesian_lora_token_step=args.bayesian_lora_token_step,
        bayesian_lora_batch_size=args.bayesian_lora_batch_size,
        bayesian_lora_n_lora=args.bayesian_lora_n_lora,
        bayesian_lora_n_kfac=args.bayesian_lora_n_kfac,
    )
    run_caption_inference(
        model=model,
        dataloader=dataloader,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        prefix=args.prefix,
        compute_softmax_entropy=args.compute_softmax_entropy,
        exclude_softmax_entropy_stopwords=args.exclude_softmax_entropy_stopwords,
    )


if __name__ == "__main__":
    main()
