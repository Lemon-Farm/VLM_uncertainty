import argparse

from vlm_uncertainty.models.blip import BlipCaptioner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLIP image captioning on a local image.")
    parser.add_argument("image_path", help="Path to an image file.")
    parser.add_argument("--prompt", default=None, help="Optional text prompt for conditional captioning.")
    parser.add_argument("--checkpoint", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--device", default=None, help="Device to use, e.g. cpu or cuda.")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captioner = BlipCaptioner(checkpoint=args.checkpoint, device=args.device)
    caption = captioner.caption(
        image_path=args.image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(caption)


if __name__ == "__main__":
    main()
