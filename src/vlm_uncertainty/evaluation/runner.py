import json
from pathlib import Path
from typing import Iterable

from vlm_uncertainty.models.blip import BLIPWrapper


def run_caption_inference(
    model: BLIPWrapper,
    dataloader: Iterable[dict],
    output_path: str | Path,
    max_new_tokens: int = 30,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for batch in dataloader:
            captions = model.generate_captions(
                pixel_values=batch["pixel_values"],
                max_new_tokens=max_new_tokens,
            )

            for index, caption in zip(batch["indices"].tolist(), captions):
                record = {
                    "index": index,
                    "caption": caption,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
