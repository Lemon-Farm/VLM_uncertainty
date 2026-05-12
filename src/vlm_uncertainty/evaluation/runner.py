import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from vlm_uncertainty.models.blip import BLIPWrapper


VISION_EMBEDDING_OUTPUT_DIR = Path("outputs/vision_embeddings")
VISION_EMBEDDING_OUTPUT_PATH = VISION_EMBEDDING_OUTPUT_DIR / "vision_embeddings.csv"
VISION_EMBEDDING_OUTPUT = "pooler_output"


def write_embedding_rows(
    writer: csv.writer,
    indices: torch.Tensor,
    embeddings: torch.Tensor,
) -> None:
    flattened = embeddings.flatten(start_dim=1)
    for index, embedding in zip(indices.tolist(), flattened.tolist()):
        writer.writerow([index, *embedding])


def run_caption_inference(
    model: BLIPWrapper,
    dataloader: Iterable[dict],
    output_path: str | Path,
    max_new_tokens: int = 30,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_file = None
    embedding_writer = None
    if model.extract_vision_embeddings:
        VISION_EMBEDDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        embedding_file = VISION_EMBEDDING_OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        )

    with output_path.open("w", encoding="utf-8") as output_file:
        try:
            for batch in tqdm(dataloader, desc="BLIP inference", total=len(dataloader)):
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

                if model.extract_vision_embeddings:
                    embeddings = model.vision_embedding(
                        pixel_values=batch["pixel_values"],
                        output=VISION_EMBEDDING_OUTPUT,
                    )
                    flattened_dim = embeddings.flatten(start_dim=1).shape[1]
                    if embedding_writer is None:
                        embedding_writer = csv.writer(embedding_file)
                        embedding_writer.writerow(
                            ["index", *[f"embedding_{i}" for i in range(flattened_dim)]]
                        )
                    write_embedding_rows(
                        writer=embedding_writer,
                        indices=batch["indices"],
                        embeddings=embeddings,
                    )
        finally:
            if embedding_file is not None:
                embedding_file.close()
