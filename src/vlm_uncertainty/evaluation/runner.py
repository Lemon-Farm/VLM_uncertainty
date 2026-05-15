import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from vlm_uncertainty.models.blip import BLIPWrapper, DEFAULT_CAPTION_PREFIX


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
    prefix: str = DEFAULT_CAPTION_PREFIX,
    compute_softmax_entropy: bool = False,
    exclude_softmax_entropy_stopwords: bool = False,
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
                softmax_entropy = None
                if compute_softmax_entropy:
                    softmax_entropy = model.softmax_entropy(
                        pixel_values=batch["pixel_values"],
                        max_new_tokens=max_new_tokens,
                        prefix=prefix,
                        exclude_stopwords=exclude_softmax_entropy_stopwords,
                    )
                    captions = softmax_entropy["captions"]
                else:
                    captions = model.generate_captions(
                        pixel_values=batch["pixel_values"],
                        max_new_tokens=max_new_tokens,
                        prefix=prefix,
                    )
                laplace_lora = None
                if model.uses_bayesian_lora:
                    laplace_lora = model.laplace_lora_topk_logits(
                        pixel_values=batch["pixel_values"],
                        prefix=prefix,
                    )

                for row, (index, caption) in enumerate(
                    zip(batch["indices"].tolist(), captions)
                ):
                    record = {
                        "index": index,
                        "caption": caption,
                    }
                    if softmax_entropy is not None:
                        record["softmax_entropy"] = {
                            "token_entropy": softmax_entropy["token_entropy"][row].tolist(),
                            "uncertainty": softmax_entropy["caption_uncertainty"][row].item(),
                            "generated_steps": softmax_entropy["generated_steps"].item(),
                            "used_token_count": softmax_entropy["used_token_count"][row].item(),
                            "excluded_stopwords": exclude_softmax_entropy_stopwords,
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
