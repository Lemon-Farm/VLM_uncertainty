import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


DEFAULT_ID_PATH = Path("outputs/blip_vqa_answers_ID.jsonl")
DEFAULT_OOD_PATH = Path("outputs/blip_vqa_answers_OOD.jsonl")
DEFAULT_OUTPUT_DIR = Path("reports/figures")
DEFAULT_OUTPUT_NAME = "blora_uncertainty_violin.png"
UNCERTAINTY_PATTERN = re.compile(r'"uncertainty"\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BLoRA uncertainty violin plot.")
    parser.add_argument("--id-path", default=DEFAULT_ID_PATH)
    parser.add_argument("--ood-path", default=DEFAULT_OOD_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional uncertainty threshold. Scores >= threshold are classified as OOD.",
    )
    return parser.parse_args()


def extract_uncertainty(record: dict, path: Path, line_number: int) -> float:
    if "uncertainty" in record:
        return float(record["uncertainty"])

    laplace_lora = record.get("laplace_lora")
    if isinstance(laplace_lora, dict) and "uncertainty" in laplace_lora:
        return float(laplace_lora["uncertainty"])

    raise ValueError(f"{path}:{line_number} does not contain an uncertainty value.")


def extract_uncertainty_from_line(line: str, path: Path, line_number: int) -> float:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        match = UNCERTAINTY_PATTERN.search(line)
        if match:
            return float(match.group(1))
        raise ValueError(f"{path}:{line_number} is not valid JSON and has no uncertainty value.")

    return extract_uncertainty(record, path, line_number)


def load_uncertainty(path: Path) -> np.ndarray:
    scores = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            scores.append(extract_uncertainty_from_line(line, path, line_number))

    if not scores:
        raise ValueError(f"{path} does not contain any uncertainty values.")

    return np.asarray(scores, dtype=np.float64)


def plot_violin(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    output_dir: Path,
    output_name: str,
) -> Path:
    output_path = output_dir / output_name

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot([id_scores, ood_scores], showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_alpha(0.35)

    ax.set_xticks([1, 2], ["ID", "OOD"])
    ax.set_ylabel("Uncertainty")
    ax.set_title("BLoRA Uncertainty")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def threshold_point(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    false_positive_rate = float(np.mean(id_scores >= threshold))
    true_positive_rate = float(np.mean(ood_scores >= threshold))
    return false_positive_rate, true_positive_rate


def plot_roc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    output_dir: Path,
    threshold: float | None,
) -> tuple[Path, float, tuple[float, float] | None]:
    output_path = output_dir / "blora_uncertainty_roc.png"

    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    point = None
    if threshold is not None:
        point = threshold_point(id_scores, ood_scores, threshold)
        ax.scatter([point[0]], [point[1]], color="red", zorder=3, label=f"threshold = {threshold:g}")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("OOD ROC by BLoRA Uncertainty")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, roc_auc, point


def main() -> None:
    args = parse_args()
    id_path = Path(args.id_path)
    ood_path = Path(args.ood_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_scores = load_uncertainty(id_path)
    ood_scores = load_uncertainty(ood_path)

    violin_path = plot_violin(id_scores, ood_scores, output_dir, args.output_name)
    roc_path, roc_auc, point = plot_roc(id_scores, ood_scores, output_dir, args.threshold)

    print(f"ID mean uncertainty: {id_scores.mean():.6f}")
    print(f"OOD mean uncertainty: {ood_scores.mean():.6f}")
    print(f"ROC AUC: {roc_auc:.6f}")
    if point is not None:
        print(f"threshold: {args.threshold}")
        print(f"FPR at threshold: {point[0]:.6f}")
        print(f"TPR at threshold: {point[1]:.6f}")
    print(f"saved violin: {violin_path}")
    print(f"saved ROC: {roc_path}")


if __name__ == "__main__":
    main()
