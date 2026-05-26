import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


DEFAULT_DIAGONAL_ID_PATH = Path(
    "outputs/mahalanobis_distance/imagenet_val3K_diagonal_mahalanobis.csv"
)
DEFAULT_DIAGONAL_OOD_PATH = Path(
    "outputs/mahalanobis_distance/ninco_texture_3K_diagonal_mahalanobis.csv"
)
DEFAULT_FULL_ID_PATH = Path(
    "outputs/mahalanobis_distance_full/imagenet_val3K_full_mahalanobis.csv"
)
DEFAULT_FULL_OOD_PATH = Path(
    "outputs/mahalanobis_distance_full/ninco_texture_3K_full_mahalanobis.csv"
)
DEFAULT_OUTPUT_DIR = Path("reports/figures/captioning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ROC curves for Mahalanobis distance results.")
    parser.add_argument("--diagonal-id-path", default=DEFAULT_DIAGONAL_ID_PATH)
    parser.add_argument("--diagonal-ood-path", default=DEFAULT_DIAGONAL_OOD_PATH)
    parser.add_argument("--full-id-path", default=DEFAULT_FULL_ID_PATH)
    parser.add_argument("--full-ood-path", default=DEFAULT_FULL_OOD_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def distance_column(dataframe: pd.DataFrame) -> str:
    columns = [column for column in dataframe.columns if column.endswith("_mahalanobis_distance")]
    if len(columns) != 1:
        raise ValueError(f"Expected one Mahalanobis distance column, found: {columns}")
    return columns[0]


def load_distances(path: Path) -> np.ndarray:
    dataframe = pd.read_csv(path)
    column = distance_column(dataframe)
    values = dataframe[column].dropna().to_numpy(dtype=np.float64)
    if len(values) == 0:
        raise ValueError(f"{path} does not contain any Mahalanobis distance values.")
    return values


def plot_roc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    output_path: Path,
    title: str,
) -> tuple[float, Path]:
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return roc_auc, output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            "diagonal",
            Path(args.diagonal_id_path),
            Path(args.diagonal_ood_path),
            output_dir / "mahalanobis_uncertainty_roc_diagonal.png",
        ),
        (
            "full",
            Path(args.full_id_path),
            Path(args.full_ood_path),
            output_dir / "mahalanobis_uncertainty_roc_full.png",
        ),
    ]

    for method, id_path, ood_path, output_path in plots:
        id_scores = load_distances(id_path)
        ood_scores = load_distances(ood_path)
        roc_auc, saved_path = plot_roc(
            id_scores,
            ood_scores,
            output_path,
            f"OOD ROC by Mahalanobis Distance ({method})",
        )

        print(f"[{method}] ID path: {id_path}")
        print(f"[{method}] OOD path: {ood_path}")
        print(f"[{method}] ID count: {len(id_scores)}")
        print(f"[{method}] OOD count: {len(ood_scores)}")
        print(f"[{method}] ID mean distance: {id_scores.mean():.6f}")
        print(f"[{method}] OOD mean distance: {ood_scores.mean():.6f}")
        print(f"[{method}] ROC AUC: {roc_auc:.6f}")
        print(f"[{method}] saved ROC: {saved_path}")


if __name__ == "__main__":
    main()
