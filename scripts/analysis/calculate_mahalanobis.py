import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GENERIC_PATH = Path("outputs/vision_embeddings/imagenet_train50K_embeddings.csv")
DEFAULT_TARGET_PATH = Path("outputs/vision_embeddings/ninco_texture_3K_embeddings.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/mahalanobis_distance")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Mahalanobis distance.")
    parser.add_argument("--generic-data", default=DEFAULT_GENERIC_PATH)
    parser.add_argument("--target-data", default=DEFAULT_TARGET_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", choices=["diagonal", "full"], default="diagonal")
    return parser.parse_args()


def embedding_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = [column for column in dataframe.columns if column.startswith("embedding_")]
    if not columns:
        raise ValueError("No embedding columns found.")
    return columns


def diagonal_mahalanobis(
    target_embeddings: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    safe_sigma = np.maximum(sigma, EPS)
    normalized = (target_embeddings - mu) / safe_sigma
    return np.sqrt(np.sum(normalized**2, axis=1))


def full_mahalanobis(
    target_embeddings: np.ndarray,
    generic_embeddings: np.ndarray,
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.cov(generic_embeddings, rowvar=False, ddof=0)
    inverse_covariance = np.linalg.pinv(covariance)
    centered = target_embeddings - mu
    squared_distances = np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)
    squared_distances = np.maximum(squared_distances, 0.0)
    return np.sqrt(squared_distances), covariance


def output_stem(path: Path) -> str:
    return path.stem.removesuffix("_embeddings")


def main() -> None:
    args = parse_args()
    generic_path = Path(args.generic_data)
    target_path = Path(args.target_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generic = pd.read_csv(generic_path)
    target = pd.read_csv(target_path)

    columns = embedding_columns(generic)
    missing_columns = [column for column in columns if column not in target.columns]
    if missing_columns:
        raise ValueError(f"Target data is missing embedding columns: {missing_columns[:5]}")

    generic_embeddings = generic[columns].to_numpy(dtype=np.float64)
    target_embeddings = target[columns].to_numpy(dtype=np.float64)

    mu = generic_embeddings.mean(axis=0)
    sigma = generic_embeddings.std(axis=0, ddof=0)
    if args.method == "diagonal":
        distances = diagonal_mahalanobis(target_embeddings, mu, sigma)
        density = pd.DataFrame(
            {
                "embedding": columns,
                "mu": mu,
                "sigma": sigma,
            }
        )
    else:
        distances, covariance = full_mahalanobis(target_embeddings, generic_embeddings, mu)
        density = pd.DataFrame(
            covariance,
            index=columns,
            columns=columns,
        )
        density.insert(0, "mu", mu)

    result = pd.DataFrame(
        {
            "index": target["index"] if "index" in target.columns else target.index,
            f"{args.method}_mahalanobis_distance": distances,
        }
    )
    generic_name = output_stem(generic_path)
    target_name = output_stem(target_path)
    output_path = output_dir / f"{target_name}_{args.method}_mahalanobis.csv"
    result.to_csv(output_path, index=False)

    density_path = output_dir / f"{generic_name}_{args.method}_density.csv"
    density.to_csv(density_path, index=args.method == "full")

    print(f"method: {args.method}")
    print(f"mean {args.method} mahalanobis distance: {distances.mean()}")
    print(f"std {args.method} mahalanobis distance: {distances.std()}")
    print(f"saved distances: {output_path}")
    print(f"saved density: {density_path}")


if __name__ == "__main__":
    main()
