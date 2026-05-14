import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


DEFAULT_INPUT_DIR = Path("outputs/mahalanobis_distance")
DEFAULT_OUTPUT_DIR = Path("reports/figures")
GRID_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KDEs for Mahalanobis distance results.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def distance_column(dataframe: pd.DataFrame) -> str:
    columns = [column for column in dataframe.columns if column.endswith("_mahalanobis_distance")]
    if len(columns) != 1:
        raise ValueError(f"Expected one Mahalanobis distance column, found: {columns}")
    return columns[0]


def bandwidth(values: np.ndarray) -> float:
    std = values.std(ddof=1)
    if std == 0 or len(values) < 2:
        return 1.0
    return 1.06 * std * (len(values) ** (-1 / 5))


def kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    model = KernelDensity(kernel="gaussian", bandwidth=bandwidth(values))
    model.fit(values[:, None])
    return np.exp(model.score_samples(grid[:, None]))


def plot_single(name: str, values: np.ndarray, output_dir: Path) -> None:
    padding = max(values.std(ddof=1), 1.0)
    grid = np.linspace(values.min() - padding, values.max() + padding, GRID_SIZE)
    density = kde_curve(values, grid)

    plt.figure(figsize=(8, 5))
    plt.plot(grid, density, linewidth=2)
    plt.fill_between(grid, density, alpha=0.2)
    plt.title(name)
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_kde.png", dpi=200)
    plt.close()


def plot_overlay(results: dict[str, np.ndarray], output_dir: Path) -> None:
    min_value = min(values.min() for values in results.values())
    max_value = max(values.max() for values in results.values())
    max_std = max(values.std(ddof=1) for values in results.values())
    padding = max(max_std, 1.0)
    grid = np.linspace(min_value - padding, max_value + padding, GRID_SIZE)

    plt.figure(figsize=(9, 6))
    for name, values in results.items():
        plt.plot(grid, kde_curve(values, grid), linewidth=2, label=name)
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mahalanobis_kde_overlay.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for path in sorted(input_dir.glob("*_mahalanobis.csv")):
        dataframe = pd.read_csv(path)
        column = distance_column(dataframe)
        values = dataframe[column].dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            continue

        name = path.stem
        results[name] = values
        plot_single(name, values, output_dir)

    if not results:
        raise FileNotFoundError(f"No Mahalanobis result CSVs found under: {input_dir}")

    plot_overlay(results, output_dir)
    print(f"saved {len(results)} individual KDE plots and 1 overlay plot to {output_dir}")


if __name__ == "__main__":
    main()
