from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("gini", "max_share", "volume", "revenue")
METRIC_LABELS = {
    "gini": "Gini",
    "max_share": "Participação máxima",
    "volume": "Volume bruto / W₀",
    "revenue": "Arrecadação / W₀",
}


def _finish(fig: plt.Figure, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_mean_series(rows: list[dict[str, Any]], output_path: Path, title: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, metric in zip(axes.ravel(), METRICS):
        metric_rows = [row for row in rows if row["metric"] == metric]
        tags = list(dict.fromkeys(row["configuration_tag"] for row in metric_rows))
        for tag in tags:
            selected = [row for row in metric_rows if row["configuration_tag"] == tag]
            steps = np.array([int(row["step"]) for row in selected])
            mean = np.array([float(row["mean"]) for row in selected])
            se = np.array([float(row["standard_error"]) for row in selected])
            ax.plot(steps, mean, linewidth=1.1, label=tag)
            ax.fill_between(steps, mean - se, mean + se, alpha=0.12)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("Passo de Monte Carlo")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    return _finish(fig, output_path)


def plot_metrics_vs_tau(rows: list[dict[str, Any]], output_path: Path, title: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, metric in zip(axes.ravel(), METRICS):
        metric_rows = [row for row in rows if row["metric"] == metric]
        none = next(row for row in metric_rows if row["regime"] == "none")
        baseline = float(none["mean"])
        baseline_se = float(none["standard_error"])
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label="none")
        ax.axhspan(baseline - baseline_se, baseline + baseline_se, color="black", alpha=0.08)
        for regime in ("wealth", "transaction"):
            selected = sorted(
                (row for row in metric_rows if row["regime"] == regime),
                key=lambda row: float(row["tax_rate"]),
            )
            tau = [float(row["tax_rate"]) for row in selected]
            mean = [float(row["mean"]) for row in selected]
            se = [float(row["standard_error"]) for row in selected]
            ax.errorbar(tau, mean, yerr=se, marker="o", capsize=3, label=regime)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("Alíquota τ")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    return _finish(fig, output_path)


def plot_lorenz(rows: list[dict[str, Any]], output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Igualdade")
    tags = list(dict.fromkeys(row["configuration_tag"] for row in rows))
    for tag in tags:
        selected = [row for row in rows if row["configuration_tag"] == tag]
        x = np.array([float(row["population_share"]) for row in selected])
        mean = np.array([float(row["mean"]) for row in selected])
        se = np.array([float(row["standard_error"]) for row in selected])
        ax.plot(x, mean, label=tag)
        ax.fill_between(x, np.maximum(0.0, mean - se), np.minimum(1.0, mean + se), alpha=0.12)
    ax.set_xlabel("Fração acumulada da população")
    ax.set_ylabel("Fração acumulada da riqueza")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title(title)
    return _finish(fig, output_path)


def plot_mobility(matrix: np.ndarray, output_path: Path, title: str) -> Path:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (5, 5):
        raise ValueError("matrix deve ter shape (5, 5)")
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    for row in range(5):
        for column in range(5):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    ax.set_xticks(range(5), range(1, 6))
    ax.set_yticks(range(5), range(1, 6))
    ax.set_xlabel("Quintil de destino")
    ax.set_ylabel("Quintil de origem")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Probabilidade")
    return _finish(fig, output_path)
