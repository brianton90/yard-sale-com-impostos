import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_wealth_distribution(
    wealth: np.ndarray,
    output_path: Path,
    bins: int = 50,
    show: bool = False,
) -> Path:
    """Plot the final wealth distribution and save it to `output_path`."""
    fig = plt.figure(figsize=(10, 6))
    plt.hist(wealth, bins=bins, edgecolor="black", alpha=0.8, label="Distribuição final")
    plt.yscale("log")
    plt.xlabel("Riqueza dos agentes", fontsize=12)
    plt.ylabel("Número de agentes (escala log)", fontsize=12)
    plt.title("Distribuição de Riqueza – Modelo Yard Sale", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_time_series(
    steps: np.ndarray,
    indicators: dict[str, np.ndarray],
    output_path: Path,
    title: str = "Indicadores ao longo do tempo",
    show: bool = False,
) -> Path:
    """Plota uma série temporal por indicador (um painel cada) e salva em `output_path`."""
    n_panels = len(indicators)
    n_cols = 2
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows), sharex=True, squeeze=False)
    flat_axes = axes.ravel()

    for ax, (label, values) in zip(flat_axes, indicators.items()):
        ax.plot(steps, values, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.6)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Passo de Monte Carlo")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)
    return output_path
