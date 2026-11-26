import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analysis.stability import STABILITY_METRICS


_LABELS = {
    "gini": "Gini",
    "max_share": "Participação máxima",
    "volume": "Volume bruto / W₀",
}


def _finish(fig: plt.Figure, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=output_path.suffix, dir=output_path.parent,
    )
    os.close(descriptor)
    try:
        fig.tight_layout()
        fig.savefig(temporary_name, dpi=150)
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(fig)
    return output_path


def plot_series_to_2t(
    rows: list[dict[str, Any]],
    output_path: Path,
    t_horizon: int,
    window_limits: tuple[int, ...],
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for axis, metric in zip(axes, STABILITY_METRICS):
        selected = [row for row in rows if row["metric"] == metric]
        tags = list(dict.fromkeys(str(row["configuration_tag"]) for row in selected))
        for tag in tags:
            values = [row for row in selected if row["configuration_tag"] == tag]
            steps = np.asarray([int(row["step"]) for row in values])
            mean = np.asarray([float(row["mean"]) for row in values])
            standard_error = np.asarray([float(row["standard_error"]) for row in values])
            axis.plot(steps, mean, linewidth=1.0, label=tag)
            axis.fill_between(steps, mean - standard_error, mean + standard_error, alpha=0.12)
        axis.axvline(t_horizon, color="black", linestyle="--", linewidth=1.0, label="T" if metric == "gini" else None)
        for limit in window_limits:
            axis.axvline(limit, color="gray", linestyle=":", linewidth=0.8)
        axis.set_ylabel(_LABELS[metric])
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Passo de Monte Carlo")
    fig.suptitle("Séries médias ± EP até 2T; T e limites de janela")
    return _finish(fig, output_path)


def plot_final_windows_comparison(rows: list[dict[str, Any]], output_path: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    for axis, metric in zip(axes, STABILITY_METRICS):
        selected = [row for row in rows if row["metric"] == metric]
        tags = list(dict.fromkeys(str(row["configuration_tag"]) for row in selected))
        for position, tag in enumerate(tags):
            values = [row for row in selected if row["configuration_tag"] == tag]
            values.sort(key=lambda row: (int(row["horizon"]), int(row["window_index"])))
            x = np.arange(len(values), dtype=float) + position * 0.05
            mean = np.asarray([float(row["mean"]) for row in values])
            standard_error = np.asarray([float(row["standard_error"]) for row in values])
            axis.errorbar(x, mean, yerr=standard_error, marker="o", capsize=2, linewidth=0.9, label=tag)
        axis.set_ylabel(_LABELS[metric])
        axis.set_xlabel("Janelas finais ordenadas: T, depois 2T")
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle("Médias temporais das janelas finais ± EP")
    return _finish(fig, output_path)
