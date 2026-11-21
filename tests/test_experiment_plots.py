import numpy as np

from visualization.experiment_plots import (
    METRICS,
    plot_lorenz,
    plot_mean_series,
    plot_metrics_vs_tau,
    plot_mobility,
)


def test_experiment_plot_functions_create_png_files(tmp_path) -> None:
    series_rows = [
        {
            "configuration_tag": "none",
            "metric": metric,
            "step": step,
            "mean": 0.1 * step,
            "standard_error": 0.01,
        }
        for metric in METRICS
        for step in (1, 2)
    ]
    summary_rows = [
        {
            "configuration_tag": tag,
            "regime": regime,
            "tax_rate": tax_rate,
            "metric": metric,
            "mean": mean,
            "standard_error": 0.01,
        }
        for metric in METRICS
        for tag, regime, tax_rate, mean in (
            ("none", "none", 0.0, 0.2),
            ("wealth_tau0.05", "wealth", 0.05, 0.15),
            ("transaction_tau0.05", "transaction", 0.05, 0.18),
        )
    ]
    lorenz_rows = [
        {
            "configuration_tag": "none",
            "population_share": share,
            "mean": share,
            "standard_error": 0.0,
        }
        for share in (0.0, 0.5, 1.0)
    ]

    paths = (
        plot_mean_series(series_rows, tmp_path / "series.png", "series"),
        plot_metrics_vs_tau(summary_rows, tmp_path / "metrics.png", "metrics"),
        plot_lorenz(lorenz_rows, tmp_path / "lorenz.png", "lorenz"),
        plot_mobility(np.eye(5), tmp_path / "mobility.png", "mobility"),
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
