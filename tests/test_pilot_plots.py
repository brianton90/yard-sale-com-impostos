from pathlib import Path

from visualization.pilot_plots import plot_final_windows_comparison, plot_series_to_2t


def test_pilot_plots_are_nonempty(tmp_path: Path) -> None:
    series_rows = [
        {"configuration_tag": "none", "metric": metric, "step": step, "mean": 0.1 * step, "standard_error": 0.01}
        for metric in ("gini", "max_share", "volume") for step in range(1, 5)
    ]
    summary_rows = [
        {"configuration_tag": "none", "metric": metric, "horizon": horizon, "window_index": index, "mean": 0.1, "standard_error": 0.01}
        for metric in ("gini", "max_share", "volume") for horizon in (2, 4) for index in range(2)
    ]
    series = plot_series_to_2t(series_rows, tmp_path / "series.png", 2, (3, 4))
    windows = plot_final_windows_comparison(summary_rows, tmp_path / "windows.png")
    assert series.stat().st_size > 0
    assert windows.stat().st_size > 0
