from pathlib import Path

import numpy as np
import pytest

from analysis.stability import (
    StabilityComparison,
    effective_margin,
    final_tail_windows,
    joint_status,
    metric_status,
    paired_equivalence,
    realization_window_means,
    t_critical,
)
from experiments.pilot_configuration import load_stability_margins


EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "stability_margins.example.json"


def test_final_tail_windows_are_inclusive_successive_and_non_overlapping() -> None:
    margins = load_stability_margins(EXAMPLE)
    windows = final_tail_windows(10, margins)
    assert [(window.start, window.end) for window in windows] == [(7, 7), (8, 8), (9, 9), (10, 10)]
    values = realization_window_means(np.arange(1, 11), np.array([np.arange(1, 11), np.arange(11, 21)]), windows[-1])
    np.testing.assert_array_equal(values, [10.0, 20.0])
    with pytest.raises(ValueError):
        final_tail_windows(1, margins)


def test_paired_equivalence_and_volume_effective_margin() -> None:
    margins = load_stability_margins(EXAMPLE)
    comparison = paired_equivalence("gini", "adjacent_windows", [1.0, 1.0], [1.001, 1.001], margins, 8, 0, 8, 1)
    assert comparison.decision == "pass"
    volume_margin = effective_margin("volume", margins, 1.0, 2.0)
    assert volume_margin == pytest.approx(0.2)
    failed = paired_equivalence("max_share", "adjacent_windows", [0.0, 0.0], [1.0, 1.0], margins, 8, 0, 8, 1)
    assert failed.decision == "fail"
    assert t_critical(0.95, 1) == pytest.approx(12.706)


def test_all_metric_and_joint_status_states() -> None:
    base = dict(metric="gini", comparison="x", horizon=None, left_horizon=8, left_window=0, right_horizon=8, right_window=1, mean_difference=0.0, standard_error=0.0, t_critical=1.0, ci_lower=0.0, ci_upper=0.0, effective_margin=1.0, n_realizations=2)
    assert metric_status([StabilityComparison(**base, decision="pass")]) == "stabilized"
    assert metric_status([StabilityComparison(**base, decision="fail")]) == "finite_time"
    assert metric_status([StabilityComparison(**base, decision="inconclusive")]) == "inconclusive"
    assert metric_status([]) == "not_assessed"
    assert joint_status(["stabilized", "stabilized", "stabilized"]) == "stabilized"
    assert joint_status(["stabilized", "finite_time", "stabilized"]) == "finite_time"
    assert joint_status(["stabilized", "inconclusive", "stabilized"]) == "inconclusive"
    assert joint_status(["stabilized", "not_assessed", "stabilized"]) == "not_assessed"
