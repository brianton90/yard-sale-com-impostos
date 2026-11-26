from dataclasses import dataclass
from math import floor
from typing import Sequence

import numpy as np

from analysis.statistics import mean_and_standard_error, window_mean
from experiments.pilot_configuration import StabilityMargins


STABILITY_METRICS = ("gini", "max_share", "volume")
STATUSES = ("stabilized", "finite_time", "inconclusive", "not_assessed")

# Two-sided Student-t critical values for confidence levels supported by StabilityMargins.
# For df > 30 the df=30 value is deliberately retained as a conservative fallback.
_T_CRITICAL = {
    0.90: (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697),
    0.95: (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042),
    0.99: (63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169, 3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845, 2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771, 2.763, 2.756, 2.750),
}


@dataclass(frozen=True)
class TailWindow:
    horizon: int
    index: int
    start: int
    end: int


@dataclass(frozen=True)
class StabilityComparison:
    metric: str
    comparison: str
    horizon: int | None
    left_horizon: int
    left_window: int
    right_horizon: int
    right_window: int
    mean_difference: float
    standard_error: float
    t_critical: float
    ci_lower: float
    ci_upper: float
    effective_margin: float
    decision: str
    n_realizations: int


def final_tail_windows(horizon: int, margins: StabilityMargins) -> tuple[TailWindow, ...]:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon deve ser inteiro positivo")
    tail_steps = floor(horizon * margins.tail_fraction)
    width = floor(tail_steps / margins.K)
    if width < 1:
        raise ValueError("a cauda final não comporta as janelas configuradas")
    first = horizon - margins.K * width + 1
    return tuple(
        TailWindow(horizon=horizon, index=index, start=first + index * width, end=first + (index + 1) * width - 1)
        for index in range(margins.K)
    )


def t_critical(confidence: float, degrees_of_freedom: int) -> float:
    if confidence not in _T_CRITICAL:
        raise ValueError("confidence sem valor crítico Student-t tabelado")
    if not isinstance(degrees_of_freedom, int) or degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom deve ser inteiro positivo")
    return _T_CRITICAL[confidence][min(degrees_of_freedom, 30) - 1]


def realization_window_means(
    steps: np.ndarray,
    series: Sequence[np.ndarray] | np.ndarray,
    window: TailWindow,
) -> np.ndarray:
    data = np.asarray(series, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or not np.all(np.isfinite(data)):
        raise ValueError("series deve conter ao menos duas realizações finitas")
    return np.asarray([window_mean(steps, values, window.start, window.end) for values in data], dtype=float)


def summarize_realization_means(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    return mean_and_standard_error(values)


def effective_margin(metric: str, margins: StabilityMargins, left_mean: float, right_mean: float) -> float:
    if metric not in STABILITY_METRICS:
        raise ValueError(f"métrica inválida: {metric}")
    absolute = margins.absolute_margin_for(metric)
    if metric != "volume":
        return absolute
    return max(absolute, margins.relative_margin_for(metric) * max(abs(left_mean), abs(right_mean)))


def paired_equivalence(
    metric: str,
    comparison: str,
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    margins: StabilityMargins,
    left_horizon: int,
    left_window: int,
    right_horizon: int,
    right_window: int,
) -> StabilityComparison:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.ndim != 1 or right_values.ndim != 1 or len(left_values) < 2 or left_values.shape != right_values.shape:
        raise ValueError("comparação pareada exige vetores finitos de mesmo tamanho R >= 2")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("comparação pareada contém valores não finitos")
    differences = right_values - left_values
    difference_mean, standard_error = mean_and_standard_error(differences)
    critical = t_critical(margins.confidence, len(differences) - 1)
    ci_lower = difference_mean - critical * standard_error
    ci_upper = difference_mean + critical * standard_error
    margin = effective_margin(metric, margins, float(left_values.mean()), float(right_values.mean()))
    if ci_lower >= -margin and ci_upper <= margin:
        decision = "pass"
    elif ci_lower > margin or ci_upper < -margin:
        decision = "fail"
    else:
        decision = "inconclusive"
    return StabilityComparison(
        metric=metric, comparison=comparison, horizon=None,
        left_horizon=left_horizon, left_window=left_window,
        right_horizon=right_horizon, right_window=right_window,
        mean_difference=difference_mean, standard_error=standard_error,
        t_critical=critical, ci_lower=ci_lower, ci_upper=ci_upper,
        effective_margin=margin, decision=decision, n_realizations=len(differences),
    )


def metric_status(comparisons: Sequence[StabilityComparison] | None) -> str:
    if not comparisons:
        return "not_assessed"
    decisions = {comparison.decision for comparison in comparisons}
    if "fail" in decisions:
        return "finite_time"
    if "inconclusive" in decisions:
        return "inconclusive"
    if decisions == {"pass"}:
        return "stabilized"
    return "not_assessed"


def joint_status(metric_statuses: Sequence[str]) -> str:
    statuses = tuple(metric_statuses)
    if len(statuses) != len(STABILITY_METRICS) or any(status not in STATUSES for status in statuses):
        raise ValueError("status de métrica inválido")
    if "not_assessed" in statuses:
        return "not_assessed"
    if "finite_time" in statuses:
        return "finite_time"
    if "inconclusive" in statuses:
        return "inconclusive"
    return "stabilized"
