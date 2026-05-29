from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationMetrics:
    total_wealth: float
    mean_wealth: float
    median_wealth: float
    min_wealth: float
    max_wealth: float
    gini: float


def compute_gini(wealth: np.ndarray) -> float:
    """Compute the Gini coefficient for non-negative wealth."""
    sorted_wealth = np.sort(np.asarray(wealth, dtype=float))
    n = len(sorted_wealth)
    total = sorted_wealth.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_wealth) / (n * total)) - (n + 1) / n
    return float(gini)


def summarize_wealth(wealth: np.ndarray) -> SimulationMetrics:
    wealth = np.asarray(wealth, dtype=float)
    return SimulationMetrics(
        total_wealth=float(wealth.sum()),
        mean_wealth=float(wealth.mean()),
        median_wealth=float(np.median(wealth)),
        min_wealth=float(wealth.min()),
        max_wealth=float(wealth.max()),
        gini=compute_gini(wealth),
    )
