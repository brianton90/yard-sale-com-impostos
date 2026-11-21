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
    max_share: float  # participação do agente mais rico: max(w) / W
    # Diagnósticos acumulados ao longo da execução
    max_conservation_error: float = 0.0
    min_wealth_observed: float = 0.0
    max_cash: float = 0.0


@dataclass(frozen=True)
class SimulationSeries:
    """Indicadores por passo de Monte Carlo (registrados após a redistribuição).

    Todos os arrays têm shape (n_steps,). `volume` e `revenue` são adimensionais (divididos por W0).
    """

    steps: np.ndarray
    gini: np.ndarray
    max_share: np.ndarray
    volume: np.ndarray  # v(t) = V(t) / W0, volume bruto transacionado no passo
    revenue: np.ndarray  # q(t) = A(t) / W0, arrecadação do ciclo


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


def compute_max_share(wealth: np.ndarray) -> float:
    """Participação do agente mais rico na riqueza total dos agentes."""
    total = float(np.sum(wealth))
    if total == 0:
        return 0.0
    return float(np.max(wealth)) / total


def compute_conservation_error(wealth: np.ndarray, cash: float, initial_total: float) -> float:
    """Erro relativo de conservação: |sum(w) + C - W0| / W0."""
    return abs(float(np.sum(wealth)) + cash - initial_total) / initial_total


def summarize_wealth(
    wealth: np.ndarray,
    max_conservation_error: float = 0.0,
    min_wealth_observed: float | None = None,
    max_cash: float = 0.0,
) -> SimulationMetrics:
    wealth = np.asarray(wealth, dtype=float)
    return SimulationMetrics(
        total_wealth=float(wealth.sum()),
        mean_wealth=float(wealth.mean()),
        median_wealth=float(np.median(wealth)),
        min_wealth=float(wealth.min()),
        max_wealth=float(wealth.max()),
        gini=compute_gini(wealth),
        max_share=compute_max_share(wealth),
        max_conservation_error=max_conservation_error,
        min_wealth_observed=float(wealth.min()) if min_wealth_observed is None else min_wealth_observed,
        max_cash=max_cash,
    )
