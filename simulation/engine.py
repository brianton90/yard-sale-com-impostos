import time
from dataclasses import dataclass

import numpy as np

from config import SimulationConfig
from simulation.initialization import initialize_state
from simulation.metrics import (
    SimulationMetrics,
    SimulationSeries,
    compute_conservation_error,
    compute_gini,
    compute_max_share,
    summarize_wealth,
)
from simulation.model import SimulationHistory, SimulationState
from simulation.rules import yard_sale_step
from simulation.taxes import apply_wealth_tax, redistribute_cash


@dataclass(frozen=True)
class SimulationResult:
    state: SimulationState
    metrics: SimulationMetrics
    history: SimulationHistory
    series: SimulationSeries
    duration_seconds: float  # duração total da execução (inicialização + todos os passos + métricas)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Executa uma simulação completa do modelo Yard-Sale.

    Sequência de cada passo de Monte Carlo (N/2 tentativas de troca):
      1. trocas sequenciais (com imposto sobre transações, se o regime for esse);
      2. cobrança patrimonial simultânea (se o regime for esse);
      3. redistribuição igualitária do caixa;
      4. registro dos indicadores, após a redistribuição.

    Snapshots da riqueza são guardados no passo 0, a cada `snapshot_interval`
    passos e no passo final.
    """
    if config.snapshot_interval < 1:
        raise ValueError("snapshot_interval deve ser >= 1")

    start_time = time.perf_counter()

    rng = np.random.default_rng(config.seed)
    state = initialize_state(config, rng)
    initial_total = state.total_wealth  # W0
    trades_per_step = config.n_agents // 2
    transaction_tax_rate = config.transaction_tax_rate
    wealth_tax_rate = config.wealth_tax_rate

    gini = np.empty(config.n_steps)
    max_share = np.empty(config.n_steps)
    volume = np.empty(config.n_steps)
    revenue = np.empty(config.n_steps)

    max_conservation_error = 0.0
    min_wealth_observed = float(state.wealth.min())
    max_cash = 0.0

    snapshot_steps: list[int] = [0]
    snapshots: list[np.ndarray] = [state.wealth.copy()]

    for step in range(1, config.n_steps + 1):
        step_volume = 0.0
        for _ in range(trades_per_step):
            trade = yard_sale_step(state, rng, transaction_tax_rate)
            step_volume += trade.stake

        if wealth_tax_rate > 0.0:
            apply_wealth_tax(state, wealth_tax_rate)

        # Estado imediatamente antes da devolução: o caixa contém a arrecadação do ciclo.
        step_revenue = state.cash
        max_cash = max(max_cash, step_revenue)
        min_wealth_observed = min(min_wealth_observed, float(state.wealth.min()))
        max_conservation_error = max(
            max_conservation_error,
            compute_conservation_error(state.wealth, state.cash, initial_total),
        )

        redistribute_cash(state)

        max_conservation_error = max(
            max_conservation_error,
            compute_conservation_error(state.wealth, state.cash, initial_total),
        )

        idx = step - 1
        gini[idx] = compute_gini(state.wealth)
        max_share[idx] = compute_max_share(state.wealth)
        volume[idx] = step_volume / initial_total
        revenue[idx] = step_revenue / initial_total

        if step % config.snapshot_interval == 0 or step == config.n_steps:
            snapshot_steps.append(step)
            snapshots.append(state.wealth.copy())

    metrics = summarize_wealth(
        state.wealth,
        max_conservation_error=max_conservation_error,
        min_wealth_observed=min_wealth_observed,
        max_cash=max_cash,
    )
    history = SimulationHistory(
        steps=np.array(snapshot_steps, dtype=int),
        wealth=np.vstack(snapshots),
    )
    series = SimulationSeries(
        steps=np.arange(1, config.n_steps + 1),
        gini=gini,
        max_share=max_share,
        volume=volume,
        revenue=revenue,
    )

    duration_seconds = time.perf_counter() - start_time
    return SimulationResult(
        state=state,
        metrics=metrics,
        history=history,
        series=series,
        duration_seconds=duration_seconds,
    )
