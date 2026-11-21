import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from simulation.rules import execute_interaction, yard_sale_step
from simulation.taxes import apply_wealth_tax, redistribute_cash

if TYPE_CHECKING:
    from experiments.randomization import SimulationRandomness


@dataclass(frozen=True)
class SimulationResult:
    state: SimulationState
    metrics: SimulationMetrics
    history: SimulationHistory
    series: SimulationSeries
    duration_seconds: float


def run_simulation(
    config: SimulationConfig,
    randomness: "SimulationRandomness | None" = None,
    required_snapshot_steps: tuple[int, ...] = (),
) -> SimulationResult:
    """Executa uma simulação completa do modelo Yard-Sale.

    Sem ``randomness``, preserva o caminho legado baseado em ``config.seed``.
    No caminho experimental, beta, pares e vencedores vêm exclusivamente dos
    arrays injetados. Os snapshots incluem os passos regulares e obrigatórios.
    """
    if config.snapshot_interval < 1:
        raise ValueError("snapshot_interval deve ser >= 1")
    if any(not isinstance(step, (int, np.integer)) for step in required_snapshot_steps):
        raise ValueError("required_snapshot_steps deve conter somente inteiros")
    required = tuple(int(step) for step in required_snapshot_steps)
    if any(step < 0 or step > config.n_steps for step in required):
        raise ValueError("required_snapshot_steps deve estar entre 0 e n_steps")

    start_time = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    if randomness is not None:
        randomness.validate(config.n_agents, config.n_steps)
    state = initialize_state(config, rng, None if randomness is None else randomness.aversion)
    initial_total = state.total_wealth
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

    steps_to_save = set(required)
    steps_to_save.update(range(0, config.n_steps + 1, config.snapshot_interval))
    steps_to_save.update((0, config.n_steps))
    snapshot_steps: list[int] = [0]
    snapshots: list[np.ndarray] = [state.wealth.copy()]

    for step in range(1, config.n_steps + 1):
        step_volume = 0.0
        for trade_index in range(trades_per_step):
            if randomness is None:
                trade = yard_sale_step(state, rng, transaction_tax_rate)
            else:
                pair = randomness.pairs[step - 1, trade_index]
                trade = execute_interaction(
                    state,
                    int(pair[0]),
                    int(pair[1]),
                    bool(randomness.first_agent_wins[step - 1, trade_index]),
                    transaction_tax_rate,
                )
            step_volume += trade.stake

        if wealth_tax_rate > 0.0:
            apply_wealth_tax(state, wealth_tax_rate)

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

        if step in steps_to_save:
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
