import numpy as np
import pytest

from simulation.model import SimulationState
from simulation.rules import (
    apply_trade,
    compute_stake,
    decide_winner,
    select_agents,
    yard_sale_step,
)


def _random_state(rng: np.random.Generator, n: int) -> SimulationState:
    """Estado aleatório com riqueza positiva e aversão em [0, 1)."""
    return SimulationState(
        wealth=rng.uniform(0.1, 5.0, size=n),
        aversion=rng.uniform(0.0, 1.0, size=n),
    )


def test_apply_trade_conserves_wealth() -> None:
    state = SimulationState(wealth=np.array([1.0, 2.0]), aversion=np.array([0.2, 0.3]))
    total_before = state.total_wealth
    apply_trade(state, winner=0, loser=1, stake=0.5)
    assert abs(state.total_wealth - total_before) < 1e-12


def test_compute_stake_is_min_of_available_amounts() -> None:
    # Disponíveis: (1 - 0.25) * 1.0 = 0.75 e (1 - 0.75) * 2.0 = 0.5
    state = SimulationState(wealth=np.array([1.0, 2.0]), aversion=np.array([0.25, 0.75]))
    stake = compute_stake(state, i=0, j=1)
    assert abs(stake - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Conservação e não negatividade em nível de troca
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.0, 0.05, 0.3, 0.7, 1.0])
def test_apply_trade_with_tax_conserves_total_and_splits_stake(tau: float) -> None:
    state = SimulationState(wealth=np.array([3.0, 2.0, 1.0]), aversion=np.zeros(3))
    total_before = state.total_wealth + state.cash
    wealth_before = state.wealth.copy()
    stake = 0.8
    apply_trade(state, winner=0, loser=2, stake=stake, transaction_tax_rate=tau)
    assert state.total_wealth + state.cash == pytest.approx(total_before, abs=1e-12)
    assert state.wealth[0] - wealth_before[0] == pytest.approx((1.0 - tau) * stake)
    assert state.cash == pytest.approx(tau * stake)
    assert wealth_before[2] - state.wealth[2] == pytest.approx(stake)
    assert state.wealth[1] == wealth_before[1]


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.5, 1.0])
def test_many_random_trades_keep_wealth_non_negative_and_conserved(tau: float) -> None:
    rng = np.random.default_rng(123)
    state = _random_state(rng, 20)
    total_before = state.total_wealth
    for _ in range(5000):
        yard_sale_step(state, rng, transaction_tax_rate=tau)
        assert state.wealth.min() >= 0.0
    assert state.total_wealth + state.cash == pytest.approx(total_before, abs=1e-9)


# ---------------------------------------------------------------------------
# Casos pequenos verificáveis à mão
# ---------------------------------------------------------------------------


def test_stake_equals_wealth_when_no_risk_aversion_and_equal_wealth() -> None:
    state = SimulationState(wealth=np.array([2.5, 2.5]), aversion=np.zeros(2))
    assert compute_stake(state, 0, 1) == pytest.approx(2.5)


def test_stake_is_zero_when_one_agent_has_no_wealth() -> None:
    state = SimulationState(wealth=np.array([0.0, 4.0]), aversion=np.array([0.0, 0.0]))
    assert compute_stake(state, 0, 1) == 0.0
    before = state.wealth.copy()
    rng = np.random.default_rng(0)
    yard_sale_step(state, rng)
    np.testing.assert_array_equal(state.wealth, before)


def test_stake_vanishes_as_aversion_approaches_one() -> None:
    state = SimulationState(wealth=np.array([5.0, 5.0]), aversion=np.array([1.0 - 1e-9, 0.0]))
    assert compute_stake(state, 0, 1) == pytest.approx(5e-9, rel=1e-6)


def test_two_agents_total_conserved_and_loser_can_reach_zero_but_never_negative() -> None:
    # Sem aversão e riqueza igual, a aposta é toda a riqueza: o perdedor zera.
    state = SimulationState(wealth=np.array([1.0, 1.0]), aversion=np.zeros(2))
    rng = np.random.default_rng(5)
    result = yard_sale_step(state, rng)
    assert state.wealth[result.loser] == 0.0
    assert state.wealth[result.winner] == pytest.approx(2.0)
    # Muitas rodadas: nunca negativo, total conservado.
    rng = np.random.default_rng(6)
    state = SimulationState(wealth=np.array([1.0, 1.0]), aversion=rng.uniform(0, 1, 2))
    for _ in range(2000):
        yard_sale_step(state, rng)
        assert state.wealth.min() >= 0.0
        assert state.total_wealth == pytest.approx(2.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Propriedades estatísticas da regra de troca (sementes fixas, tolerância ~4 sigma)
# ---------------------------------------------------------------------------


def test_decide_winner_is_fair() -> None:
    n = 20000
    rng = np.random.default_rng(2024)
    wins_i = sum(decide_winner(0, 1, rng)[0] == 0 for _ in range(n))
    sigma = np.sqrt(0.25 / n)
    assert abs(wins_i / n - 0.5) < 4 * sigma


def test_winner_probability_is_independent_of_wealth() -> None:
    # Um agente rico (100) e um pobre (1): a aposta é sempre 1 e o rico deve vencer ~50%.
    n = 4000
    rng = np.random.default_rng(77)
    rich_wins = 0
    for _ in range(n):
        state = SimulationState(wealth=np.array([100.0, 1.0]), aversion=np.zeros(2))
        result = yard_sale_step(state, rng)
        assert result.stake == pytest.approx(1.0)
        rich_wins += result.winner == 0
    sigma = np.sqrt(0.25 / n)
    assert abs(rich_wins / n - 0.5) < 4 * sigma


def test_expected_wealth_change_is_zero_without_tax() -> None:
    # Martingale: E[Δw_i | i selecionado] = 0. Estado fixo, uma troca por semente.
    wealth0 = np.array([1.0, 2.0, 3.0])
    aversion = np.array([0.2, 0.5, 0.1])
    deltas: list[list[float]] = [[], [], []]
    for seed in range(6000):
        rng = np.random.default_rng(seed)
        state = SimulationState(wealth=wealth0.copy(), aversion=aversion.copy())
        result = yard_sale_step(state, rng)
        for k in (result.i, result.j):
            deltas[k].append(float(state.wealth[k] - wealth0[k]))
    for k in range(3):
        d = np.array(deltas[k])
        assert len(d) > 3000  # cada agente é selecionado em ~2/3 das tentativas
        stderr = d.std(ddof=1) / np.sqrt(len(d))
        assert stderr > 0.0
        assert abs(d.mean()) < 4 * stderr


def test_select_agents_returns_distinct_indices() -> None:
    rng = np.random.default_rng(1)
    for n_agents in (2, 3, 10, 1000):
        for _ in range(500):
            i, j = select_agents(n_agents, rng)
            assert i != j
            assert 0 <= i < n_agents and 0 <= j < n_agents


def test_select_agents_pairs_are_uniform() -> None:
    n_agents, draws = 5, 20000
    rng = np.random.default_rng(99)
    counts: dict[tuple[int, int], int] = {}
    for _ in range(draws):
        i, j = select_agents(n_agents, rng)
        key = (min(i, j), max(i, j))
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 10  # todos os pares não ordenados aparecem
    p = 1.0 / 10
    sigma = np.sqrt(p * (1 - p) / draws)
    for count in counts.values():
        assert abs(count / draws - p) < 4 * sigma
