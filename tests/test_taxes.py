import numpy as np
import pytest

from simulation.model import SimulationState
from simulation.taxes import apply_wealth_tax, redistribute_cash


def _state(seed: int, n: int = 50) -> SimulationState:
    rng = np.random.default_rng(seed)
    return SimulationState(wealth=rng.exponential(1.0, size=n), aversion=rng.uniform(0, 1, size=n))


def test_redistribute_cash_zeroes_cash_and_conserves_total() -> None:
    state = _state(0)
    state.cash = 3.7
    total_before = state.total_wealth + state.cash
    cash_before = state.cash
    share = redistribute_cash(state)
    assert state.cash == 0.0
    assert share == pytest.approx(cash_before / len(state.wealth))
    assert state.total_wealth + state.cash == pytest.approx(total_before, abs=1e-12)


def test_redistribute_cash_adds_same_share_to_everyone() -> None:
    state = _state(1)
    before = state.wealth.copy()
    state.cash = 5.0
    share = redistribute_cash(state)
    np.testing.assert_allclose(state.wealth - before, share)


def test_wealth_tax_with_zero_rate_changes_nothing() -> None:
    state = _state(2)
    before = state.wealth.copy()
    collected = apply_wealth_tax(state, 0.0)
    assert collected == 0.0
    assert state.cash == 0.0
    np.testing.assert_array_equal(state.wealth, before)


@pytest.mark.parametrize("rate", [0.01, 0.2, 0.5, 1.0])
def test_wealth_tax_collects_rate_times_total_and_conserves(rate: float) -> None:
    state = _state(3)
    before = state.wealth.copy()
    total_before = state.total_wealth
    collected = apply_wealth_tax(state, rate)
    assert collected == pytest.approx(rate * before.sum())
    assert state.cash == pytest.approx(collected)
    np.testing.assert_allclose(state.wealth, (1.0 - rate) * before)
    assert state.total_wealth + state.cash == pytest.approx(total_before, abs=1e-12)


def test_wealth_tax_is_order_independent() -> None:
    state = _state(4)
    perm = np.random.default_rng(10).permutation(len(state.wealth))
    permuted = SimulationState(wealth=state.wealth[perm].copy(), aversion=state.aversion[perm].copy())
    collected = apply_wealth_tax(state, 0.3)
    collected_permuted = apply_wealth_tax(permuted, 0.3)
    assert collected == pytest.approx(collected_permuted, rel=1e-12)
    np.testing.assert_allclose(permuted.wealth, state.wealth[perm])


@pytest.mark.parametrize("tau", [0.0, 0.05, 0.3, 0.8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wealth_tax_plus_redistribution_contracts_towards_mean(tau: float, seed: int) -> None:
    # Identidade: w' - média = (1 - τ)(w - média); a média é preservada.
    state = _state(seed, n=100)
    wealth = state.wealth.copy()
    mean = wealth.mean()
    apply_wealth_tax(state, tau)
    redistribute_cash(state)
    assert state.wealth.mean() == pytest.approx(mean)
    np.testing.assert_allclose(state.wealth - mean, (1.0 - tau) * (wealth - mean), atol=1e-12)


def test_full_wealth_tax_equalizes_everyone_to_the_mean() -> None:
    state = _state(5, n=80)
    mean = state.wealth.mean()
    apply_wealth_tax(state, 1.0)
    redistribute_cash(state)
    np.testing.assert_allclose(state.wealth, mean, atol=1e-12)
