import numpy as np

from simulation.model import SimulationState
from simulation.rules import apply_trade, compute_stake


def test_apply_trade_conserves_wealth() -> None:
    state = SimulationState(wealth=np.array([1.0, 2.0]), aversion=np.array([0.2, 0.3]))
    total_before = state.total_wealth
    apply_trade(state, winner=0, loser=1, stake=0.5)
    assert abs(state.total_wealth - total_before) < 1e-12


def test_compute_stake_uses_poorer_aversion() -> None:
    state = SimulationState(wealth=np.array([1.0, 2.0]), aversion=np.array([0.25, 0.75]))
    stake = compute_stake(state, poorer=0)
    assert abs(stake - 0.75) < 1e-12
