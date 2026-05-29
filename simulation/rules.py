from dataclasses import dataclass
import numpy as np

from simulation.model import SimulationState


@dataclass(frozen=True)
class TradeResult:
    i: int
    j: int
    poorer: int
    stake: float
    winner: int
    loser: int


def select_agents(n_agents: int, rng: np.random.Generator) -> tuple[int, int]:
    """Randomly select two distinct agents."""
    i, j = rng.choice(n_agents, size=2, replace=False)
    return int(i), int(j)


def compute_poorer_agent(wealth: np.ndarray, i: int, j: int) -> int:
    return i if wealth[i] < wealth[j] else j


def compute_stake(state: SimulationState, poorer: int) -> float:
    """Amount placed at risk by the poorer agent."""
    return (1.0 - state.aversion[poorer]) * state.wealth[poorer]


def decide_winner(i: int, j: int, rng: np.random.Generator) -> tuple[int, int]:
    """Choose the winner with 50% probability."""
    if rng.random() <= 0.5:
        return i, j
    return j, i


def apply_trade(state: SimulationState, winner: int, loser: int, stake: float) -> None:
    """Update wealth arrays in place."""
    state.wealth[winner] += stake
    state.wealth[loser] -= stake


def yard_sale_step(state: SimulationState, rng: np.random.Generator) -> TradeResult:
    """Execute one Yard-Sale interaction."""
    i, j = select_agents(len(state.wealth), rng)
    poorer = compute_poorer_agent(state.wealth, i, j)
    stake = compute_stake(state, poorer)
    winner, loser = decide_winner(i, j, rng)
    apply_trade(state, winner, loser, stake)
    return TradeResult(i=i, j=j, poorer=poorer, stake=stake, winner=winner, loser=loser)
