from dataclasses import dataclass
import numpy as np

from simulation.model import SimulationState


@dataclass(frozen=True)
class TradeResult:
    i: int
    j: int
    stake: float
    winner: int
    loser: int


def select_agents(n_agents: int, rng: np.random.Generator) -> tuple[int, int]:
    """Randomly select two distinct agents."""
    i, j = rng.choice(n_agents, size=2, replace=False)
    return int(i), int(j)


def compute_stake(state: SimulationState, i: int, j: int) -> float:
    """Valor bruto da troca (regra fair): mínimo dos montantes disponibilizados pelos dois agentes.

    a_e = min{(1 - beta_i) * w_i, (1 - beta_j) * w_j}
    """
    available_i = (1.0 - state.aversion[i]) * state.wealth[i]
    available_j = (1.0 - state.aversion[j]) * state.wealth[j]
    return float(min(available_i, available_j))


def decide_winner(i: int, j: int, rng: np.random.Generator) -> tuple[int, int]:
    """Choose the winner with 50% probability."""
    if rng.random() <= 0.5:
        return i, j
    return j, i


def apply_trade(
    state: SimulationState,
    winner: int,
    loser: int,
    stake: float,
    transaction_tax_rate: float = 0.0,
) -> None:
    """Atualiza riqueza e caixa no lugar.

    O perdedor perde `stake`. O vencedor recebe `(1 - tau_X) * stake` e o imposto
    `tau_X * stake` vai para o caixa.
    """
    tax = transaction_tax_rate * stake
    state.wealth[winner] += stake - tax
    state.wealth[loser] -= stake
    state.cash += tax


def yard_sale_step(
    state: SimulationState,
    rng: np.random.Generator,
    transaction_tax_rate: float = 0.0,
) -> TradeResult:
    """Execute one Yard-Sale interaction."""
    i, j = select_agents(len(state.wealth), rng)
    stake = compute_stake(state, i, j)
    winner, loser = decide_winner(i, j, rng)
    apply_trade(state, winner, loser, stake, transaction_tax_rate)
    return TradeResult(i=i, j=j, stake=stake, winner=winner, loser=loser)
