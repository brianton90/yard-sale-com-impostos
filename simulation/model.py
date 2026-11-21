from dataclasses import dataclass
import numpy as np


##estado da simulação

@dataclass
class SimulationState:
    wealth: np.ndarray
    aversion: np.ndarray
    cash: float = 0.0  # caixa C(t): recursos arrecadados e ainda não devolvidos

    @property
    def total_wealth(self) -> float:
        """Riqueza em posse dos agentes (não inclui o caixa)."""
        return float(self.wealth.sum())


@dataclass(frozen=True)
class SimulationHistory:
    """Snapshots da riqueza ao longo da simulação.

    steps: shape (n_snapshots,), passo de Monte Carlo de cada snapshot (0 = estado inicial).
    wealth: shape (n_snapshots, n_agents), riqueza de cada agente em cada snapshot.
    """

    steps: np.ndarray
    wealth: np.ndarray
