from dataclasses import dataclass
import numpy as np


##estado da simulação

@dataclass
class SimulationState:
    wealth: np.ndarray
    aversion: np.ndarray

    @property
    def total_wealth(self) -> float:
        return float(self.wealth.sum())