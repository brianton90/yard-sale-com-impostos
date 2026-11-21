import numpy as np
from config import SimulationConfig
from simulation.model import SimulationState

def initialize_state(
    config: SimulationConfig,
    rng: np.random.Generator,
    aversion: np.ndarray | None = None,
) -> SimulationState:
    wealth = np.full(config.n_agents, config.initial_wealth, dtype=float)
    if aversion is None:
        state_aversion = rng.uniform(0.0, 1.0, size=config.n_agents)
    else:
        state_aversion = np.asarray(aversion, dtype=float).copy()
        if state_aversion.shape != (config.n_agents,):
            raise ValueError("aversion tem shape incompatível")
    return SimulationState(wealth=wealth, aversion=state_aversion)
