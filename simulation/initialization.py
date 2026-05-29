import numpy as np
from config import SimulationConfig
from simulation.model import SimulationState

def initialize_state(config: SimulationConfig, rng: np.random.Generator) -> SimulationState:
    wealth = np.full(config.n_agents, config.initial_wealth, dtype=float)
    aversion = rng.uniform(0.0, 1.0, size=config.n_agents)
    return SimulationState(wealth=wealth, aversion=aversion)