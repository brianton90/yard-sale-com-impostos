import numpy as np

from config import SimulationConfig
from simulation.initialization import initialize_state
from simulation.metrics import SimulationMetrics, summarize_wealth
from simulation.model import SimulationState
from simulation.rules import yard_sale_step


def run_simulation(config: SimulationConfig) -> tuple[SimulationState, SimulationMetrics]:
    """Run the Yard-Sale simulation and return final state and summary metrics."""
    rng = np.random.default_rng(config.seed)
    state = initialize_state(config, rng)

    for _ in range(config.n_steps):
        for _ in range(config.n_agents // 2):
            yard_sale_step(state, rng)

    metrics = summarize_wealth(state.wealth)
    return state, metrics
