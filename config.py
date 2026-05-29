from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the Yard-Sale simulation."""

    n_agents: int = 10000
    n_steps: int = 300
    initial_wealth: float = 1.0
    seed: int = 42
    hist_bins: int = 50
