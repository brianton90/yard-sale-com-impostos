import numpy as np

from config import SimulationConfig
from experiments.randomization import generate_all_randomness
from simulation.engine import run_simulation


def test_injected_randomness_is_shared_unmodified_and_snapshots_are_forced() -> None:
    randomness, _ = generate_all_randomness(7, 2, 20, 8)[0]
    before = (
        randomness.aversion.copy(), randomness.pairs.copy(), randomness.first_agent_wins.copy()
    )
    results = []
    for regime in ("none", "wealth", "transaction"):
        config = SimulationConfig(
            n_agents=20, n_steps=8, snapshot_interval=4,
            regime=regime, tax_rate=0.05,
        )
        results.append(run_simulation(config, randomness, (0, 3, 5, 8)))
    np.testing.assert_array_equal(randomness.aversion, before[0])
    np.testing.assert_array_equal(randomness.pairs, before[1])
    np.testing.assert_array_equal(randomness.first_agent_wins, before[2])
    for result in results:
        assert list(result.history.steps) == [0, 3, 4, 5, 8]
        np.testing.assert_array_equal(result.state.aversion, randomness.aversion)
    assert not np.array_equal(results[0].state.wealth, results[1].state.wealth)
