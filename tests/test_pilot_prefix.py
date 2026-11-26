from pathlib import Path

import numpy as np

from config import SimulationConfig
from experiments.pilot_configuration import PilotConfig, load_stability_margins
from experiments.pilot_runner import run_pilot
from experiments.pilot_validation import load_and_validate_common
from experiments.randomization import SimulationRandomness
from experiments.storage import load_npz
from simulation.engine import run_simulation


EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "stability_margins.example.json"


def test_t_is_a_bit_identical_prefix_of_persisted_2t_randomness(tmp_path: Path) -> None:
    config = PilotConfig(
        pilot_id="prefix", results_root=str(tmp_path), n_realizations=2,
        n_agents=20, t_horizon=8, snapshot_interval=4,
    )
    margins = load_stability_margins(EXAMPLE)
    run_pilot(config, margins)
    protocol_hash = str(load_npz(config.pilot_dir / "raw" / "realization_000" / "none.npz")["protocol_sha256"].item())
    randomness, _, _ = load_and_validate_common(
        config.pilot_dir / "raw" / "realization_000" / "common_inputs.npz",
        config, 0, protocol_hash,
    )
    prefix = SimulationRandomness(
        aversion=randomness.aversion,
        pairs=randomness.pairs[:config.t_horizon],
        first_agent_wins=randomness.first_agent_wins[:config.t_horizon],
    )
    direct_config = SimulationConfig(
        n_agents=config.n_agents,
        n_steps=config.t_horizon,
        initial_wealth=config.initial_wealth,
        seed=config.base_seed,
        snapshot_interval=config.snapshot_interval,
        regime="none",
        tax_rate=0.0,
    )
    direct = run_simulation(direct_config, randomness=prefix)
    persisted = load_npz(config.pilot_dir / "raw" / "realization_000" / "none.npz")

    np.testing.assert_array_equal(direct.series.gini, persisted["gini"][:config.t_horizon])
    np.testing.assert_array_equal(direct.series.max_share, persisted["max_share"][:config.t_horizon])
    np.testing.assert_array_equal(direct.series.volume, persisted["volume"][:config.t_horizon])
