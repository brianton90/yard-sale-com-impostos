from pathlib import Path

import numpy as np
import pytest

from experiments.pilot_configuration import PilotConfig, load_stability_margins, protocol_document
from experiments.pilot_runner import run_pilot
from experiments.pilot_validation import validate_result_file
from experiments.storage import atomic_write_npz, load_npz


EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "stability_margins.example.json"


def _pilot(tmp_path: Path) -> tuple[PilotConfig, object]:
    config = PilotConfig(
        pilot_id="validation", results_root=str(tmp_path), n_realizations=2,
        n_agents=20, t_horizon=8, snapshot_interval=4,
    )
    margins = load_stability_margins(EXAMPLE)
    return config, margins


def test_result_validation_rejects_corrupted_diagnostics(tmp_path: Path) -> None:
    config, margins = _pilot(tmp_path)
    run_pilot(config, margins)
    protocol = protocol_document(config, margins)
    path = config.pilot_dir / "raw" / "realization_000" / "none.npz"
    data = load_npz(path)
    data["max_conservation_error"] = np.array(float("nan"))
    atomic_write_npz(path, **data)

    with pytest.raises(ValueError, match="resultado não finito"):
        validate_result_file(
            path, config, 0, config.configurations[0],
            protocol["protocol_sha256"], margins.sha256,
            str(data["common_randomness_sha256"].item()),
        )


def test_corrupted_common_input_is_rejected(tmp_path: Path) -> None:
    config, margins = _pilot(tmp_path)
    run_pilot(config, margins)
    path = config.pilot_dir / "raw" / "realization_000" / "common_inputs.npz"
    data = load_npz(path)
    data["pairs"] = data["pairs"].copy()
    data["pairs"][0, 0] = data["pairs"][0, 0, ::-1]
    atomic_write_npz(path, **data)

    with pytest.raises(ValueError, match="hash de aleatoriedade comum"):
        run_pilot(config, margins)
