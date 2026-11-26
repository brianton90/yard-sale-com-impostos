from pathlib import Path

import pytest

from experiments.pilot_configuration import PilotConfig, load_stability_margins
from experiments.pilot_runner import run_pilot
from experiments.storage import atomic_write_json, load_npz, read_json


EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "stability_margins.example.json"


def _pilot(tmp_path: Path, pilot_id: str = "runner") -> tuple[PilotConfig, object]:
    return (
        PilotConfig(
            pilot_id=pilot_id, results_root=str(tmp_path), n_realizations=2,
            n_agents=20, t_horizon=8, snapshot_interval=4,
        ),
        load_stability_margins(EXAMPLE),
    )


def test_runner_creates_one_2t_result_per_scenario_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, margins = _pilot(tmp_path)
    first = run_pilot(config, margins)
    assert (first.completed, first.skipped, first.total) == (18, 0, 18)
    for realization_id in range(2):
        directory = config.pilot_dir / "raw" / f"realization_{realization_id:03d}"
        files = sorted(path for path in directory.glob("*.npz") if path.name != "common_inputs.npz")
        assert len(files) == 9
        assert all(load_npz(path)["gini"].shape == (16,) for path in files)
        assert len({load_npz(path)["common_randomness_sha256"].item() for path in files}) == 1

    def fail_if_simulated(*args: object, **kwargs: object) -> object:
        raise AssertionError("resultado publicado foi simulado novamente")

    monkeypatch.setattr("experiments.pilot_runner.run_simulation", fail_if_simulated)
    second = run_pilot(config, margins)
    assert (second.completed, second.skipped) == (0, 18)


def test_corrupted_protocol_and_margin_artifacts_are_rejected(tmp_path: Path) -> None:
    config, margins = _pilot(tmp_path, "artifacts")
    run_pilot(config, margins)
    protocol_path = config.pilot_dir / "protocol.json"
    protocol = read_json(protocol_path)
    protocol["n_agents"] = 999
    atomic_write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="protocolo incompatível"):
        run_pilot(config, margins)

    config, margins = _pilot(tmp_path, "margins")
    run_pilot(config, margins)
    margins_path = config.pilot_dir / "stability_margins.json"
    saved_margins = read_json(margins_path)
    saved_margins["K"] = 5
    atomic_write_json(margins_path, saved_margins)
    with pytest.raises(ValueError, match="margens de estabilidade incompatíveis"):
        run_pilot(config, margins)


def test_persisted_artifacts_reject_boolean_numeric_substitutions(tmp_path: Path) -> None:
    config, margins = _pilot(tmp_path, "boolean-protocol")
    run_pilot(config, margins)
    protocol_path = config.pilot_dir / "protocol.json"
    protocol = read_json(protocol_path)
    protocol["schema_version"] = True
    atomic_write_json(protocol_path, protocol)

    with pytest.raises(ValueError, match="protocolo incompatível"):
        run_pilot(config, margins)

    config, margins = _pilot(tmp_path, "boolean-margins")
    run_pilot(config, margins)
    margins_path = config.pilot_dir / "stability_margins.json"
    saved_margins = read_json(margins_path)
    saved_margins["schema_version"] = True
    atomic_write_json(margins_path, saved_margins)

    with pytest.raises(ValueError, match="margens de estabilidade incompatíveis"):
        run_pilot(config, margins)
