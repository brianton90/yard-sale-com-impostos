from pathlib import Path

import numpy as np
import pytest

from analysis.reporting import generate_report
from experiments.configuration import ExperimentConfig
from experiments.runner import run_experiment
from experiments.storage import (
    atomic_write_json,
    atomic_write_npz,
    load_npz,
    read_json,
)


def _small_config(tmp_path, experiment_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        results_root=str(tmp_path),
        n_realizations=2,
        n_agents=20,
        n_steps=8,
        transient_start=3,
        analysis_end=8,
        mobility_delta=2,
        snapshot_interval=4,
    )


def test_runner_creates_18_results_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config(tmp_path, "runner")
    first = run_experiment(config)
    assert (first.completed, first.skipped, first.total) == (18, 0, 18)
    hashes = []
    for realization_id in range(2):
        directory = tmp_path / "runner" / "raw" / f"realization_{realization_id:03d}"
        files = sorted(path for path in directory.glob("*.npz") if path.name != "common_inputs.npz")
        assert len(files) == 9
        realization_hashes = {load_npz(path)["common_randomness_sha256"].item() for path in files}
        assert len(realization_hashes) == 1
        hashes.extend(realization_hashes)
    assert hashes[0] != hashes[1]

    manifest_path = config.experiment_dir / "manifest.json"
    manifest = read_json(manifest_path)
    interrupted_entry = manifest["realizations"]["000"]["configurations"]["none"]
    interrupted_entry.clear()
    interrupted_entry["status"] = "running"
    atomic_write_json(manifest_path, manifest)

    def fail_if_resimulated(*args: object, **kwargs: object) -> None:
        raise AssertionError("resultado publicado foi simulado novamente")

    monkeypatch.setattr("experiments.runner.run_simulation", fail_if_resimulated)
    second = run_experiment(config)
    assert (second.completed, second.skipped) == (0, 18)
    recovered = read_json(manifest_path)["realizations"]["000"]["configurations"]["none"]
    assert recovered["status"] == "completed"
    assert recovered["file"] == "raw/realization_000/none.npz"


def test_invalid_published_result_marks_interrupted_run_failed(tmp_path: Path) -> None:
    config = _small_config(tmp_path, "invalid-interrupted")
    run_experiment(config)
    result_path = config.experiment_dir / "raw" / "realization_000" / "none.npz"
    data = load_npz(result_path)
    data["tax_rate"] = np.array(0.3)
    atomic_write_npz(result_path, **data)

    manifest_path = config.experiment_dir / "manifest.json"
    manifest = read_json(manifest_path)
    entry = manifest["realizations"]["000"]["configurations"]["none"]
    entry.clear()
    entry["status"] = "running"
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="tax_rate incompatível"):
        run_experiment(config)

    failed = read_json(manifest_path)["realizations"]["000"]["configurations"]["none"]
    assert failed["status"] == "failed"
    assert "tax_rate incompatível" in failed["error"]


@pytest.mark.parametrize("field", ["mobility_tiebreak", "pairs"])
def test_common_input_corruption_blocks_resume_and_analysis(tmp_path, field) -> None:
    config = _small_config(tmp_path, f"common-{field}")
    run_experiment(config)
    path = config.experiment_dir / "raw" / "realization_000" / "common_inputs.npz"
    data = load_npz(path)
    corrupted = data[field].copy()
    if field == "mobility_tiebreak":
        corrupted[[0, 1]] = corrupted[[1, 0]]
    else:
        corrupted[0, 0] = corrupted[0, 0, ::-1]
    data[field] = corrupted
    atomic_write_npz(path, **data)

    with pytest.raises(ValueError, match="hash de aleatoriedade comum"):
        run_experiment(config)
    with pytest.raises(ValueError, match="hash de aleatoriedade comum"):
        generate_report(config.experiment_dir)


@pytest.mark.parametrize("mutation", ["missing", "derived"])
def test_noncanonical_protocol_blocks_resume_and_analysis(tmp_path, mutation) -> None:
    config = _small_config(tmp_path, f"protocol-{mutation}")
    run_experiment(config)
    path = config.experiment_dir / "protocol.json"
    protocol = read_json(path)
    if mutation == "missing":
        del protocol["configuration_tags"]
    else:
        protocol["temporal_parameters_provisional"] = False
    atomic_write_json(path, protocol)

    with pytest.raises(ValueError, match="protocolo incompatível"):
        run_experiment(config)
    with pytest.raises(ValueError, match="protocolo"):
        generate_report(config.experiment_dir)


@pytest.mark.parametrize("mutation", ["tax_rate", "missing_metric", "final_snapshot"])
def test_result_corruption_blocks_resume_and_analysis(tmp_path, mutation) -> None:
    config = _small_config(tmp_path, f"result-{mutation}")
    run_experiment(config)
    path = config.experiment_dir / "raw" / "realization_000" / "none.npz"
    data = load_npz(path)
    if mutation == "tax_rate":
        data["tax_rate"] = np.array(0.3)
    elif mutation == "missing_metric":
        del data["final_gini"]
    else:
        data["final_wealth"] = data["final_wealth"].copy()
        data["final_wealth"][0] += 1.0
    atomic_write_npz(path, **data)

    with pytest.raises(ValueError):
        run_experiment(config)
    with pytest.raises(ValueError):
        generate_report(config.experiment_dir)
