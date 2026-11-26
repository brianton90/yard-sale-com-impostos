from pathlib import Path

import pytest

from analysis.pilot_reporting import FIGURE_NAMES, TABLE_NAMES, generate_pilot_report
from experiments.pilot_configuration import PilotConfig, load_stability_margins
from experiments.pilot_runner import run_pilot
from experiments.storage import atomic_write_json, read_json


EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "stability_margins.example.json"


def test_reporting_rebuilds_derived_artifacts_without_simulation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = PilotConfig(pilot_id="report", results_root=str(tmp_path), n_realizations=2, n_agents=20, t_horizon=8, snapshot_interval=4)
    run_pilot(config, load_stability_margins(EXAMPLE))
    first = generate_pilot_report(config.pilot_dir)
    assert {path.name for path in first.tables} == set(TABLE_NAMES)
    assert {path.name for path in first.figures} == set(FIGURE_NAMES)
    assert first.report.is_file() and first.timing.is_file()
    for path in (*first.tables, *first.figures, first.report):
        path.unlink()

    def fail_if_simulated(*args: object, **kwargs: object) -> object:
        raise AssertionError("reanálise não deve executar simulação")

    monkeypatch.setattr("experiments.pilot_runner.run_simulation", fail_if_simulated)
    rebuilt = generate_pilot_report(config.pilot_dir)
    assert all(path.is_file() for path in (*rebuilt.tables, *rebuilt.figures, rebuilt.report, rebuilt.timing))


def test_reporting_rejects_boolean_margin_schema_version(tmp_path: Path) -> None:
    config = PilotConfig(pilot_id="boolean-margin", results_root=str(tmp_path), n_realizations=2, n_agents=20, t_horizon=8, snapshot_interval=4)
    run_pilot(config, load_stability_margins(EXAMPLE))
    margins_path = config.pilot_dir / "stability_margins.json"
    margins = read_json(margins_path)
    margins["schema_version"] = True
    atomic_write_json(margins_path, margins)

    with pytest.raises(ValueError, match="schema_version deve ser 1"):
        generate_pilot_report(config.pilot_dir)
