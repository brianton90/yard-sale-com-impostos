import pytest

from config import TAX_GRID
from experiments.configuration import ExperimentConfig, build_configurations, configuration_tag
from experiments.run_experiment import main


def test_canonical_grid_and_tags() -> None:
    configurations = build_configurations()
    assert TAX_GRID == (0.05, 0.10, 0.20, 0.30)
    assert [item.tag for item in configurations] == [
        "none",
        "wealth_tau0.05", "wealth_tau0.1", "wealth_tau0.2", "wealth_tau0.3",
        "transaction_tau0.05", "transaction_tau0.1", "transaction_tau0.2", "transaction_tau0.3",
    ]
    assert configurations[0].tax_rate == 0.0
    assert configuration_tag("wealth", 0.10) == "wealth_tau0.1"


def test_protocol_defaults_and_validation() -> None:
    protocol = ExperimentConfig()
    assert protocol.n_realizations == 10
    assert protocol.required_snapshot_steps == (0, 150, 200, 300)
    document = protocol.to_dict()
    assert document["temporal_parameters_provisional"] is True
    assert document["interpretation"] == "finite_time"
    assert set(document["stability_status_by_configuration"].values()) == {"not_assessed"}
    with pytest.raises(ValueError):
        ExperimentConfig(n_realizations=1)
    with pytest.raises(ValueError):
        ExperimentConfig(n_agents=22)
    with pytest.raises(ValueError):
        ExperimentConfig(tax_rates=(0.05,))
    with pytest.raises(ValueError):
        ExperimentConfig(transient_start=280, mobility_delta=50)


def test_experiment_id_cannot_escape_results_root(tmp_path) -> None:
    for experiment_id in (".", "..", "nested/name", "/absolute"):
        with pytest.raises(ValueError):
            ExperimentConfig(experiment_id=experiment_id)
    with pytest.raises(ValueError):
        main([
            "analyze",
            "--experiment-id", "..",
            "--results-root", str(tmp_path / "results"),
        ])
    assert not (tmp_path / "protocol.json").exists()
