import csv

from analysis.reporting import (
    FIGURE_NAMES,
    LORENZ_TAGS,
    MOBILITY_TAGS,
    TABLE_NAMES,
    generate_report,
)
from experiments.configuration import ExperimentConfig
from experiments.runner import run_experiment


def test_report_contains_predefined_analyses_and_all_outputs(tmp_path) -> None:
    config = ExperimentConfig(
        experiment_id="report",
        results_root=str(tmp_path),
        n_realizations=2,
        n_agents=20,
        n_steps=8,
        transient_start=3,
        analysis_end=8,
        mobility_delta=2,
        snapshot_interval=4,
    )
    run_experiment(config)
    report = generate_report(config.experiment_dir)

    assert {path.name for path in report.tables} == set(TABLE_NAMES)
    assert {path.name for path in report.figures} == set(FIGURE_NAMES)
    assert all(path.is_file() for path in (*report.tables, *report.figures))

    with (config.experiment_dir / "tables" / "lorenz_mean_se.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        lorenz_rows = list(csv.DictReader(handle))
    assert {row["configuration_tag"] for row in lorenz_rows} == set(LORENZ_TAGS)
    assert {row["interpretation"] for row in lorenz_rows} == {"finite_time"}

    with (config.experiment_dir / "tables" / "mobility_matrices.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        mobility_rows = list(csv.DictReader(handle))
    assert {row["configuration_tag"] for row in mobility_rows} == set(MOBILITY_TAGS)
    assert len(mobility_rows) == len(MOBILITY_TAGS) * 25
