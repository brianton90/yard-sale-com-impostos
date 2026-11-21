import os
from pathlib import Path
import subprocess
import sys

from analysis.reporting import FIGURE_NAMES, TABLE_NAMES


def test_cli_all_resume_and_disk_only_analysis(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, "-m", "experiments.run_experiment", "all",
        "--experiment-id", "integration", "--results-root", str(tmp_path),
        "--n-realizations", "2", "--n-agents", "20", "--n-steps", "8",
        "--transient-start", "3", "--analysis-end", "8",
        "--mobility-delta", "2", "--snapshot-interval", "4",
    ]
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    first = subprocess.run(command, cwd=root, env=environment, check=True, capture_output=True, text=True)
    assert "completed=18" in first.stdout
    experiment = tmp_path / "integration"
    assert len(list((experiment / "raw").glob("realization_*/*.npz"))) == 20
    assert {path.name for path in (experiment / "tables").iterdir()} == set(TABLE_NAMES)
    assert {path.name for path in (experiment / "figures").iterdir()} == set(FIGURE_NAMES)

    simulate = command.copy()
    simulate[3] = "simulate"
    second = subprocess.run(simulate, cwd=root, env=environment, check=True, capture_output=True, text=True)
    assert "completed=0" in second.stdout and "skipped=18" in second.stdout

    for path in (experiment / "tables").iterdir():
        path.unlink()
    for path in (experiment / "figures").iterdir():
        path.unlink()
    analyze = [
        sys.executable, "-m", "experiments.run_experiment", "analyze",
        "--experiment-id", "integration", "--results-root", str(tmp_path),
    ]
    subprocess.run(analyze, cwd=root, env=environment, check=True, capture_output=True, text=True)
    assert {path.name for path in (experiment / "tables").iterdir()} == set(TABLE_NAMES)
    assert {path.name for path in (experiment / "figures").iterdir()} == set(FIGURE_NAMES)
