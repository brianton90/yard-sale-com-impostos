import os
from pathlib import Path
import subprocess
import sys

from analysis.pilot_reporting import FIGURE_NAMES, TABLE_NAMES


ROOT = Path(__file__).resolve().parents[1]
MARGINS = ROOT / "config" / "stability_margins.example.json"


def test_pilot_cli_all_resume_and_disk_only_analysis(tmp_path: Path) -> None:
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    command = [
        sys.executable, "-m", "experiments.run_pilot", "all", "--pilot-id", "integration",
        "--results-root", str(tmp_path), "--stability-config", str(MARGINS),
        "--n-realizations", "2", "--n-agents", "20", "--t-horizon", "8",
    ]
    first = subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
    assert "completed=18" in first.stdout
    pilot = tmp_path / "integration"
    assert len(list((pilot / "raw").glob("realization_*/*.npz"))) == 20
    assert {path.name for path in (pilot / "tables").iterdir()} == set(TABLE_NAMES)
    assert {path.name for path in (pilot / "figures").iterdir()} == set(FIGURE_NAMES)
    assert (pilot / "report.md").is_file() and (pilot / "timing.json").is_file()

    second = subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
    assert "completed=0" in second.stdout and "skipped=18" in second.stdout
    for path in (pilot / "tables").iterdir():
        path.unlink()
    for path in (pilot / "figures").iterdir():
        path.unlink()
    (pilot / "report.md").unlink()
    subprocess.run([
        sys.executable, "-m", "experiments.run_pilot", "analyze", "--pilot-id", "integration",
        "--results-root", str(tmp_path),
    ], cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
    assert {path.name for path in (pilot / "tables").iterdir()} == set(TABLE_NAMES)
    assert {path.name for path in (pilot / "figures").iterdir()} == set(FIGURE_NAMES)
