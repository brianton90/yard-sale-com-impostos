import json
from pathlib import Path

import pytest

from experiments.pilot_configuration import (
    PilotConfig,
    load_stability_margins,
    protocol_document,
    validate_pilot_id,
)


EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "stability_margins.example.json"


def test_example_margins_and_canonical_pilot_protocol_load() -> None:
    margins = load_stability_margins(EXAMPLE)
    config = PilotConfig()
    protocol = protocol_document(config, margins)

    assert margins.K == 4
    assert margins.confidence == pytest.approx(0.95)
    assert config.n_steps == 600
    assert len(config.configurations) == 9
    assert protocol["stability_margins_sha256"] == margins.sha256
    assert protocol["protocol_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unknown": 1}, "campos ausentes ou desconhecidos"),
        ({"confidence": float("nan")}, "confidence deve ser finito"),
        ({"K": 0}, "K deve ser inteiro >= 2"),
        ({"margins": {"gini": {"absolute": 0.0}, "max_share": {"absolute": 0.01}, "volume": {"absolute": 0.01, "relative": 0.1}}}, "margins.gini"),
    ],
)
def test_margins_loader_rejects_invalid_values(tmp_path: Path, mutation: dict[str, object], message: str) -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data.update(mutation)
    path = tmp_path / "margins.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_stability_margins(path)


@pytest.mark.parametrize("pilot_id", ["../unsafe", "unsafe/name", "", "."])
def test_pilot_id_rejects_unsafe_paths(pilot_id: str) -> None:
    with pytest.raises(ValueError):
        validate_pilot_id(pilot_id)


@pytest.mark.parametrize("relative_path", ["Results/experiments", "Results/experiments/protected"])
def test_pilot_results_root_rejects_protected_experiment_area(relative_path: str) -> None:
    with pytest.raises(ValueError, match="results_root não pode"):
        PilotConfig(results_root=relative_path)
