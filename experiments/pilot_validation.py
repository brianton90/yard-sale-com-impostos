from pathlib import Path
from typing import Any

import numpy as np

from experiments.configuration import ConfigurationKey
from experiments.pilot_configuration import PilotConfig
from experiments.randomization import SimulationRandomness, common_randomness_sha256
from experiments.storage import load_npz
from simulation.metrics import compute_gini, compute_max_share


COMMON_FIELDS = {
    "aversion",
    "pairs",
    "first_agent_wins",
    "mobility_tiebreak",
    "realization_id",
    "protocol_sha256",
    "common_randomness_sha256",
}
RESULT_FIELDS = {
    "series_steps",
    "gini",
    "max_share",
    "volume",
    "revenue",
    "history_steps",
    "history_wealth",
    "final_wealth",
    "final_cash",
    "final_total_wealth",
    "final_mean_wealth",
    "final_median_wealth",
    "final_min_wealth",
    "final_max_wealth",
    "final_gini",
    "final_max_share",
    "max_conservation_error",
    "min_wealth_observed",
    "max_cash",
    "duration_seconds",
    "realization_id",
    "configuration_tag",
    "regime",
    "tax_rate",
    "protocol_sha256",
    "stability_margins_sha256",
    "common_randomness_sha256",
}


def scalar(data: dict[str, np.ndarray], name: str) -> Any:
    if name not in data or data[name].shape != ():
        raise ValueError(f"campo escalar ausente ou inválido: {name}")
    return data[name].item()


def common_metadata(config: PilotConfig, realization_id: int) -> dict[str, object]:
    return {
        "base_seed": config.base_seed,
        "realization_id": realization_id,
        "n_agents": config.n_agents,
        "n_steps": config.n_steps,
        "trajectory": "single_common_2T",
    }


def _finite_array(data: dict[str, np.ndarray], name: str, path: Path) -> None:
    if not np.issubdtype(data[name].dtype, np.number) or not np.all(np.isfinite(data[name])):
        raise ValueError(f"resultado não finito em {path}: {name}")


def load_and_validate_common(
    path: Path,
    config: PilotConfig,
    realization_id: int,
    protocol_sha256: str,
) -> tuple[SimulationRandomness, np.ndarray, str]:
    data = load_npz(path)
    if set(data) != COMMON_FIELDS:
        raise ValueError(f"esquema de entrada comum incompatível em {path}")
    if int(scalar(data, "realization_id")) != realization_id:
        raise ValueError(f"realização incompatível em {path}")
    if str(scalar(data, "protocol_sha256")) != protocol_sha256:
        raise ValueError(f"protocolo incompatível em {path}")

    randomness = SimulationRandomness(
        aversion=data["aversion"],
        pairs=data["pairs"],
        first_agent_wins=data["first_agent_wins"],
    )
    randomness.validate(config.n_agents, config.n_steps)
    for array in (randomness.aversion, randomness.pairs, randomness.first_agent_wins):
        array.setflags(write=False)

    tiebreak = data["mobility_tiebreak"]
    if (
        tiebreak.shape != (config.n_agents,)
        or not np.issubdtype(tiebreak.dtype, np.integer)
        or not np.array_equal(np.sort(tiebreak), np.arange(config.n_agents))
    ):
        raise ValueError(f"chave de desempate inválida em {path}")
    tiebreak.setflags(write=False)

    expected_hash = common_randomness_sha256(
        randomness,
        tiebreak,
        common_metadata(config, realization_id),
    )
    if str(scalar(data, "common_randomness_sha256")) != expected_hash:
        raise ValueError(f"hash de aleatoriedade comum inválido em {path}")
    return randomness, tiebreak, expected_hash


def _expected_history_steps(config: PilotConfig) -> np.ndarray:
    steps = set(range(0, config.n_steps + 1, config.snapshot_interval))
    steps.update(config.required_snapshot_steps)
    steps.update((0, config.n_steps))
    return np.array(sorted(steps), dtype=int)


def validate_result_file(
    path: Path,
    config: PilotConfig,
    realization_id: int,
    key: ConfigurationKey,
    protocol_sha256: str,
    margins_sha256: str,
    common_hash: str,
) -> dict[str, np.ndarray]:
    data = load_npz(path)
    if set(data) != RESULT_FIELDS:
        raise ValueError(f"esquema de resultado incompatível em {path}")
    expected_scalars = {
        "realization_id": realization_id,
        "configuration_tag": key.tag,
        "regime": key.regime,
        "tax_rate": key.tax_rate,
        "protocol_sha256": protocol_sha256,
        "stability_margins_sha256": margins_sha256,
        "common_randomness_sha256": common_hash,
    }
    for name, expected in expected_scalars.items():
        if scalar(data, name) != expected:
            raise ValueError(f"{name} incompatível em {path}")

    expected_series_steps = np.arange(1, config.n_steps + 1)
    if not np.array_equal(data["series_steps"], expected_series_steps):
        raise ValueError(f"passos da série incompatíveis em {path}")
    for name in ("gini", "max_share", "volume", "revenue"):
        if data[name].shape != (config.n_steps,):
            raise ValueError(f"métrica com shape incompatível em {path}: {name}")

    expected_history_steps = _expected_history_steps(config)
    if not np.array_equal(data["history_steps"], expected_history_steps):
        raise ValueError(f"passos do histórico incompatíveis em {path}")
    if data["history_wealth"].shape != (len(expected_history_steps), config.n_agents):
        raise ValueError(f"histórico com shape incompatível em {path}")
    if data["final_wealth"].shape != (config.n_agents,):
        raise ValueError(f"riqueza final com shape incompatível em {path}")

    numeric_fields = (
        "series_steps", "gini", "max_share", "volume", "revenue", "history_steps",
        "history_wealth", "final_wealth", "final_cash", "final_total_wealth",
        "final_mean_wealth", "final_median_wealth", "final_min_wealth",
        "final_max_wealth", "final_gini", "final_max_share", "max_conservation_error",
        "min_wealth_observed", "max_cash", "duration_seconds", "tax_rate",
    )
    for name in numeric_fields:
        _finite_array(data, name, path)
    if not np.array_equal(data["final_wealth"], data["history_wealth"][-1]):
        raise ValueError(f"riqueza final diverge do snapshot final em {path}")
    initial_total = config.n_agents * config.initial_wealth
    if not np.array_equal(data["history_wealth"][0], np.full(config.n_agents, config.initial_wealth)):
        raise ValueError(f"estado inicial incompatível em {path}")
    if float(data["history_wealth"].min()) < -config.nonnegative_tolerance:
        raise ValueError(f"riqueza negativa no histórico em {path}")
    history_errors = np.abs(data["history_wealth"].sum(axis=1) - initial_total) / initial_total
    if np.any(history_errors > config.conservation_tolerance):
        raise ValueError(f"conservação inválida no histórico em {path}")
    if any(np.any(data[name] < -config.nonnegative_tolerance) for name in ("gini", "max_share", "volume", "revenue")):
        raise ValueError(f"série contém indicador negativo em {path}")

    if not 0.0 <= float(scalar(data, "max_conservation_error")) <= config.conservation_tolerance:
        raise ValueError(f"conservação inválida em {path}")
    if float(scalar(data, "min_wealth_observed")) < -config.nonnegative_tolerance:
        raise ValueError(f"riqueza negativa em {path}")
    if abs(float(scalar(data, "final_cash"))) > config.cash_tolerance:
        raise ValueError(f"caixa final inválido em {path}")
    if float(scalar(data, "max_cash")) < 0.0:
        raise ValueError(f"caixa máximo inválido em {path}")
    if float(scalar(data, "duration_seconds")) < 0.0:
        raise ValueError(f"duração inválida em {path}")

    final_wealth = data["final_wealth"]
    expected_summary = {
        "final_total_wealth": float(final_wealth.sum()),
        "final_mean_wealth": float(final_wealth.mean()),
        "final_median_wealth": float(np.median(final_wealth)),
        "final_min_wealth": float(final_wealth.min()),
        "final_max_wealth": float(final_wealth.max()),
        "final_gini": compute_gini(final_wealth),
        "final_max_share": compute_max_share(final_wealth),
    }
    for name, expected in expected_summary.items():
        if not np.isclose(float(scalar(data, name)), expected, rtol=1e-12, atol=1e-15):
            raise ValueError(f"métrica final incompatível em {path}: {name}")
    if not np.isclose(data["gini"][-1], float(scalar(data, "final_gini")), rtol=1e-12, atol=1e-15):
        raise ValueError(f"série de Gini incompatível com a métrica final em {path}")
    if not np.isclose(data["max_share"][-1], float(scalar(data, "final_max_share")), rtol=1e-12, atol=1e-15):
        raise ValueError(f"série de participação máxima incompatível em {path}")
    return data
