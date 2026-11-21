from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.configuration import (
    SCHEMA_VERSION,
    ExperimentConfig,
    protocol_document,
)
from experiments.randomization import (
    SimulationRandomness,
    common_randomness_sha256,
    generate_realization_randomness,
)
from experiments.storage import atomic_write_json, atomic_write_npz, read_json
from experiments.validation import (
    common_metadata,
    load_and_validate_common,
    validate_result_file,
)
from simulation.engine import SimulationResult, run_simulation


@dataclass(frozen=True)
class RunSummary:
    completed: int
    skipped: int
    total: int
    experiment_dir: Path


def _initial_manifest(config: ExperimentConfig, protocol_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "realizations": {
            f"{realization_id:03d}": {
                "common_randomness_sha256": None,
                "configurations": {
                    key.tag: {"status": "pending"} for key in config.configurations
                },
            }
            for realization_id in range(config.n_realizations)
        },
    }


def _prepare_experiment(config: ExperimentConfig) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    experiment_dir = config.experiment_dir
    protocol_path = experiment_dir / "protocol.json"
    manifest_path = experiment_dir / "manifest.json"
    expected_protocol = protocol_document(config)

    if protocol_path.exists():
        existing_protocol = read_json(protocol_path)
        if existing_protocol != expected_protocol:
            raise ValueError(
                f"protocolo incompatível em {experiment_dir}; use outro experiment_id"
            )
    else:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(protocol_path, expected_protocol)

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("protocol_sha256") != expected_protocol["protocol_sha256"]
        ):
            raise ValueError("manifesto incompatível com o protocolo")
    else:
        manifest = _initial_manifest(config, expected_protocol["protocol_sha256"])
        atomic_write_json(manifest_path, manifest)
    return experiment_dir, expected_protocol, manifest


def _write_common(
    path: Path,
    randomness: SimulationRandomness,
    tiebreak: np.ndarray,
    realization_id: int,
    protocol_sha256: str,
    common_hash: str,
) -> None:
    atomic_write_npz(
        path,
        aversion=randomness.aversion,
        pairs=randomness.pairs,
        first_agent_wins=randomness.first_agent_wins,
        mobility_tiebreak=tiebreak,
        realization_id=np.array(realization_id, dtype=np.int64),
        protocol_sha256=np.array(protocol_sha256),
        common_randomness_sha256=np.array(common_hash),
    )


def _validate_result(result: SimulationResult, config: ExperimentConfig) -> None:
    arrays = (
        result.state.wealth,
        result.state.aversion,
        result.history.wealth,
        result.series.gini,
        result.series.max_share,
        result.series.volume,
        result.series.revenue,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("a execução produziu valor não finito")
    if result.metrics.max_conservation_error > config.conservation_tolerance:
        raise ValueError("erro de conservação acima da tolerância")
    if result.metrics.min_wealth_observed < -config.nonnegative_tolerance:
        raise ValueError("riqueza negativa abaixo da tolerância")
    if float(result.state.wealth.min()) < -config.nonnegative_tolerance:
        raise ValueError("riqueza final negativa abaixo da tolerância")
    if abs(result.state.cash) > config.cash_tolerance:
        raise ValueError("caixa final acima da tolerância")
    required = set(config.required_snapshot_steps)
    if not required.issubset(set(int(step) for step in result.history.steps)):
        raise ValueError("snapshots obrigatórios ausentes")


def _write_result(
    path: Path,
    result: SimulationResult,
    realization_id: int,
    tag: str,
    regime: str,
    tax_rate: float,
    protocol_sha256: str,
    common_hash: str,
) -> None:
    atomic_write_npz(
        path,
        series_steps=result.series.steps,
        gini=result.series.gini,
        max_share=result.series.max_share,
        volume=result.series.volume,
        revenue=result.series.revenue,
        history_steps=result.history.steps,
        history_wealth=result.history.wealth,
        final_wealth=result.state.wealth,
        final_cash=np.array(result.state.cash),
        final_total_wealth=np.array(result.metrics.total_wealth),
        final_mean_wealth=np.array(result.metrics.mean_wealth),
        final_median_wealth=np.array(result.metrics.median_wealth),
        final_min_wealth=np.array(result.metrics.min_wealth),
        final_max_wealth=np.array(result.metrics.max_wealth),
        final_gini=np.array(result.metrics.gini),
        final_max_share=np.array(result.metrics.max_share),
        max_conservation_error=np.array(result.metrics.max_conservation_error),
        min_wealth_observed=np.array(result.metrics.min_wealth_observed),
        max_cash=np.array(result.metrics.max_cash),
        duration_seconds=np.array(result.duration_seconds),
        realization_id=np.array(realization_id, dtype=np.int64),
        configuration_tag=np.array(tag),
        regime=np.array(regime),
        tax_rate=np.array(tax_rate),
        protocol_sha256=np.array(protocol_sha256),
        common_randomness_sha256=np.array(common_hash),
    )


def run_experiment(config: ExperimentConfig) -> RunSummary:
    experiment_dir, protocol, manifest = _prepare_experiment(config)
    manifest_path = experiment_dir / "manifest.json"
    realization_seeds = np.random.SeedSequence(config.base_seed).spawn(config.n_realizations)
    completed = 0
    skipped = 0

    for realization_id, realization_seed in enumerate(realization_seeds):
        realization_name = f"{realization_id:03d}"
        realization_dir = experiment_dir / "raw" / f"realization_{realization_name}"
        common_path = realization_dir / "common_inputs.npz"
        if common_path.exists():
            randomness, _, common_hash = load_and_validate_common(
                common_path, config, realization_id, protocol["protocol_sha256"]
            )
        else:
            randomness, tiebreak = generate_realization_randomness(
                realization_seed, config.n_agents, config.n_steps
            )
            common_hash = common_randomness_sha256(
                randomness, tiebreak, common_metadata(config, realization_id)
            )
            _write_common(
                common_path,
                randomness,
                tiebreak,
                realization_id,
                protocol["protocol_sha256"],
                common_hash,
            )
            randomness, _, common_hash = load_and_validate_common(
                common_path, config, realization_id, protocol["protocol_sha256"]
            )

        realization_manifest = manifest["realizations"][realization_name]
        recorded_hash = realization_manifest.get("common_randomness_sha256")
        if recorded_hash not in (None, common_hash):
            raise ValueError("hash comum incompatível com o manifesto")
        realization_manifest["common_randomness_sha256"] = common_hash
        atomic_write_json(manifest_path, manifest)

        for key in config.configurations:
            entry = realization_manifest["configurations"][key.tag]
            output_path = realization_dir / f"{key.tag}.npz"
            if entry.get("status") == "completed" and output_path.exists():
                if (
                    entry.get("protocol_sha256") != protocol["protocol_sha256"]
                    or entry.get("common_randomness_sha256") != common_hash
                ):
                    raise ValueError(f"estado completed incompatível no manifesto: {key.tag}")
                validate_result_file(
                    output_path,
                    config,
                    realization_id,
                    key,
                    protocol["protocol_sha256"],
                    common_hash,
                )
                skipped += 1
                continue
            if entry.get("status") == "running" and output_path.exists():
                try:
                    validate_result_file(
                        output_path,
                        config,
                        realization_id,
                        key,
                        protocol["protocol_sha256"],
                        common_hash,
                    )
                except Exception as error:
                    entry["status"] = "failed"
                    entry["error"] = str(error)
                    atomic_write_json(manifest_path, manifest)
                    raise
                entry.clear()
                entry.update({
                    "status": "completed",
                    "file": str(output_path.relative_to(experiment_dir)),
                    "protocol_sha256": protocol["protocol_sha256"],
                    "common_randomness_sha256": common_hash,
                })
                atomic_write_json(manifest_path, manifest)
                skipped += 1
                continue
            if entry.get("status") == "completed" and not output_path.exists():
                entry["status"] = "pending"
            elif output_path.exists():
                raise ValueError(f"artefato final sem estado completed: {output_path}")

            entry["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            try:
                result = run_simulation(
                    config.simulation_config(key),
                    randomness=randomness,
                    required_snapshot_steps=config.required_snapshot_steps,
                )
                _validate_result(result, config)
                _write_result(
                    output_path,
                    result,
                    realization_id,
                    key.tag,
                    key.regime,
                    key.tax_rate,
                    protocol["protocol_sha256"],
                    common_hash,
                )
                validate_result_file(
                    output_path,
                    config,
                    realization_id,
                    key,
                    protocol["protocol_sha256"],
                    common_hash,
                )
            except Exception as error:
                entry["status"] = "failed"
                entry["error"] = str(error)
                atomic_write_json(manifest_path, manifest)
                raise
            entry.clear()
            entry.update({
                "status": "completed",
                "file": str(output_path.relative_to(experiment_dir)),
                "protocol_sha256": protocol["protocol_sha256"],
                "common_randomness_sha256": common_hash,
            })
            atomic_write_json(manifest_path, manifest)
            completed += 1

    return RunSummary(
        completed=completed,
        skipped=skipped,
        total=config.n_realizations * len(config.configurations),
        experiment_dir=experiment_dir,
    )
