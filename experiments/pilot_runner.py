from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pilot_configuration import (
    PILOT_SCHEMA_VERSION,
    PilotConfig,
    StabilityMargins,
    protocol_document,
    stability_margins_document,
    stability_margins_from_document,
    validate_pilot_protocol_document,
)
from experiments.pilot_validation import (
    common_metadata,
    load_and_validate_common,
    validate_result_file,
)
from experiments.randomization import (
    SimulationRandomness,
    common_randomness_sha256,
    generate_realization_randomness,
)
from experiments.storage import atomic_write_json, atomic_write_npz, read_json
from simulation.engine import SimulationResult, run_simulation


@dataclass(frozen=True)
class PilotRunSummary:
    completed: int
    skipped: int
    total: int
    pilot_dir: Path
    simulation_seconds: float


def _initial_manifest(config: PilotConfig, protocol_sha256: str, margins_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "stability_margins_sha256": margins_sha256,
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


def _prepare_pilot(
    config: PilotConfig,
    margins: StabilityMargins,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    pilot_dir = config.pilot_dir
    protocol_path = pilot_dir / "protocol.json"
    margins_path = pilot_dir / "stability_margins.json"
    manifest_path = pilot_dir / "manifest.json"
    expected_protocol = protocol_document(config, margins)
    expected_margins = stability_margins_document(margins)

    if protocol_path.exists():
        try:
            validate_pilot_protocol_document(read_json(protocol_path), config, margins)
        except ValueError as error:
            raise ValueError(f"protocolo incompatível em {pilot_dir}; use outro pilot_id") from error
    else:
        pilot_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(protocol_path, expected_protocol)

    if margins_path.exists():
        try:
            persisted_margins = stability_margins_from_document(read_json(margins_path))
        except ValueError as error:
            raise ValueError(f"margens de estabilidade incompatíveis em {pilot_dir}") from error
        if persisted_margins != margins:
            raise ValueError(f"margens de estabilidade incompatíveis em {pilot_dir}")
    else:
        atomic_write_json(margins_path, expected_margins)

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema_version") != PILOT_SCHEMA_VERSION
            or manifest.get("protocol_sha256") != expected_protocol["protocol_sha256"]
            or manifest.get("stability_margins_sha256") != expected_margins["margins_sha256"]
        ):
            raise ValueError("manifesto incompatível com o protocolo do piloto")
    else:
        manifest = _initial_manifest(
            config,
            expected_protocol["protocol_sha256"],
            expected_margins["margins_sha256"],
        )
        atomic_write_json(manifest_path, manifest)
    return pilot_dir, expected_protocol, manifest


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


def _validate_simulation_result(result: SimulationResult, config: PilotConfig) -> None:
    arrays = (
        result.state.wealth, result.state.aversion, result.history.wealth,
        result.series.gini, result.series.max_share, result.series.volume,
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
    if not set(config.required_snapshot_steps).issubset(set(int(step) for step in result.history.steps)):
        raise ValueError("snapshots obrigatórios ausentes")


def _write_result(
    path: Path,
    result: SimulationResult,
    realization_id: int,
    tag: str,
    regime: str,
    tax_rate: float,
    protocol_sha256: str,
    margins_sha256: str,
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
        stability_margins_sha256=np.array(margins_sha256),
        common_randomness_sha256=np.array(common_hash),
    )


def run_pilot(config: PilotConfig, margins: StabilityMargins) -> PilotRunSummary:
    pilot_dir, protocol, manifest = _prepare_pilot(config, margins)
    manifest_path = pilot_dir / "manifest.json"
    protocol_sha256 = str(protocol["protocol_sha256"])
    margins_sha256 = margins.sha256
    realization_seeds = np.random.SeedSequence(config.base_seed).spawn(config.n_realizations)
    completed = 0
    skipped = 0
    simulation_seconds = 0.0

    for realization_id, realization_seed in enumerate(realization_seeds):
        realization_name = f"{realization_id:03d}"
        realization_dir = pilot_dir / "raw" / f"realization_{realization_name}"
        common_path = realization_dir / "common_inputs.npz"
        if common_path.exists():
            randomness, _, common_hash = load_and_validate_common(
                common_path, config, realization_id, protocol_sha256
            )
        else:
            randomness, tiebreak = generate_realization_randomness(
                realization_seed, config.n_agents, config.n_steps
            )
            common_hash = common_randomness_sha256(
                randomness, tiebreak, common_metadata(config, realization_id)
            )
            _write_common(
                common_path, randomness, tiebreak, realization_id, protocol_sha256, common_hash
            )
            randomness, _, common_hash = load_and_validate_common(
                common_path, config, realization_id, protocol_sha256
            )

        realization_manifest = manifest.get("realizations", {}).get(realization_name)
        if not isinstance(realization_manifest, dict):
            raise ValueError(f"realização ausente no manifesto: {realization_name}")
        recorded_hash = realization_manifest.get("common_randomness_sha256")
        if recorded_hash not in (None, common_hash):
            raise ValueError("hash comum incompatível com o manifesto")
        realization_manifest["common_randomness_sha256"] = common_hash
        atomic_write_json(manifest_path, manifest)

        configurations = realization_manifest.get("configurations")
        expected_tags = {key.tag for key in config.configurations}
        if not isinstance(configurations, dict) or set(configurations) != expected_tags:
            raise ValueError(f"configurações incompatíveis no manifesto: {realization_name}")
        for key in config.configurations:
            entry = configurations[key.tag]
            if not isinstance(entry, dict):
                raise ValueError(f"entrada de manifesto inválida: {key.tag}")
            output_path = realization_dir / f"{key.tag}.npz"
            expected_relative = (Path("raw") / f"realization_{realization_name}" / f"{key.tag}.npz").as_posix()
            if entry.get("status") == "completed" and output_path.exists():
                if (
                    entry.get("file") != expected_relative
                    or entry.get("protocol_sha256") != protocol_sha256
                    or entry.get("stability_margins_sha256") != margins_sha256
                    or entry.get("common_randomness_sha256") != common_hash
                ):
                    raise ValueError(f"estado completed incompatível no manifesto: {key.tag}")
                validate_result_file(
                    output_path, config, realization_id, key, protocol_sha256,
                    margins_sha256, common_hash,
                )
                skipped += 1
                continue
            if entry.get("status") == "running" and output_path.exists():
                try:
                    validate_result_file(
                        output_path, config, realization_id, key, protocol_sha256,
                        margins_sha256, common_hash,
                    )
                except Exception as error:
                    entry["status"] = "failed"
                    entry["error"] = str(error)
                    atomic_write_json(manifest_path, manifest)
                    raise
                entry.clear()
                entry.update({
                    "status": "completed", "file": expected_relative,
                    "protocol_sha256": protocol_sha256,
                    "stability_margins_sha256": margins_sha256,
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
                    config.simulation_config(key), randomness=randomness,
                    required_snapshot_steps=config.required_snapshot_steps,
                )
                _validate_simulation_result(result, config)
                _write_result(
                    output_path, result, realization_id, key.tag, key.regime, key.tax_rate,
                    protocol_sha256, margins_sha256, common_hash,
                )
                validate_result_file(
                    output_path, config, realization_id, key, protocol_sha256,
                    margins_sha256, common_hash,
                )
            except Exception as error:
                entry["status"] = "failed"
                entry["error"] = str(error)
                atomic_write_json(manifest_path, manifest)
                raise
            entry.clear()
            entry.update({
                "status": "completed", "file": expected_relative,
                "protocol_sha256": protocol_sha256,
                "stability_margins_sha256": margins_sha256,
                "common_randomness_sha256": common_hash,
            })
            atomic_write_json(manifest_path, manifest)
            completed += 1
            simulation_seconds += result.duration_seconds

    return PilotRunSummary(
        completed=completed,
        skipped=skipped,
        total=config.n_realizations * len(config.configurations),
        pilot_dir=pilot_dir,
        simulation_seconds=simulation_seconds,
    )
