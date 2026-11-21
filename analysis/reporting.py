import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis.statistics import (
    mean_and_standard_error,
    paired_difference,
    series_mean_and_standard_error,
    window_mean,
)
from experiments.configuration import (
    SCHEMA_VERSION,
    ExperimentConfig,
    protocol_document,
)
from experiments.storage import read_json
from experiments.validation import load_and_validate_common, validate_result_file
from simulation.metrics import compute_lorenz
from simulation.mobility import assign_quintiles, boundary_tie_records, transition_matrix
from visualization.experiment_plots import (
    METRICS,
    plot_lorenz,
    plot_mean_series,
    plot_metrics_vs_tau,
    plot_mobility,
)


LORENZ_TAGS = (
    "none",
    "wealth_tau0.05",
    "transaction_tau0.05",
    "wealth_tau0.3",
    "transaction_tau0.3",
)
MOBILITY_TAGS = ("none", "wealth_tau0.05", "transaction_tau0.05")
TABLE_NAMES = (
    "series_mean_se.csv",
    "metric_summary.csv",
    "metric_summary.md",
    "paired_differences.csv",
    "paired_differences.md",
    "diagnostics.csv",
    "lorenz_mean_se.csv",
    "mobility_matrices.csv",
    "boundary_ties.csv",
)
FIGURE_NAMES = (
    "series_mean_se.png",
    "metrics_vs_tau.png",
    "lorenz_selected.png",
    "mobility_none.png",
    "mobility_wealth_tau0.05.png",
    "mobility_transaction_tau0.05.png",
)


@dataclass(frozen=True)
class ReportSummary:
    experiment_dir: Path
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]


def _scalar(data: dict[str, np.ndarray], name: str) -> Any:
    if name not in data or data[name].shape != ():
        raise ValueError(f"campo escalar ausente ou inválido: {name}")
    return data[name].item()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_complete_experiment(
    experiment_dir: Path,
) -> tuple[ExperimentConfig, dict[str, Any], dict[str, Any], dict[int, dict[str, dict[str, np.ndarray]]], dict[int, np.ndarray]]:
    protocol = read_json(experiment_dir / "protocol.json")
    manifest = read_json(experiment_dir / "manifest.json")
    if not isinstance(protocol, dict):
        raise ValueError("documento de protocolo inválido")
    config = ExperimentConfig.from_dict(protocol)
    expected_protocol = protocol_document(config)
    if protocol != expected_protocol:
        raise ValueError("documento de protocolo incompatível com o protocolo canônico")
    protocol_hash = expected_protocol["protocol_sha256"]
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("manifesto e protocolo são incompatíveis")
    expected_tags = [key.tag for key in config.configurations]
    realization_entries = manifest.get("realizations", {})
    expected_realizations = {f"{index:03d}" for index in range(config.n_realizations)}
    if set(realization_entries) != expected_realizations:
        raise ValueError("conjunto de realizações incompleto")

    outputs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    tiebreaks: dict[int, np.ndarray] = {}
    for realization_id in range(config.n_realizations):
        name = f"{realization_id:03d}"
        realization = realization_entries[name]
        configurations = realization.get("configurations", {})
        if set(configurations) != set(expected_tags):
            raise ValueError(f"grade incompleta na realização {name}")
        if any(entry.get("status") != "completed" for entry in configurations.values()):
            raise ValueError(f"realização {name} não está completa")
        common_hash = realization.get("common_randomness_sha256")
        if any(
            entry.get("protocol_sha256") != protocol_hash
            or entry.get("common_randomness_sha256") != common_hash
            for entry in configurations.values()
        ):
            raise ValueError(f"hash de manifesto incompatível na realização {name}")
        raw_dir = experiment_dir / "raw" / f"realization_{name}"
        final_npz_names = {
            path.name for path in raw_dir.glob("*.npz") if not path.name.startswith(".")
        }
        expected_npz_names = {"common_inputs.npz", *(f"{tag}.npz" for tag in expected_tags)}
        if final_npz_names != expected_npz_names:
            raise ValueError(f"artefatos brutos incompatíveis na realização {name}")
        _, tiebreak, validated_common_hash = load_and_validate_common(
            raw_dir / "common_inputs.npz",
            config,
            realization_id,
            protocol_hash,
        )
        if validated_common_hash != common_hash:
            raise ValueError(f"hash comum incompatível na realização {name}")
        tiebreaks[realization_id] = tiebreak
        outputs[realization_id] = {}
        keys_by_tag = {key.tag: key for key in config.configurations}
        for tag in expected_tags:
            expected_file = str(
                (Path("raw") / f"realization_{name}" / f"{tag}.npz").as_posix()
            )
            if configurations[tag].get("file") != expected_file:
                raise ValueError(f"caminho de resultado incompatível em realização {name}, {tag}")
            outputs[realization_id][tag] = validate_result_file(
                raw_dir / f"{tag}.npz",
                config,
                realization_id,
                keys_by_tag[tag],
                protocol_hash,
                common_hash,
            )
    return config, protocol, manifest, outputs, tiebreaks


def generate_report(experiment_dir: Path) -> ReportSummary:
    experiment_dir = Path(experiment_dir)
    config, protocol, manifest, outputs, tiebreaks = _load_complete_experiment(experiment_dir)
    del manifest
    tables_dir = experiment_dir / "tables"
    figures_dir = experiment_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = str(protocol["protocol_sha256"])
    keys_by_tag = {key.tag: key for key in config.configurations}

    series_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_realization: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key in config.configurations:
        datasets = [outputs[r][key.tag] for r in range(config.n_realizations)]
        steps = datasets[0]["series_steps"]
        if any(not np.array_equal(data["series_steps"], steps) for data in datasets[1:]):
            raise ValueError(f"passos incompatíveis em {key.tag}")
        steps_hash = hashlib.sha256(np.ascontiguousarray(steps).tobytes()).hexdigest()
        for metric in METRICS:
            stacked = np.stack([data[metric] for data in datasets])
            mean, standard_error = series_mean_and_standard_error(stacked)
            for step, point_mean, point_se in zip(steps, mean, standard_error):
                series_rows.append({
                    "configuration_tag": key.tag,
                    "regime": key.regime,
                    "tax_rate": key.tax_rate,
                    "metric": metric,
                    "step": int(step),
                    "mean": float(point_mean),
                    "standard_error": float(point_se),
                    "n_realizations": config.n_realizations,
                    "stability_status": config.stability_status,
                    "interpretation": config.interpretation,
                })
            records: list[dict[str, Any]] = []
            for realization_id, data in enumerate(datasets):
                records.append({
                    "realization_id": realization_id,
                    "configuration_tag": key.tag,
                    "regime": key.regime,
                    "value": window_mean(
                        data["series_steps"], data[metric],
                        config.transient_start, config.analysis_end,
                    ),
                    "common_randomness_sha256": str(_scalar(data, "common_randomness_sha256")),
                    "protocol_sha256": protocol_hash,
                    "window_start": config.transient_start,
                    "window_end": config.analysis_end,
                    "steps_sha256": steps_hash,
                    "tax_rate": key.tax_rate,
                })
            values = [float(record["value"]) for record in records]
            metric_mean, metric_se = mean_and_standard_error(values)
            per_realization[(key.tag, metric)] = records
            summary_rows.append({
                "configuration_tag": key.tag,
                "regime": key.regime,
                "tax_rate": key.tax_rate,
                "metric": metric,
                "mean": metric_mean,
                "standard_error": metric_se,
                "mean ± EP": f"{metric_mean:.6g} ± {metric_se:.3g}",
                "window_start": config.transient_start,
                "window_end": config.analysis_end,
                "n_realizations": config.n_realizations,
                "stability_status": config.stability_status,
                "interpretation": config.interpretation,
            })

    paired_rows: list[dict[str, Any]] = []
    for tax_rate in config.tax_rates:
        wealth_tag = keys_by_tag[f"wealth_tau{tax_rate:g}"].tag
        transaction_tag = keys_by_tag[f"transaction_tau{tax_rate:g}"].tag
        for metric in METRICS:
            differences, mean, standard_error = paired_difference(
                per_realization[(wealth_tag, metric)],
                per_realization[(transaction_tag, metric)],
            )
            paired_rows.append({
                "tax_rate": tax_rate,
                "metric": metric,
                "comparison": "wealth - transaction",
                "mean": mean,
                "standard_error": standard_error,
                "mean ± EP": f"{mean:.6g} ± {standard_error:.3g}",
                "n_realizations": len(differences),
                "window_start": config.transient_start,
                "window_end": config.analysis_end,
                "stability_status": config.stability_status,
                "interpretation": config.interpretation,
            })

    diagnostic_rows: list[dict[str, Any]] = []
    for realization_id in range(config.n_realizations):
        for key in config.configurations:
            data = outputs[realization_id][key.tag]
            diagnostic_rows.append({
                "realization_id": realization_id,
                "configuration_tag": key.tag,
                "common_randomness_sha256": str(_scalar(data, "common_randomness_sha256")),
                "protocol_sha256": protocol_hash,
                "max_conservation_error": float(_scalar(data, "max_conservation_error")),
                "min_wealth_observed": float(_scalar(data, "min_wealth_observed")),
                "max_cash": float(_scalar(data, "max_cash")),
                "final_cash": float(_scalar(data, "final_cash")),
                "valid": True,
                "interpretation": config.interpretation,
            })

    lorenz_rows: list[dict[str, Any]] = []
    for tag in LORENZ_TAGS:
        curves: list[np.ndarray] = []
        population_share: np.ndarray | None = None
        for realization_id in range(config.n_realizations):
            data = outputs[realization_id][tag]
            indices = np.flatnonzero(data["history_steps"] == config.analysis_end)
            if len(indices) != 1:
                raise ValueError(f"snapshot analysis_end ausente em {tag}")
            population_share, curve = compute_lorenz(data["history_wealth"][indices[0]])
            curves.append(curve)
        mean, standard_error = series_mean_and_standard_error(np.stack(curves))
        assert population_share is not None
        for share, point_mean, point_se in zip(population_share, mean, standard_error):
            lorenz_rows.append({
                "configuration_tag": tag,
                "step": config.analysis_end,
                "population_share": float(share),
                "mean": float(point_mean),
                "standard_error": float(point_se),
                "n_realizations": config.n_realizations,
                "stability_status": config.stability_status,
                "interpretation": config.interpretation,
            })

    mobility_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    mobility_means: dict[str, np.ndarray] = {}
    mobility_end = config.transient_start + config.mobility_delta
    for tag in MOBILITY_TAGS:
        matrices: list[np.ndarray] = []
        counts_by_realization: list[np.ndarray] = []
        for realization_id in range(config.n_realizations):
            data = outputs[realization_id][tag]
            history_steps = data["history_steps"]
            start_indices = np.flatnonzero(history_steps == config.transient_start)
            end_indices = np.flatnonzero(history_steps == mobility_end)
            if len(start_indices) != 1 or len(end_indices) != 1:
                raise ValueError(f"snapshots de mobilidade ausentes em {tag}")
            start_wealth = data["history_wealth"][start_indices[0]]
            end_wealth = data["history_wealth"][end_indices[0]]
            key = tiebreaks[realization_id]
            start_quintiles = assign_quintiles(start_wealth, key)
            end_quintiles = assign_quintiles(end_wealth, key)
            counts, probabilities = transition_matrix(
                start_quintiles, end_quintiles, origin_step=config.transient_start
            )
            counts_by_realization.append(counts)
            matrices.append(probabilities)
            for step, wealth in (
                (config.transient_start, start_wealth),
                (mobility_end, end_wealth),
            ):
                for tie in boundary_tie_records(wealth, key):
                    boundary_rows.append({
                        "realization_id": realization_id,
                        "configuration_tag": tag,
                        "step": step,
                        "boundary": tie["boundary"],
                        "tied": tie["tied"],
                        "tied_value": "" if tie["tied_value"] is None else tie["tied_value"],
                        "multiplicity": tie["multiplicity"],
                        "common_randomness_sha256": str(_scalar(data, "common_randomness_sha256")),
                        "interpretation": config.interpretation,
                    })
        stacked = np.stack(matrices)
        mean, standard_error = np.mean(stacked, axis=0), np.std(stacked, axis=0, ddof=1) / np.sqrt(config.n_realizations)
        mobility_means[tag] = mean
        for origin in range(5):
            for destination in range(5):
                mobility_rows.append({
                    "configuration_tag": tag,
                    "origin_quintile": origin + 1,
                    "destination_quintile": destination + 1,
                    "mean": float(mean[origin, destination]),
                    "standard_error": float(standard_error[origin, destination]),
                    "n_realizations": config.n_realizations,
                    "origin_step": config.transient_start,
                    "destination_step": mobility_end,
                    "mean_count": float(np.mean([counts[origin, destination] for counts in counts_by_realization])),
                    "stability_status": config.stability_status,
                    "interpretation": config.interpretation,
                })

    _write_csv(tables_dir / "series_mean_se.csv", series_rows, list(series_rows[0]))
    _write_csv(tables_dir / "metric_summary.csv", summary_rows, list(summary_rows[0]))
    _write_markdown(
        tables_dir / "metric_summary.md", summary_rows,
        ["configuration_tag", "metric", "mean ± EP", "window_start", "window_end", "n_realizations", "interpretation"],
    )
    _write_csv(tables_dir / "paired_differences.csv", paired_rows, list(paired_rows[0]))
    _write_markdown(
        tables_dir / "paired_differences.md", paired_rows,
        ["tax_rate", "metric", "comparison", "mean ± EP", "n_realizations", "interpretation"],
    )
    _write_csv(tables_dir / "diagnostics.csv", diagnostic_rows, list(diagnostic_rows[0]))
    _write_csv(tables_dir / "lorenz_mean_se.csv", lorenz_rows, list(lorenz_rows[0]))
    _write_csv(tables_dir / "mobility_matrices.csv", mobility_rows, list(mobility_rows[0]))
    _write_csv(tables_dir / "boundary_ties.csv", boundary_rows, list(boundary_rows[0]))

    title_suffix = (
        f"R={config.n_realizations}, janela [{config.transient_start}, {config.analysis_end}], "
        "tempo finito (not_assessed)"
    )
    plot_mean_series(series_rows, figures_dir / "series_mean_se.png", f"Séries médias ± EP — {title_suffix}")
    plot_metrics_vs_tau(summary_rows, figures_dir / "metrics_vs_tau.png", f"Métricas por alíquota — {title_suffix}")
    plot_lorenz(
        lorenz_rows,
        figures_dir / "lorenz_selected.png",
        f"Curvas de Lorenz em t={config.analysis_end} — tempo finito (not_assessed)",
    )
    for tag in MOBILITY_TAGS:
        plot_mobility(
            mobility_means[tag],
            figures_dir / f"mobility_{tag}.png",
            f"Mobilidade {tag}: t={config.transient_start}→{mobility_end} — tempo finito",
        )

    return ReportSummary(
        experiment_dir=experiment_dir,
        tables=tuple(tables_dir / name for name in TABLE_NAMES),
        figures=tuple(figures_dir / name for name in FIGURE_NAMES),
    )
