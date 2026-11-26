import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import time
from typing import Any

import numpy as np

from analysis.statistics import series_mean_and_standard_error
from analysis.stability import (
    STABILITY_METRICS,
    StabilityComparison,
    final_tail_windows,
    joint_status,
    metric_status,
    paired_equivalence,
    realization_window_means,
    summarize_realization_means,
)
from experiments.pilot_configuration import (
    PilotConfig,
    StabilityMargins,
    pilot_config_from_protocol_document,
    protocol_document,
    stability_margins_from_document,
    validate_pilot_directory,
)
from experiments.pilot_validation import load_and_validate_common, validate_result_file
from experiments.storage import atomic_write_json, atomic_write_text, read_json
from visualization.pilot_plots import plot_final_windows_comparison, plot_series_to_2t


TABLE_NAMES = (
    "window_realization_means.csv",
    "window_summary.csv",
    "stability_comparisons.csv",
    "stability_status.csv",
)
FIGURE_NAMES = ("series_to_2t.png", "final_windows_comparison.png")


@dataclass(frozen=True)
class PilotReportSummary:
    pilot_dir: Path
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    report: Path
    timing: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    document = StringIO(newline="")
    writer = csv.DictWriter(document, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, document.getvalue())


def _margins_from_document(document: dict[str, Any]) -> StabilityMargins:
    return stability_margins_from_document(document)


def _config_from_protocol(document: dict[str, Any], margins: StabilityMargins) -> PilotConfig:
    return pilot_config_from_protocol_document(document, margins)


def load_complete_pilot(
    pilot_dir: Path,
) -> tuple[PilotConfig, StabilityMargins, dict[int, dict[str, dict[str, np.ndarray]]]]:
    pilot_dir = Path(pilot_dir)
    margins_document = read_json(pilot_dir / "stability_margins.json")
    protocol = read_json(pilot_dir / "protocol.json")
    manifest = read_json(pilot_dir / "manifest.json")
    if not isinstance(margins_document, dict) or not isinstance(protocol, dict) or not isinstance(manifest, dict):
        raise ValueError("artefato JSON de piloto inválido")
    margins = _margins_from_document(margins_document)
    config = _config_from_protocol(protocol, margins)
    protocol_hash = str(protocol["protocol_sha256"])
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol_sha256") != protocol_hash
        or manifest.get("stability_margins_sha256") != margins.sha256
    ):
        raise ValueError("manifesto incompatível com o piloto")
    expected_realizations = {f"{index:03d}" for index in range(config.n_realizations)}
    realizations = manifest.get("realizations")
    if not isinstance(realizations, dict) or set(realizations) != expected_realizations:
        raise ValueError("conjunto de realizações incompleto")

    outputs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    keys_by_tag = {key.tag: key for key in config.configurations}
    expected_tags = set(keys_by_tag)
    for realization_id in range(config.n_realizations):
        name = f"{realization_id:03d}"
        entry = realizations[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("configurations"), dict):
            raise ValueError(f"realização inválida: {name}")
        if set(entry["configurations"]) != expected_tags:
            raise ValueError(f"grade incompleta na realização {name}")
        raw_dir = pilot_dir / "raw" / f"realization_{name}"
        expected_files = {"common_inputs.npz", *(f"{tag}.npz" for tag in expected_tags)}
        if {path.name for path in raw_dir.glob("*.npz") if not path.name.startswith(".")} != expected_files:
            raise ValueError(f"artefatos brutos incompatíveis na realização {name}")
        _, _, common_hash = load_and_validate_common(raw_dir / "common_inputs.npz", config, realization_id, protocol_hash)
        if entry.get("common_randomness_sha256") != common_hash:
            raise ValueError(f"hash comum incompatível na realização {name}")
        outputs[realization_id] = {}
        for tag, key in keys_by_tag.items():
            status = entry["configurations"][tag]
            expected_file = (Path("raw") / f"realization_{name}" / f"{tag}.npz").as_posix()
            if not isinstance(status, dict) or status.get("status") != "completed" or status.get("file") != expected_file:
                raise ValueError(f"resultado incompleto em {name}/{tag}")
            if (
                status.get("protocol_sha256") != protocol_hash
                or status.get("stability_margins_sha256") != margins.sha256
                or status.get("common_randomness_sha256") != common_hash
            ):
                raise ValueError(f"metadados incompatíveis em {name}/{tag}")
            outputs[realization_id][tag] = validate_result_file(
                raw_dir / f"{tag}.npz", config, realization_id, key,
                protocol_hash, margins.sha256, common_hash,
            )
    return config, margins, outputs


def _comparison_row(comparison: StabilityComparison, tag: str) -> dict[str, Any]:
    return {
        "configuration_tag": tag, "metric": comparison.metric,
        "comparison": comparison.comparison,
        "left_horizon": comparison.left_horizon, "left_window": comparison.left_window,
        "right_horizon": comparison.right_horizon, "right_window": comparison.right_window,
        "mean_difference": comparison.mean_difference, "standard_error": comparison.standard_error,
        "t_critical": comparison.t_critical, "ci_lower": comparison.ci_lower,
        "ci_upper": comparison.ci_upper, "effective_margin": comparison.effective_margin,
        "decision": comparison.decision, "n_realizations": comparison.n_realizations,
    }


def _timing_document(
    pilot_dir: Path,
    outputs: dict[int, dict[str, dict[str, np.ndarray]]],
    analysis_started: str,
    analysis_seconds: float,
    simulation_summary: dict[str, Any] | None,
    total_seconds: float | None,
) -> dict[str, Any]:
    prior_path = pilot_dir / "timing.json"
    prior = read_json(prior_path) if prior_path.exists() else {}
    durations = [float(data["duration_seconds"].item()) for by_tag in outputs.values() for data in by_tag.values()]
    document: dict[str, Any] = {
        "analysis_started_at_utc": analysis_started,
        "analysis_finished_at_utc": _utc_now(),
        "analysis_seconds": analysis_seconds,
        "persisted_2t_compute_seconds": float(sum(durations)),
        "persisted_scenario_count": len(durations),
        "t_duration_seconds": None,
        "timing_note": "T é prefixo da trajetória 2T; não há duração separada inventada para T.",
    }
    if isinstance(prior, dict):
        document = {**prior, **document}
    if simulation_summary is not None:
        document["simulation"] = simulation_summary
    if total_seconds is not None:
        document["all_command_seconds"] = total_seconds
    return document


def generate_pilot_report(
    pilot_dir: Path,
    simulation_summary: dict[str, Any] | None = None,
    total_seconds: float | None = None,
) -> PilotReportSummary:
    started_at = _utc_now()
    started = time.perf_counter()
    pilot_dir = validate_pilot_directory(pilot_dir)
    config, margins, outputs = load_complete_pilot(pilot_dir)
    tables_dir = pilot_dir / "tables"
    figures_dir = pilot_dir / "figures"
    realization_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    series_rows: list[dict[str, Any]] = []
    by_window: dict[tuple[str, str, int, int], np.ndarray] = {}

    for key in config.configurations:
        datasets = [outputs[realization_id][key.tag] for realization_id in range(config.n_realizations)]
        steps = datasets[0]["series_steps"]
        if any(not np.array_equal(data["series_steps"], steps) for data in datasets[1:]):
            raise ValueError(f"passos incompatíveis em {key.tag}")
        for metric in STABILITY_METRICS:
            stacked = np.stack([data[metric] for data in datasets])
            mean, standard_error = series_mean_and_standard_error(stacked)
            series_rows.extend({
                "configuration_tag": key.tag, "metric": metric, "step": int(step),
                "mean": float(point_mean), "standard_error": float(point_se),
            } for step, point_mean, point_se in zip(steps, mean, standard_error))
            for horizon in (config.t_horizon, config.n_steps):
                for window in final_tail_windows(horizon, margins):
                    values = realization_window_means(steps, stacked, window)
                    by_window[(key.tag, metric, horizon, window.index)] = values
                    value_mean, value_se = summarize_realization_means(values)
                    summary_rows.append({
                        "configuration_tag": key.tag, "regime": key.regime, "tax_rate": key.tax_rate,
                        "horizon": horizon, "window_index": window.index,
                        "window_start": window.start, "window_end": window.end, "metric": metric,
                        "mean": value_mean, "standard_error": value_se, "n_realizations": len(values),
                    })
                    realization_rows.extend({
                        "realization_id": realization_id, "configuration_tag": key.tag,
                        "regime": key.regime, "tax_rate": key.tax_rate, "horizon": horizon,
                        "window_index": window.index, "window_start": window.start,
                        "window_end": window.end, "metric": metric, "temporal_mean": float(value),
                    } for realization_id, value in enumerate(values))

            comparisons: list[StabilityComparison] = []
            for horizon in (config.t_horizon, config.n_steps):
                for index in range(margins.K - 1):
                    comparisons.append(paired_equivalence(
                        metric, "adjacent_windows", by_window[(key.tag, metric, horizon, index)],
                        by_window[(key.tag, metric, horizon, index + 1)], margins,
                        horizon, index, horizon, index + 1,
                    ))
            comparisons.append(paired_equivalence(
                metric, "T_vs_2T_final", by_window[(key.tag, metric, config.t_horizon, margins.K - 1)],
                by_window[(key.tag, metric, config.n_steps, margins.K - 1)], margins,
                config.t_horizon, margins.K - 1, config.n_steps, margins.K - 1,
            ))
            comparison_rows.extend(_comparison_row(comparison, key.tag) for comparison in comparisons)
            status_rows.append({
                "configuration_tag": key.tag, "metric": metric,
                "metric_status": metric_status(comparisons), "joint_status": "",
            })
        statuses = [row["metric_status"] for row in status_rows if row["configuration_tag"] == key.tag]
        scenario_status = joint_status(statuses)
        for row in status_rows:
            if row["configuration_tag"] == key.tag:
                row["joint_status"] = scenario_status

    _write_csv(tables_dir / TABLE_NAMES[0], realization_rows, list(realization_rows[0]))
    _write_csv(tables_dir / TABLE_NAMES[1], summary_rows, list(summary_rows[0]))
    _write_csv(tables_dir / TABLE_NAMES[2], comparison_rows, list(comparison_rows[0]))
    _write_csv(tables_dir / TABLE_NAMES[3], status_rows, list(status_rows[0]))
    window_limits = tuple(window.end for window in final_tail_windows(config.n_steps, margins))
    plot_series_to_2t(series_rows, figures_dir / FIGURE_NAMES[0], config.t_horizon, window_limits)
    plot_final_windows_comparison(summary_rows, figures_dir / FIGURE_NAMES[1])

    report_path = pilot_dir / "report.md"
    status_lines = "\n".join(
        f"| {row['configuration_tag']} | {row['metric']} | {row['metric_status']} | {row['joint_status']} |"
        for row in status_rows
    )
    atomic_write_text(
        report_path,
        "# Relatório de estabilidade do piloto\n\n"
        f"Protocolo provisório: R={config.n_realizations}, N={config.n_agents}, T={config.t_horizon}, 2T={config.n_steps}. "
        f"Hash do protocolo: `{protocol_document(config, margins)['protocol_sha256']}`. "
        f"Hash das margens: `{margins.sha256}`.\n\n"
        f"A cauda final de cada horizonte usa `floor(floor(H × {margins.tail_fraction}) / {margins.K})` passos por janela, "
        "em janelas inclusivas, sucessivas e não sobrepostas. Cada média temporal é calculada por realização antes da média e EP entre realizações.\n\n"
        "Cada comparação usa IC bilateral Student-t. Equivalência exige que todo o IC fique dentro da margem; para volume, a margem efetiva é o máximo da margem absoluta e da margem relativa multiplicada pela maior média absoluta comparada. "
        "Uma métrica é stabilized se todas as comparações passam, finite_time se alguma falha claramente, inconclusive se resta uma comparação ambígua e not_assessed se não houver avaliação válida. O status conjunto exige as três métricas stabilized.\n\n"
        "| Configuração | Métrica | Status da métrica | Status conjunto |\n| --- | --- | --- | --- |\n"
        f"{status_lines}\n\n"
        "T é somente o prefixo da trajetória 2T medida. Não existe duração independente para T. Este diagnóstico não prova equilíbrio matemático e não torna o protocolo principal definitivo.\n",
    )
    analysis_seconds = time.perf_counter() - started
    timing_path = pilot_dir / "timing.json"
    atomic_write_json(timing_path, _timing_document(
        pilot_dir, outputs, started_at, analysis_seconds, simulation_summary, total_seconds,
    ))
    return PilotReportSummary(
        pilot_dir=pilot_dir, tables=tuple(tables_dir / name for name in TABLE_NAMES),
        figures=tuple(figures_dir / name for name in FIGURE_NAMES), report=report_path, timing=timing_path,
    )
