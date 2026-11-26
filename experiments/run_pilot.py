import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from experiments.pilot_configuration import (
    PilotConfig,
    load_stability_margins,
    validate_pilot_directory,
    validate_pilot_id,
    validate_pilot_results_root,
)
from experiments.storage import atomic_write_json, read_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_location_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pilot-id", default="main", help="identificador seguro do piloto")
    parser.add_argument("--results-root", default="Results/pilots", help="diretório que contém os pilotos")


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    _add_location_arguments(parser)
    parser.add_argument(
        "--stability-config", default="config/stability_margins.example.json",
        help="arquivo JSON com as margens de estabilidade",
    )
    parser.add_argument("--n-realizations", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=10000)
    parser.add_argument("--t-horizon", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=2025)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa e analisa o piloto 2T de estabilidade Yard-Sale.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate", help="simula ou retoma os resultados brutos 2T")
    _add_simulation_arguments(simulate)
    analyze = subparsers.add_parser("analyze", help="recria tabelas e figuras apenas dos dados brutos")
    _add_location_arguments(analyze)
    all_parser = subparsers.add_parser("all", help="simula/retoma e depois analisa")
    _add_simulation_arguments(all_parser)
    return parser


def _config_from_args(args: argparse.Namespace) -> PilotConfig:
    values: dict[str, Any] = {
        "pilot_id": args.pilot_id, "results_root": args.results_root,
        "n_realizations": args.n_realizations, "n_agents": args.n_agents,
        "t_horizon": args.t_horizon, "base_seed": args.base_seed,
    }
    return PilotConfig(**values)


def _record_simulation_timing(pilot_dir: Path, summary: Any, started_at: str, seconds: float) -> dict[str, Any]:
    path = pilot_dir / "timing.json"
    prior = read_json(path) if path.exists() else {}
    document = dict(prior) if isinstance(prior, dict) else {}
    document["simulation"] = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "wall_seconds": seconds,
        "reported_2t_compute_seconds": summary.simulation_seconds,
        "completed": summary.completed,
        "skipped": summary.skipped,
        "total": summary.total,
    }
    document["timing_note"] = "T é prefixo da trajetória 2T; não há duração separada inventada para T."
    atomic_write_json(path, document)
    return document["simulation"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pilot_id = validate_pilot_id(args.pilot_id)
    results_root = validate_pilot_results_root(args.results_root)
    pilot_dir = validate_pilot_directory(Path(results_root) / pilot_id)
    all_started = time.perf_counter()
    simulation_summary: dict[str, Any] | None = None

    if args.command in ("simulate", "all"):
        margins = load_stability_margins(args.stability_config)
        from experiments.pilot_runner import run_pilot

        started_at = _utc_now()
        phase_started = time.perf_counter()
        summary = run_pilot(_config_from_args(args), margins)
        simulation_summary = _record_simulation_timing(
            summary.pilot_dir, summary, started_at, time.perf_counter() - phase_started,
        )
        print(
            f"simulação 2T: total={summary.total}, completed={summary.completed}, "
            f"skipped={summary.skipped}, simulation_seconds={summary.simulation_seconds:.6f}, "
            f"diretório={summary.pilot_dir}"
        )

    if args.command in ("analyze", "all"):
        from analysis.pilot_reporting import generate_pilot_report

        report = generate_pilot_report(pilot_dir, simulation_summary=simulation_summary)
        if args.command == "all":
            timing = read_json(report.timing)
            if not isinstance(timing, dict):
                raise ValueError("timing.json inválido")
            timing["all_command_seconds"] = time.perf_counter() - all_started
            atomic_write_json(report.timing, timing)
        print(f"análise: tables={len(report.tables)}, figures={len(report.figures)}, diretório={report.pilot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
