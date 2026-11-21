import argparse
from pathlib import Path
from typing import Any

from experiments.configuration import ExperimentConfig, validate_experiment_id


def _add_location_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-id", default="main", help="identificador do experimento")
    parser.add_argument(
        "--results-root",
        default="Results/experiments",
        help="diretório que contém os experiment_id",
    )


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    _add_location_arguments(parser)
    parser.add_argument("--n-realizations", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--initial-wealth", type=float, default=1.0)
    parser.add_argument("--transient-start", type=int, default=150)
    parser.add_argument("--analysis-end", type=int, default=300)
    parser.add_argument("--mobility-delta", type=int, default=50)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=2025)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa e analisa o experimento Yard-Sale com nove configurações."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate", help="simula ou retoma os resultados brutos")
    _add_simulation_arguments(simulate)
    analyze = subparsers.add_parser("analyze", help="recria tabelas e figuras somente do disco")
    _add_location_arguments(analyze)
    all_parser = subparsers.add_parser("all", help="simula/retoma e depois analisa")
    _add_simulation_arguments(all_parser)
    return parser


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    values: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "results_root": args.results_root,
        "n_realizations": args.n_realizations,
        "n_agents": args.n_agents,
        "n_steps": args.n_steps,
        "initial_wealth": args.initial_wealth,
        "transient_start": args.transient_start,
        "analysis_end": args.analysis_end,
        "mobility_delta": args.mobility_delta,
        "snapshot_interval": args.snapshot_interval,
        "base_seed": args.base_seed,
    }
    return ExperimentConfig(**values)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_id = validate_experiment_id(args.experiment_id)
    experiment_dir = Path(args.results_root) / experiment_id

    if args.command in ("simulate", "all"):
        from experiments.runner import run_experiment

        summary = run_experiment(_config_from_args(args))
        print(
            f"simulação: total={summary.total}, completed={summary.completed}, "
            f"skipped={summary.skipped}, diretório={summary.experiment_dir}"
        )

    if args.command in ("analyze", "all"):
        from analysis.reporting import generate_report

        report = generate_report(experiment_dir)
        print(
            f"análise: tables={len(report.tables)}, figures={len(report.figures)}, "
            f"diretório={report.experiment_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
