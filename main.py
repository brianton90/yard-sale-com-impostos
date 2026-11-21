from pathlib import Path

from config import SimulationConfig
from simulation.engine import run_simulation
from visualization.plots import plot_time_series, plot_wealth_distribution


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{seconds:.2f} s ({int(minutes)} min {secs:.2f} s)"


def main() -> None:
    config = SimulationConfig()
    result = run_simulation(config)
    metrics = result.metrics

    tag = config.regime if config.regime == "none" else f"{config.regime}_tau{config.tax_rate:g}"

    print("Resumo da simulação")
    print(f"Regime: {config.regime}" + ("" if config.regime == "none" else f" (alíquota {config.tax_rate:g})"))
    print(f"Agentes: {config.n_agents}")
    print(f"Passos de tempo: {config.n_steps}")
    print(f"Riqueza total: {metrics.total_wealth:.4f}")
    print(f"Riqueza média: {metrics.mean_wealth:.4f}")
    print(f"Riqueza mediana: {metrics.median_wealth:.4f}")
    print(f"Riqueza mínima: {metrics.min_wealth:.4f}")
    print(f"Riqueza máxima: {metrics.max_wealth:.4f}")
    print(f"Gini: {metrics.gini:.4f}")
    print(f"Participação do agente mais rico: {metrics.max_share:.4f}")
    print(f"Erro máximo de conservação: {metrics.max_conservation_error:.3e}")
    print(f"Menor saldo individual observado: {metrics.min_wealth_observed:.4e}")
    print(f"Maior saldo do caixa: {metrics.max_cash:.4f}")
    print(f"Snapshots salvos: {len(result.history.steps)} (a cada {config.snapshot_interval} passos)")
    print(f"Duração total da simulação: {format_duration(result.duration_seconds)}")

    results_dir = Path(config.results_dir)

    histogram_path = plot_wealth_distribution(
        result.state.wealth,
        output_path=results_dir / f"distribuicao_riqueza_{tag}.png",
        bins=config.hist_bins,
        show=config.show_plots,
    )
    print(f"Figura salva em: {histogram_path}")

    series = result.series
    series_path = plot_time_series(
        series.steps,
        {
            "Gini G(t)": series.gini,
            "Participação do mais rico": series.max_share,
            "Volume transacionado v(t)": series.volume,
            "Arrecadação q(t)": series.revenue,
        },
        output_path=results_dir / f"series_temporais_{tag}.png",
        title=f"Indicadores por passo – regime: {tag}",
        show=config.show_plots,
    )
    print(f"Figura salva em: {series_path}")


if __name__ == "__main__":
    main()
