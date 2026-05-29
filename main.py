from config import SimulationConfig
from simulation.engine import run_simulation
from visualization.plots import plot_wealth_distribution


def main() -> None:
    config = SimulationConfig()
    state, metrics = run_simulation(config)

    print("Resumo da simulação")
    print(f"Agentes: {config.n_agents}")
    print(f"Passos de tempo: {config.n_steps}")
    print(f"Riqueza total: {metrics.total_wealth:.4f}")
    print(f"Riqueza média: {metrics.mean_wealth:.4f}")
    print(f"Riqueza mediana: {metrics.median_wealth:.4f}")
    print(f"Riqueza mínima: {metrics.min_wealth:.4f}")
    print(f"Riqueza máxima: {metrics.max_wealth:.4f}")
    print(f"Gini: {metrics.gini:.4f}")

    plot_wealth_distribution(state.wealth, bins=config.hist_bins)


if __name__ == "__main__":
    main()
