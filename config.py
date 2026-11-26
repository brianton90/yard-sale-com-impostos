from dataclasses import dataclass

REGIMES = ("none", "wealth", "transaction")
TAX_GRID = (0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the Yard-Sale simulation."""

    n_agents: int = 10000
    n_steps: int = 1200
    initial_wealth: float = 1.0
    seed: int = 42
    hist_bins: int = 50
    snapshot_interval: int = 10  # salva um snapshot a cada k passos de Monte Carlo
    results_dir: str = "Results"  # onde as figuras são salvas
    show_plots: bool = False  # abre a janela do gráfico além de salvar
    regime: str = "none"  # "none" (sem imposto), "wealth" (patrimonial) ou "transaction"
    tax_rate: float = 0.05  # alíquota τ do regime escolhido (ignorada em "none")

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"regime deve ser um de {REGIMES}, recebido: {self.regime!r}")
        if not 0.0 <= self.tax_rate <= 1.0:
            raise ValueError(f"tax_rate deve estar em [0, 1], recebido: {self.tax_rate}")

    @property
    def wealth_tax_rate(self) -> float:
        """Alíquota τ_W aplicada ao fim de cada passo (0 fora do regime patrimonial)."""
        return self.tax_rate if self.regime == "wealth" else 0.0

    @property
    def transaction_tax_rate(self) -> float:
        """Alíquota τ_X aplicada ao valor recebido pelo vencedor (0 fora do regime de transações)."""
        return self.tax_rate if self.regime == "transaction" else 0.0
