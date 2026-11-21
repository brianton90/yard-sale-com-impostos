import numpy as np
import pytest

from config import REGIMES, SimulationConfig
from simulation.engine import run_simulation
from simulation.initialization import initialize_state


def _config(regime: str = "none", tax_rate: float = 0.05, **kwargs: object) -> SimulationConfig:
    """Configuração pequena e rápida; nunca usa o N padrão (10000)."""
    params: dict[str, object] = dict(n_agents=100, n_steps=20, seed=11, snapshot_interval=5)
    params.update(kwargs)
    return SimulationConfig(regime=regime, tax_rate=tax_rate, **params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Invariantes exatas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime,tax_rate", [("none", 0.05), ("wealth", 0.05), ("wealth", 0.3),
                                              ("transaction", 0.05), ("transaction", 0.3)])
def test_total_wealth_is_conserved(regime: str, tax_rate: float) -> None:
    config = _config(regime, tax_rate)
    result = run_simulation(config)
    assert result.metrics.max_conservation_error < 1e-12
    assert result.state.wealth.sum() == pytest.approx(config.n_agents * config.initial_wealth, abs=1e-9)
    assert result.state.cash == 0.0


@pytest.mark.parametrize("regime,tax_rate", [("none", 0.0), ("wealth", 0.05), ("wealth", 1.0),
                                              ("transaction", 0.3), ("transaction", 1.0)])
def test_wealth_never_negative(regime: str, tax_rate: float) -> None:
    result = run_simulation(_config(regime, tax_rate))
    assert result.metrics.min_wealth_observed >= 0.0
    assert result.state.wealth.min() >= 0.0
    assert result.history.wealth.min() >= 0.0


def test_cash_is_collected_only_in_taxed_regimes() -> None:
    assert run_simulation(_config("none", 0.3)).metrics.max_cash == 0.0
    assert run_simulation(_config("wealth", 0.3)).metrics.max_cash > 0.0
    assert run_simulation(_config("transaction", 0.3)).metrics.max_cash > 0.0


# ---------------------------------------------------------------------------
# Casos-limite entre regimes
# ---------------------------------------------------------------------------


def test_zero_tax_rate_makes_all_regimes_identical() -> None:
    results = [run_simulation(_config(regime, 0.0)) for regime in REGIMES]
    base = results[0]
    for other in results[1:]:
        np.testing.assert_array_equal(other.state.wealth, base.state.wealth)
        np.testing.assert_array_equal(other.series.gini, base.series.gini)
        np.testing.assert_array_equal(other.series.max_share, base.series.max_share)
        np.testing.assert_array_equal(other.series.volume, base.series.volume)
        np.testing.assert_array_equal(other.series.revenue, base.series.revenue)


def test_positive_tax_changes_outcome_but_shares_initial_aversion() -> None:
    base_config = _config("none", 0.2)
    base = run_simulation(base_config)
    base_aversion = initialize_state(base_config, np.random.default_rng(base_config.seed)).aversion
    for regime in ("wealth", "transaction"):
        config = _config(regime, 0.2)
        # Números aleatórios comuns: a mesma semente gera as mesmas aversões iniciais.
        aversion = initialize_state(config, np.random.default_rng(config.seed)).aversion
        np.testing.assert_array_equal(aversion, base_aversion)
        taxed = run_simulation(config)
        np.testing.assert_array_equal(taxed.state.aversion, base.state.aversion)
        assert not np.allclose(taxed.state.wealth, base.state.wealth)


def test_same_seed_is_bit_identical_and_different_seeds_differ() -> None:
    a = run_simulation(_config("transaction", 0.1, seed=5))
    b = run_simulation(_config("transaction", 0.1, seed=5))
    c = run_simulation(_config("transaction", 0.1, seed=6))
    np.testing.assert_array_equal(a.state.wealth, b.state.wealth)
    np.testing.assert_array_equal(a.series.gini, b.series.gini)
    assert not np.array_equal(a.state.wealth, c.state.wealth)


# ---------------------------------------------------------------------------
# Física qualitativa (execuções pequenas, semente fixa)
# ---------------------------------------------------------------------------


def test_without_tax_inequality_grows() -> None:
    # N = 200, 100 passos: suficiente para a concentração aparecer com folga.
    result = run_simulation(_config("none", 0.0, n_agents=200, n_steps=100, seed=1))
    assert result.metrics.gini > 0.0  # Gini inicial é 0 (riqueza igual)
    assert result.metrics.max_share > 1.0 / 200
    g = result.series.gini
    quarter = len(g) // 4
    assert g[-quarter:].mean() > g[:quarter].mean()


def test_higher_wealth_tax_gives_lower_gini() -> None:
    # Tamanhos escolhidos: N = 500, 150 passos, média de 3 sementes (~1 s no total).
    # O Gini estacionário cai com τ_W; as taxas 0, 0.02 e 0.2 são bem separadas.
    def mean_gini(regime: str, tax_rate: float) -> float:
        ginis = [
            run_simulation(_config(regime, tax_rate, n_agents=500, n_steps=150, seed=s)).metrics.gini
            for s in (1, 2, 3)
        ]
        return float(np.mean(ginis))

    g_none = mean_gini("none", 0.0)
    g_low = mean_gini("wealth", 0.02)
    g_high = mean_gini("wealth", 0.2)
    assert g_high < g_low < g_none


# ---------------------------------------------------------------------------
# Configuração e estrutura da saída
# ---------------------------------------------------------------------------


def test_config_rejects_invalid_regime_and_tax_rate() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(regime="invalido")
    for rate in (-0.01, 1.01):
        with pytest.raises(ValueError):
            SimulationConfig(regime="wealth", tax_rate=rate)
    SimulationConfig(regime="wealth", tax_rate=0.0)
    SimulationConfig(regime="wealth", tax_rate=1.0)


def test_tax_rate_properties_apply_only_to_their_own_regime() -> None:
    assert SimulationConfig(regime="wealth", tax_rate=0.3).wealth_tax_rate == 0.3
    assert SimulationConfig(regime="wealth", tax_rate=0.3).transaction_tax_rate == 0.0
    assert SimulationConfig(regime="transaction", tax_rate=0.3).transaction_tax_rate == 0.3
    assert SimulationConfig(regime="transaction", tax_rate=0.3).wealth_tax_rate == 0.0
    assert SimulationConfig(regime="none", tax_rate=0.3).wealth_tax_rate == 0.0
    assert SimulationConfig(regime="none", tax_rate=0.3).transaction_tax_rate == 0.0


def test_engine_rejects_invalid_snapshot_interval() -> None:
    with pytest.raises(ValueError):
        run_simulation(_config(snapshot_interval=0))


def test_history_and_series_shapes() -> None:
    config = _config("wealth", 0.1, n_agents=60, n_steps=23, snapshot_interval=5)
    result = run_simulation(config)
    assert result.history.steps[0] == 0
    assert result.history.steps[-1] == config.n_steps
    assert list(result.history.steps) == [0, 5, 10, 15, 20, 23]
    assert result.history.wealth.shape == (len(result.history.steps), config.n_agents)
    np.testing.assert_array_equal(result.history.wealth[0], np.full(config.n_agents, config.initial_wealth))
    np.testing.assert_array_equal(result.history.wealth[-1], result.state.wealth)
    for arr in (result.series.steps, result.series.gini, result.series.max_share,
                result.series.volume, result.series.revenue):
        assert arr.shape == (config.n_steps,)


def test_initial_state_has_equal_wealth_and_uniform_aversion() -> None:
    config = SimulationConfig(n_agents=20000, initial_wealth=2.0, seed=3)
    state = initialize_state(config, np.random.default_rng(config.seed))
    np.testing.assert_array_equal(state.wealth, np.full(config.n_agents, 2.0))
    assert state.cash == 0.0
    assert state.aversion.min() >= 0.0 and state.aversion.max() < 1.0
    # U(0,1): média 0.5, desvio 1/sqrt(12); tolerância de 4 erros-padrão.
    stderr = np.sqrt(1.0 / 12.0) / np.sqrt(config.n_agents)
    assert abs(state.aversion.mean() - 0.5) < 4 * stderr
