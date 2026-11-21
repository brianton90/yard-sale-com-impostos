import numpy as np
import pytest

from simulation.metrics import compute_conservation_error, compute_gini, compute_max_share


def _gini_brute_force(wealth: np.ndarray) -> float:
    """Definição direta: G = sum_ij |w_i - w_j| / (2 N W)."""
    n = len(wealth)
    diffs = np.abs(wealth[:, None] - wealth[None, :]).sum()
    return float(diffs / (2.0 * n * wealth.sum()))


def test_gini_equal_distribution() -> None:
    wealth = np.array([1.0, 1.0, 1.0, 1.0])
    assert abs(compute_gini(wealth)) < 1e-12


def test_gini_extreme_distribution() -> None:
    wealth = np.array([0.0, 0.0, 0.0, 10.0])
    assert compute_gini(wealth) > 0.7


def test_gini_intermediate_distribution() -> None:
    wealth = np.array([1.0, 2.0, 3.0, 4.0])
    g = compute_gini(wealth)
    assert 0.0 < g < 1.0


def test_gini_order_invariance() -> None:
    wealth1 = np.array([1.0, 2.0, 3.0, 4.0])
    wealth2 = np.array([4.0, 1.0, 3.0, 2.0])
    assert abs(compute_gini(wealth1) - compute_gini(wealth2)) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n", [2, 5, 17, 60])
def test_gini_matches_brute_force_double_sum(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    wealth = rng.exponential(1.0, size=n)
    assert compute_gini(wealth) == pytest.approx(_gini_brute_force(wealth), abs=1e-12)


def test_gini_hand_computed_value() -> None:
    # [1, 2, 3, 4]: sum|wi-wj| = 2*(1+2+3+1+2+1) = 20; G = 20 / (2*4*10) = 0.25
    assert compute_gini(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.25)


def test_gini_is_scale_invariant() -> None:
    wealth = np.random.default_rng(7).exponential(1.0, size=40)
    for k in (0.001, 3.0, 1e6):
        assert compute_gini(k * wealth) == pytest.approx(compute_gini(wealth), abs=1e-12)


def test_gini_of_total_concentration_approaches_n_minus_one_over_n() -> None:
    for n in (2, 10, 1000):
        wealth = np.zeros(n)
        wealth[-1] = 7.0
        assert compute_gini(wealth) == pytest.approx((n - 1) / n)


def test_gini_of_all_zero_wealth_is_zero() -> None:
    assert compute_gini(np.zeros(5)) == 0.0


def test_max_share_basics() -> None:
    assert compute_max_share(np.array([1.0, 1.0, 2.0])) == pytest.approx(0.5)
    assert compute_max_share(np.full(10, 3.0)) == pytest.approx(0.1)
    assert compute_max_share(np.array([0.0, 0.0, 4.0])) == pytest.approx(1.0)
    assert compute_max_share(np.zeros(3)) == 0.0


def test_conservation_error_basics() -> None:
    wealth = np.array([1.0, 2.0, 3.0])
    assert compute_conservation_error(wealth, cash=0.0, initial_total=6.0) == 0.0
    assert compute_conservation_error(wealth, cash=1.0, initial_total=7.0) == 0.0
    # Relativo a W0: |6 + 0 - 12| / 12 = 0.5; vale também para desvio negativo.
    assert compute_conservation_error(wealth, cash=0.0, initial_total=12.0) == pytest.approx(0.5)
    assert compute_conservation_error(wealth, cash=3.0, initial_total=6.0) == pytest.approx(0.5)
