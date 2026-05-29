import numpy as np

from simulation.metrics import compute_gini


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
