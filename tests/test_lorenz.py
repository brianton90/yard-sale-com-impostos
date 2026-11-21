import numpy as np
import pytest

from simulation.metrics import compute_lorenz


def test_lorenz_equal_and_concentrated() -> None:
    population, equal = compute_lorenz(np.ones(5))
    np.testing.assert_allclose(equal, population)
    _, concentrated = compute_lorenz(np.array([0, 0, 0, 0, 5.0]))
    np.testing.assert_array_equal(concentrated[:-1], 0.0)
    assert concentrated[-1] == 1.0
    assert np.all(np.diff(concentrated) >= 0.0)


@pytest.mark.parametrize("wealth", [np.array([]), np.array([0.0, 0.0]), np.array([1.0, -1.0]), np.array([np.nan])])
def test_lorenz_rejects_invalid_input(wealth) -> None:
    with pytest.raises(ValueError):
        compute_lorenz(wealth)
