import numpy as np
import pytest

from simulation.mobility import assign_quintiles, boundary_tie_records, transition_matrix


def test_quintiles_preserve_identity_and_have_equal_sizes() -> None:
    wealth = np.array([5, 1, 4, 2, 3, 10, 6, 9, 7, 8], dtype=float)
    key = np.arange(10)[::-1]
    quintiles = assign_quintiles(wealth, key)
    assert [np.count_nonzero(quintiles == group) for group in range(1, 6)] == [2] * 5
    moved = quintiles[::-1]
    counts, probabilities = transition_matrix(quintiles, moved, origin_step=3)
    assert counts.sum() == 10
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        transition_matrix(quintiles, moved, origin_step=0)


def test_boundary_ties_are_auditable() -> None:
    wealth = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3], dtype=float)
    records = boundary_tie_records(wealth, np.arange(10))
    assert len(records) == 4
    assert records[1]["tied"] is True
    assert records[1]["tied_value"] == 1.0
    assert records[1]["multiplicity"] == 4
    with pytest.raises(ValueError):
        assign_quintiles(wealth, np.zeros(10))
