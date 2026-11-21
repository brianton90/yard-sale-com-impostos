import numpy as np
import pytest

from analysis.statistics import (
    mean_and_standard_error,
    paired_difference,
    temporal_then_realization_summary,
)


def test_temporal_then_realization_order_and_ddof() -> None:
    steps = np.arange(1, 5)
    series = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    per_realization, mean, se = temporal_then_realization_summary(steps, series, 2, 4)
    np.testing.assert_allclose(per_realization, [3.0, 5.0])
    assert mean == pytest.approx(4.0)
    assert se == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mean_and_standard_error([1.0])


def test_paired_difference_validates_common_inputs() -> None:
    base = {
        "protocol_sha256": "p", "window_start": 2, "window_end": 4,
        "steps_sha256": "s", "tax_rate": 0.1,
    }
    wealth = [dict(base, realization_id=0, common_randomness_sha256="a", value=5.0),
              dict(base, realization_id=1, common_randomness_sha256="b", value=8.0)]
    transaction = [dict(base, realization_id=0, common_randomness_sha256="a", value=3.0),
                   dict(base, realization_id=1, common_randomness_sha256="b", value=4.0)]
    values, mean, se = paired_difference(wealth, transaction)
    np.testing.assert_array_equal(values, [2.0, 4.0])
    assert mean == pytest.approx(3.0)
    assert se == pytest.approx(1.0)
    transaction[0]["common_randomness_sha256"] = "wrong"
    with pytest.raises(ValueError):
        paired_difference(wealth, transaction)
