import numpy as np
import pytest

from experiments.randomization import common_randomness_sha256, generate_all_randomness


def test_realizations_are_independent_and_valid() -> None:
    generated = generate_all_randomness(2025, 2, 20, 8)
    hashes = []
    for realization_id, (randomness, tiebreak) in enumerate(generated):
        randomness.validate(20, 8)
        assert randomness.pairs.shape == (8, 10, 2)
        assert np.all(randomness.pairs[..., 0] != randomness.pairs[..., 1])
        assert len(np.unique(tiebreak)) == 20
        hashes.append(common_randomness_sha256(randomness, tiebreak, {"id": realization_id}))
    assert hashes[0] != hashes[1]
    assert not np.array_equal(generated[0][0].aversion, generated[1][0].aversion)


def test_randomness_validation_rejects_invalid_pair() -> None:
    randomness, _ = generate_all_randomness(1, 2, 20, 8)[0]
    pairs = randomness.pairs.copy()
    pairs[0, 0] = (3, 3)
    invalid = type(randomness)(randomness.aversion, pairs, randomness.first_agent_wins)
    with pytest.raises(ValueError):
        invalid.validate(20, 8)
