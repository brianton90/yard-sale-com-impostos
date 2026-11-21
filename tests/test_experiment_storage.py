import numpy as np

from experiments.storage import atomic_write_json, atomic_write_npz, load_npz, read_json


def test_atomic_json_and_npz_roundtrip(tmp_path) -> None:
    json_path = tmp_path / "metadata.json"
    npz_path = tmp_path / "arrays.npz"
    atomic_write_json(json_path, {"b": 2, "a": 1})
    assert read_json(json_path) == {"a": 1, "b": 2}
    atomic_write_npz(npz_path, values=np.arange(5), label=np.array("ok"))
    loaded = load_npz(npz_path)
    np.testing.assert_array_equal(loaded["values"], np.arange(5))
    assert loaded["label"].item() == "ok"
    assert not list(tmp_path.glob(".*.tmp"))
