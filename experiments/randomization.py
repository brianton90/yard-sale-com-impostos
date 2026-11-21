from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class SimulationRandomness:
    aversion: np.ndarray
    pairs: np.ndarray
    first_agent_wins: np.ndarray

    def validate(self, n_agents: int, n_steps: int) -> None:
        trades_per_step = n_agents // 2
        if self.aversion.shape != (n_agents,):
            raise ValueError("aversion tem shape incompatível")
        if self.pairs.shape != (n_steps, trades_per_step, 2):
            raise ValueError("pairs tem shape incompatível")
        if self.first_agent_wins.shape != (n_steps, trades_per_step):
            raise ValueError("first_agent_wins tem shape incompatível")
        if not np.issubdtype(self.aversion.dtype, np.floating):
            raise ValueError("aversion deve ter dtype de ponto flutuante")
        if not np.issubdtype(self.pairs.dtype, np.integer):
            raise ValueError("pairs deve ter dtype inteiro")
        if self.first_agent_wins.dtype != np.bool_:
            raise ValueError("first_agent_wins deve ter dtype bool")
        if not np.all(np.isfinite(self.aversion)):
            raise ValueError("aversion contém valor não finito")
        if np.any(self.aversion < 0.0) or np.any(self.aversion >= 1.0):
            raise ValueError("aversion deve estar em [0, 1)")
        if np.any(self.pairs < 0) or np.any(self.pairs >= n_agents):
            raise ValueError("pairs contém índice inválido")
        if np.any(self.pairs[..., 0] == self.pairs[..., 1]):
            raise ValueError("pairs deve conter agentes distintos")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def generate_realization_randomness(
    realization_seed: np.random.SeedSequence,
    n_agents: int,
    n_steps: int,
) -> tuple[SimulationRandomness, np.ndarray]:
    if n_agents < 2:
        raise ValueError("n_agents deve ser >= 2")
    if n_steps < 1:
        raise ValueError("n_steps deve ser >= 1")
    population_seed, events_seed, tiebreak_seed = realization_seed.spawn(3)
    population_rng = np.random.default_rng(population_seed)
    events_rng = np.random.default_rng(events_seed)
    tiebreak_rng = np.random.default_rng(tiebreak_seed)

    aversion = population_rng.uniform(0.0, 1.0, size=n_agents)
    shape = (n_steps, n_agents // 2)
    first = events_rng.integers(0, n_agents, size=shape, dtype=np.int64)
    second = events_rng.integers(0, n_agents - 1, size=shape, dtype=np.int64)
    second += second >= first
    pairs = np.stack((first, second), axis=-1)
    first_agent_wins = events_rng.integers(0, 2, size=shape, dtype=np.int8).astype(bool)
    mobility_tiebreak = tiebreak_rng.permutation(n_agents).astype(np.int64)

    randomness = SimulationRandomness(
        aversion=_readonly(aversion),
        pairs=_readonly(pairs),
        first_agent_wins=_readonly(first_agent_wins),
    )
    randomness.validate(n_agents, n_steps)
    return randomness, _readonly(mobility_tiebreak)


def generate_all_randomness(
    base_seed: int,
    n_realizations: int,
    n_agents: int,
    n_steps: int,
) -> list[tuple[SimulationRandomness, np.ndarray]]:
    if n_realizations < 1:
        raise ValueError("n_realizations deve ser >= 1")
    return [
        generate_realization_randomness(seed, n_agents, n_steps)
        for seed in np.random.SeedSequence(base_seed).spawn(n_realizations)
    ]


def common_randomness_sha256(
    randomness: SimulationRandomness,
    mobility_tiebreak: np.ndarray,
    metadata: dict[str, object] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata or {}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, array in (
        ("aversion", randomness.aversion),
        ("pairs", randomness.pairs),
        ("first_agent_wins", randomness.first_agent_wins),
        ("mobility_tiebreak", mobility_tiebreak),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()
