from typing import Any

import numpy as np


def _validated_vectors(wealth: np.ndarray, tiebreak_key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wealth = np.asarray(wealth, dtype=float)
    tiebreak_key = np.asarray(tiebreak_key)
    if wealth.ndim != 1 or tiebreak_key.ndim != 1 or wealth.shape != tiebreak_key.shape:
        raise ValueError("wealth e tiebreak_key devem ser vetores do mesmo tamanho")
    if len(wealth) == 0 or len(wealth) % 5 != 0:
        raise ValueError("a população deve ser não vazia e divisível por 5")
    if not np.all(np.isfinite(wealth)) or not np.all(np.isfinite(tiebreak_key)):
        raise ValueError("wealth e tiebreak_key devem ser finitos")
    if len(np.unique(tiebreak_key)) != len(tiebreak_key):
        raise ValueError("tiebreak_key deve conter chaves únicas")
    return wealth, tiebreak_key


def assign_quintiles(wealth: np.ndarray, tiebreak_key: np.ndarray) -> np.ndarray:
    wealth, tiebreak_key = _validated_vectors(wealth, tiebreak_key)
    order = np.lexsort((tiebreak_key, wealth))
    quintiles = np.empty(len(wealth), dtype=np.int8)
    group_size = len(wealth) // 5
    quintiles[order] = np.repeat(np.arange(1, 6, dtype=np.int8), group_size)
    return quintiles


def boundary_tie_records(
    wealth: np.ndarray,
    tiebreak_key: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    wealth = np.asarray(wealth, dtype=float)
    if tiebreak_key is None:
        tiebreak_key = np.arange(len(wealth))
    wealth, tiebreak_key = _validated_vectors(wealth, np.asarray(tiebreak_key))
    ordered = wealth[np.lexsort((tiebreak_key, wealth))]
    group_size = len(wealth) // 5
    records: list[dict[str, Any]] = []
    for boundary in range(1, 5):
        cut = boundary * group_size
        tied = bool(ordered[cut - 1] == ordered[cut])
        value = float(ordered[cut]) if tied else None
        multiplicity = int(np.count_nonzero(wealth == value)) if tied else 0
        records.append({
            "boundary": boundary,
            "tied": tied,
            "tied_value": value,
            "multiplicity": multiplicity,
        })
    return records


def transition_matrix(
    origin_quintiles: np.ndarray,
    destination_quintiles: np.ndarray,
    origin_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if origin_step == 0:
        raise ValueError("t=0 não pode ser origem da análise de mobilidade")
    origin = np.asarray(origin_quintiles)
    destination = np.asarray(destination_quintiles)
    if origin.ndim != 1 or destination.ndim != 1 or origin.shape != destination.shape:
        raise ValueError("os quintis devem ser vetores do mesmo tamanho")
    if len(origin) == 0 or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(destination)):
        raise ValueError("rótulos de quintil devem ser finitos")
    if np.any(origin != origin.astype(int)) or np.any(destination != destination.astype(int)):
        raise ValueError("rótulos de quintil devem ser inteiros")
    if np.any((origin < 1) | (origin > 5)) or np.any((destination < 1) | (destination > 5)):
        raise ValueError("rótulos de quintil devem estar entre 1 e 5")
    origin_int = origin.astype(int)
    destination_int = destination.astype(int)
    expected_group_size = len(origin) // 5
    if len(origin) % 5 != 0 or any(
        np.count_nonzero(labels == quintile) != expected_group_size
        for labels in (origin_int, destination_int)
        for quintile in range(1, 6)
    ):
        raise ValueError("cada vetor deve conter cinco quintis de mesmo tamanho")
    counts = np.zeros((5, 5), dtype=np.int64)
    np.add.at(counts, (origin_int - 1, destination_int - 1), 1)
    row_totals = counts.sum(axis=1)
    if np.any(row_totals == 0):
        raise ValueError("cada quintil de origem deve conter agentes")
    probabilities = counts / row_totals[:, None]
    return counts, probabilities
