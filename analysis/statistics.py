from collections.abc import Mapping, Sequence
from math import sqrt
from typing import Any

import numpy as np


def inclusive_window(steps: np.ndarray, values: np.ndarray, start: int, end: int) -> np.ndarray:
    steps = np.asarray(steps)
    values = np.asarray(values, dtype=float)
    if steps.ndim != 1 or values.ndim != 1 or steps.shape != values.shape:
        raise ValueError("steps e values devem ser vetores do mesmo tamanho")
    selected = (steps >= start) & (steps <= end)
    if not np.any(selected):
        raise ValueError("a janela não contém observações")
    if steps[selected][0] != start or steps[selected][-1] != end:
        raise ValueError("os limites da janela devem existir na série")
    return values[selected]


def window_mean(steps: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    return float(np.mean(inclusive_window(steps, values, start, end)))


def mean_and_standard_error(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("são necessárias ao menos duas realizações")
    if not np.all(np.isfinite(array)):
        raise ValueError("valores não finitos")
    return float(array.mean()), float(array.std(ddof=1) / sqrt(len(array)))


def series_mean_and_standard_error(series: Sequence[np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(series, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("series deve ter shape (R, T), com R >= 2")
    if not np.all(np.isfinite(array)):
        raise ValueError("séries não finitas")
    return array.mean(axis=0), array.std(axis=0, ddof=1) / sqrt(array.shape[0])


def temporal_then_realization_summary(
    steps: np.ndarray,
    series: Sequence[np.ndarray] | np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, float, float]:
    array = np.asarray(series, dtype=float)
    if array.ndim != 2:
        raise ValueError("series deve ter shape (R, T)")
    per_realization = np.array([window_mean(steps, row, start, end) for row in array])
    mean, standard_error = mean_and_standard_error(per_realization)
    return per_realization, mean, standard_error


def paired_difference(
    wealth_records: Sequence[Mapping[str, Any]],
    transaction_records: Sequence[Mapping[str, Any]],
    value_key: str = "value",
) -> tuple[np.ndarray, float, float]:
    def index(records: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
        indexed: dict[int, Mapping[str, Any]] = {}
        for record in records:
            realization_id = int(record["realization_id"])
            if realization_id in indexed:
                raise ValueError("realization_id duplicado")
            indexed[realization_id] = record
        return indexed

    wealth = index(wealth_records)
    transaction = index(transaction_records)
    if len(wealth) < 2 or wealth.keys() != transaction.keys():
        raise ValueError("pareamento exige os mesmos R >= 2 realization_id")

    differences: list[float] = []
    comparison_fields = (
        "common_randomness_sha256",
        "protocol_sha256",
        "window_start",
        "window_end",
        "steps_sha256",
        "tax_rate",
    )
    for realization_id in sorted(wealth):
        left = wealth[realization_id]
        right = transaction[realization_id]
        if "regime" in left or "regime" in right:
            if left.get("regime") != "wealth" or right.get("regime") != "transaction":
                raise ValueError("pareamento exige wealth - transaction")
        if "configuration_tag" in left and not str(left["configuration_tag"]).startswith("wealth_tau"):
            raise ValueError("configuração esquerda não é patrimonial")
        if "configuration_tag" in right and not str(right["configuration_tag"]).startswith("transaction_tau"):
            raise ValueError("configuração direita não é transacional")
        for field in comparison_fields:
            if field in left or field in right:
                if left.get(field) != right.get(field):
                    raise ValueError(f"par incompatível no campo {field}")
        differences.append(float(left[value_key]) - float(right[value_key]))
    values = np.asarray(differences)
    mean, standard_error = mean_and_standard_error(values)
    return values, mean, standard_error
