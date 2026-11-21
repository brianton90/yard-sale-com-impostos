from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from config import REGIMES, TAX_GRID, SimulationConfig
from experiments.storage import json_sha256


SCHEMA_VERSION = 1
_EXPERIMENT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_experiment_id(experiment_id: str) -> str:
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id deve começar com caractere alfanumérico e conter somente "
            "letras ASCII, números, ponto, sublinhado ou hífen"
        )
    return experiment_id


@dataclass(frozen=True)
class ConfigurationKey:
    regime: str
    tax_rate: float

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"regime inválido: {self.regime!r}")
        if self.regime == "none" and self.tax_rate != 0.0:
            raise ValueError("a configuração none deve usar tax_rate=0.0")
        if self.regime != "none" and self.tax_rate not in TAX_GRID:
            raise ValueError(f"alíquota fora da grade canônica: {self.tax_rate}")

    @property
    def tag(self) -> str:
        return configuration_tag(self.regime, self.tax_rate)


def configuration_tag(regime: str, tax_rate: float) -> str:
    if regime not in REGIMES:
        raise ValueError(f"regime inválido: {regime!r}")
    if regime == "none":
        if float(tax_rate) != 0.0:
            raise ValueError("a configuração none deve usar tax_rate=0.0")
        return "none"
    if float(tax_rate) not in TAX_GRID:
        raise ValueError(f"alíquota fora da grade canônica: {tax_rate}")
    return f"{regime}_tau{float(tax_rate):g}"


def build_configurations(tax_rates: tuple[float, ...] = TAX_GRID) -> tuple[ConfigurationKey, ...]:
    if tuple(tax_rates) != TAX_GRID:
        raise ValueError(f"tax_rates deve ser exatamente {TAX_GRID}")
    return (
        ConfigurationKey("none", 0.0),
        *(ConfigurationKey("wealth", rate) for rate in tax_rates),
        *(ConfigurationKey("transaction", rate) for rate in tax_rates),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = "main"
    results_root: str = "Results/experiments"
    n_realizations: int = 10
    n_agents: int = 10000
    n_steps: int = 300
    initial_wealth: float = 1.0
    snapshot_interval: int = 10
    base_seed: int = 2025
    transient_start: int = 150
    analysis_end: int = 300
    mobility_delta: int = 50
    tax_rates: tuple[float, ...] = TAX_GRID
    conservation_tolerance: float = 1e-12
    nonnegative_tolerance: float = 1e-12
    cash_tolerance: float = 1e-12
    stability_status: str = "not_assessed"
    interpretation: str = "finite_time"

    def __post_init__(self) -> None:
        validate_experiment_id(self.experiment_id)
        if self.n_realizations < 2:
            raise ValueError("n_realizations deve ser >= 2")
        if self.n_agents < 2 or self.n_agents % 5 != 0:
            raise ValueError("n_agents deve ser >= 2 e divisível por 5")
        if self.n_steps < 1:
            raise ValueError("n_steps deve ser >= 1")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval deve ser >= 1")
        if self.initial_wealth <= 0.0:
            raise ValueError("initial_wealth deve ser positivo")
        if tuple(self.tax_rates) != TAX_GRID:
            raise ValueError(f"tax_rates deve ser exatamente {TAX_GRID}")
        if not 1 <= self.transient_start < self.analysis_end <= self.n_steps:
            raise ValueError("exigir 1 <= transient_start < analysis_end <= n_steps")
        if self.mobility_delta <= 0:
            raise ValueError("mobility_delta deve ser positivo")
        if self.transient_start + self.mobility_delta > self.analysis_end:
            raise ValueError("transient_start + mobility_delta deve ser <= analysis_end")
        if min(self.conservation_tolerance, self.nonnegative_tolerance, self.cash_tolerance) < 0.0:
            raise ValueError("tolerâncias devem ser não negativas")
        if self.stability_status != "not_assessed" or self.interpretation != "finite_time":
            raise ValueError("esta fase exige status not_assessed e interpretação finite_time")

    @property
    def configurations(self) -> tuple[ConfigurationKey, ...]:
        return build_configurations(self.tax_rates)

    @property
    def experiment_dir(self) -> Path:
        return Path(self.results_root) / self.experiment_id

    @property
    def required_snapshot_steps(self) -> tuple[int, ...]:
        return tuple(sorted({
            0,
            self.transient_start,
            self.transient_start + self.mobility_delta,
            self.analysis_end,
            self.n_steps,
        }))

    def simulation_config(self, key: ConfigurationKey) -> SimulationConfig:
        return SimulationConfig(
            n_agents=self.n_agents,
            n_steps=self.n_steps,
            initial_wealth=self.initial_wealth,
            seed=self.base_seed,
            snapshot_interval=self.snapshot_interval,
            regime=key.regime,
            tax_rate=key.tax_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tax_rates"] = list(self.tax_rates)
        data["configuration_tags"] = [key.tag for key in self.configurations]
        data["stability_status_by_configuration"] = {
            key.tag: self.stability_status for key in self.configurations
        }
        data["required_snapshot_steps"] = list(self.required_snapshot_steps)
        data["temporal_parameters_provisional"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        fields = {
            name: data[name]
            for name in cls.__dataclass_fields__
            if name in data
        }
        if "tax_rates" in fields:
            fields["tax_rates"] = tuple(float(value) for value in fields["tax_rates"])
        return cls(**fields)


def protocol_document(config: ExperimentConfig) -> dict[str, Any]:
    fields = config.to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": json_sha256(fields),
        **fields,
    }
