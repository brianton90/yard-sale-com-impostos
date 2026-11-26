from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any

from config import TAX_GRID, SimulationConfig
from experiments.configuration import ConfigurationKey, build_configurations
from experiments.storage import json_sha256


PILOT_SCHEMA_VERSION = 1
_MARGINS_SCHEMA_VERSION = 1
_PILOT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MARGIN_METRICS = ("gini", "max_share", "volume")
_MARGIN_KEYS = {
    "schema_version",
    "K",
    "confidence",
    "margins",
    "tail_fraction",
    "joint_status",
}
_PROTECTED_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "Results" / "experiments"


def _is_protected_experiments_path(path: Path) -> bool:
    resolved_path = path.resolve()
    protected_root = _PROTECTED_EXPERIMENTS_ROOT.resolve()
    return resolved_path == protected_root or protected_root in resolved_path.parents


def validate_pilot_results_root(results_root: str) -> str:
    if not isinstance(results_root, str) or not results_root:
        raise ValueError("results_root deve ser texto não vazio")
    if _is_protected_experiments_path(Path(results_root)):
        raise ValueError("results_root não pode ser Results/experiments nem um subdiretório")
    return results_root


def validate_pilot_directory(pilot_dir: Path) -> Path:
    pilot_dir = Path(pilot_dir)
    if _is_protected_experiments_path(pilot_dir):
        raise ValueError("diretório do piloto não pode estar em Results/experiments")
    return pilot_dir


def validate_pilot_id(pilot_id: str) -> str:
    if not isinstance(pilot_id, str) or not _PILOT_ID_PATTERN.fullmatch(pilot_id):
        raise ValueError(
            "pilot_id deve começar com caractere alfanumérico e conter somente "
            "letras ASCII, números, ponto, sublinhado ou hífen"
        )
    return pilot_id


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} deve ser numérico")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} deve ser finito")
    return result


@dataclass(frozen=True)
class StabilityMargins:
    K: int
    confidence: float
    margins: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    tail_fraction: float
    joint_status: str

    def __post_init__(self) -> None:
        if not _is_int(self.K) or self.K < 2:
            raise ValueError("K deve ser inteiro >= 2")
        confidence = _finite_float(self.confidence, "confidence")
        if confidence not in (0.90, 0.95, 0.99):
            raise ValueError("confidence deve ser 0.90, 0.95 ou 0.99")
        tail_fraction = _finite_float(self.tail_fraction, "tail_fraction")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction deve estar em (0, 1]")
        if self.joint_status != "all_metrics_all_comparisons":
            raise ValueError("joint_status inválido")
        if not isinstance(self.margins, tuple) or tuple(name for name, _ in self.margins) != _MARGIN_METRICS:
            raise ValueError(f"margins deve conter exatamente {_MARGIN_METRICS}")
        for name, values in self.margins:
            expected = ("absolute", "relative") if name == "volume" else ("absolute",)
            if not isinstance(name, str) or not isinstance(values, tuple) or tuple(key for key, _ in values) != expected:
                raise ValueError(f"margins.{name} inválido")
            for key, value in values:
                if _finite_float(value, f"margins.{name}.{key}") <= 0.0:
                    raise ValueError(f"margins.{name}.{key} deve ser positivo e finito")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _MARGINS_SCHEMA_VERSION,
            "K": self.K,
            "confidence": self.confidence,
            "margins": {name: dict(values) for name, values in self.margins},
            "tail_fraction": self.tail_fraction,
            "joint_status": self.joint_status,
        }

    @property
    def sha256(self) -> str:
        return json_sha256(self.to_dict())

    def absolute_margin_for(self, metric: str) -> float:
        if metric not in _MARGIN_METRICS:
            raise ValueError(f"métrica de margem inválida: {metric}")
        return float(dict(dict(self.margins)[metric])["absolute"])

    def relative_margin_for(self, metric: str) -> float:
        if metric != "volume":
            return 0.0
        return float(dict(dict(self.margins)[metric])["relative"])

    def margin_for(self, metric: str) -> float:
        return self.absolute_margin_for(metric)


def stability_margins_document(margins: StabilityMargins) -> dict[str, Any]:
    fields = margins.to_dict()
    return {"margins_sha256": json_sha256(fields), **fields}


def _strict_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} deve ser um objeto JSON")
    return value


def _parse_stability_margins_fields(data: object) -> StabilityMargins:
    data = _strict_object(data, "configuração de margens")
    if set(data) != _MARGIN_KEYS:
        raise ValueError("configuração de margens contém campos ausentes ou desconhecidos")
    if not _is_int(data["schema_version"]) or data["schema_version"] != _MARGINS_SCHEMA_VERSION:
        raise ValueError(f"schema_version deve ser {_MARGINS_SCHEMA_VERSION}")
    margins = _strict_object(data["margins"], "margins")
    if set(margins) != set(_MARGIN_METRICS):
        raise ValueError(f"margins deve conter exatamente {_MARGIN_METRICS}")
    parsed_margins: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for name in _MARGIN_METRICS:
        values = _strict_object(margins[name], f"margins.{name}")
        expected = {"absolute", "relative"} if name == "volume" else {"absolute"}
        if set(values) != expected:
            raise ValueError(f"margins.{name} contém campos ausentes ou desconhecidos")
        parsed_margins.append(
            (name, tuple((key, _finite_float(values[key], f"margins.{name}.{key}")) for key in sorted(expected)))
        )
    if not _is_int(data["K"]):
        raise ValueError("K deve ser inteiro")
    if not isinstance(data["joint_status"], str):
        raise ValueError("joint_status deve ser texto")
    return StabilityMargins(
        K=data["K"],
        confidence=_finite_float(data["confidence"], "confidence"),
        margins=tuple(parsed_margins),
        tail_fraction=_finite_float(data["tail_fraction"], "tail_fraction"),
        joint_status=data["joint_status"],
    )


def load_stability_margins(path: Path) -> StabilityMargins:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON de margens inválido: {path}") from error
    return _parse_stability_margins_fields(data)


@dataclass(frozen=True)
class PilotConfig:
    pilot_id: str = "main"
    results_root: str = "Results/pilots"
    n_realizations: int = 10
    n_agents: int = 10000
    t_horizon: int = 300
    initial_wealth: float = 1.0
    snapshot_interval: int = 10
    base_seed: int = 2025
    conservation_tolerance: float = 1e-12
    nonnegative_tolerance: float = 1e-12
    cash_tolerance: float = 1e-12
    tax_rates: tuple[float, ...] = TAX_GRID

    def __post_init__(self) -> None:
        validate_pilot_id(self.pilot_id)
        validate_pilot_results_root(self.results_root)
        if not _is_int(self.n_realizations) or self.n_realizations < 2:
            raise ValueError("n_realizations deve ser inteiro >= 2")
        if not _is_int(self.n_agents) or self.n_agents < 2:
            raise ValueError("n_agents deve ser inteiro >= 2")
        if not _is_int(self.t_horizon) or self.t_horizon < 1:
            raise ValueError("t_horizon deve ser inteiro >= 1")
        if not _is_int(self.snapshot_interval) or self.snapshot_interval < 1:
            raise ValueError("snapshot_interval deve ser inteiro >= 1")
        if not _is_int(self.base_seed):
            raise ValueError("base_seed deve ser inteiro")
        if _finite_float(self.initial_wealth, "initial_wealth") <= 0.0:
            raise ValueError("initial_wealth deve ser positivo")
        tolerances = (
            _finite_float(self.conservation_tolerance, "conservation_tolerance"),
            _finite_float(self.nonnegative_tolerance, "nonnegative_tolerance"),
            _finite_float(self.cash_tolerance, "cash_tolerance"),
        )
        if min(tolerances) < 0.0:
            raise ValueError("tolerâncias devem ser não negativas")
        if tuple(self.tax_rates) != TAX_GRID:
            raise ValueError(f"tax_rates deve ser exatamente {TAX_GRID}")

    @property
    def configurations(self) -> tuple[ConfigurationKey, ...]:
        return build_configurations(self.tax_rates)

    @property
    def n_steps(self) -> int:
        return 2 * self.t_horizon

    @property
    def pilot_dir(self) -> Path:
        return Path(self.results_root) / self.pilot_id

    @property
    def required_snapshot_steps(self) -> tuple[int, ...]:
        return (0, self.t_horizon, self.n_steps)

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
        data["n_steps"] = self.n_steps
        data["required_snapshot_steps"] = list(self.required_snapshot_steps)
        data["trajectory"] = "single_common_2T"
        return data


def protocol_document(config: PilotConfig, margins: StabilityMargins) -> dict[str, Any]:
    fields = config.to_dict()
    fields["stability_margins_sha256"] = margins.sha256
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "protocol_sha256": json_sha256(fields),
        **fields,
    }


def stability_margins_from_document(document: object) -> StabilityMargins:
    document = _strict_object(document, "documento de margens")
    if set(document) != {"margins_sha256", *_MARGIN_KEYS}:
        raise ValueError("documento de margens inválido")
    if not isinstance(document["margins_sha256"], str):
        raise ValueError("hash das margens deve ser texto")
    fields = {name: document[name] for name in _MARGIN_KEYS}
    margins = _parse_stability_margins_fields(fields)
    if document != stability_margins_document(margins):
        raise ValueError("documento de margens incompatível")
    return margins


def _strict_text_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} deve ser uma lista de texto")
    return value


def pilot_config_from_protocol_document(document: object, margins: StabilityMargins) -> PilotConfig:
    document = _strict_object(document, "documento de protocolo")
    expected_keys = {
        "schema_version", "protocol_sha256", "configuration_tags", "n_steps",
        "required_snapshot_steps", "trajectory", "stability_margins_sha256",
        *PilotConfig.__dataclass_fields__,
    }
    if set(document) != expected_keys:
        raise ValueError("documento de protocolo inválido")
    if not _is_int(document["schema_version"]) or document["schema_version"] != PILOT_SCHEMA_VERSION:
        raise ValueError(f"schema_version deve ser {PILOT_SCHEMA_VERSION}")
    for name in ("protocol_sha256", "pilot_id", "results_root", "trajectory", "stability_margins_sha256"):
        if not isinstance(document[name], str):
            raise ValueError(f"{name} deve ser texto")
    for name in ("n_realizations", "n_agents", "t_horizon", "snapshot_interval", "base_seed", "n_steps"):
        if not _is_int(document[name]):
            raise ValueError(f"{name} deve ser inteiro")
    for name in ("initial_wealth", "conservation_tolerance", "nonnegative_tolerance", "cash_tolerance"):
        _finite_float(document[name], name)
    tax_rates = document["tax_rates"]
    if not isinstance(tax_rates, list):
        raise ValueError("tax_rates deve ser uma lista")
    for rate in tax_rates:
        _finite_float(rate, "tax_rates")
    required_steps = document["required_snapshot_steps"]
    if not isinstance(required_steps, list) or not all(_is_int(step) for step in required_steps):
        raise ValueError("required_snapshot_steps deve ser uma lista de inteiros")
    _strict_text_list(document["configuration_tags"], "configuration_tags")
    config = PilotConfig(
        pilot_id=document["pilot_id"],
        results_root=document["results_root"],
        n_realizations=document["n_realizations"],
        n_agents=document["n_agents"],
        t_horizon=document["t_horizon"],
        initial_wealth=_finite_float(document["initial_wealth"], "initial_wealth"),
        snapshot_interval=document["snapshot_interval"],
        base_seed=document["base_seed"],
        conservation_tolerance=_finite_float(document["conservation_tolerance"], "conservation_tolerance"),
        nonnegative_tolerance=_finite_float(document["nonnegative_tolerance"], "nonnegative_tolerance"),
        cash_tolerance=_finite_float(document["cash_tolerance"], "cash_tolerance"),
        tax_rates=tuple(_finite_float(rate, "tax_rates") for rate in tax_rates),
    )
    if document != protocol_document(config, margins):
        raise ValueError("documento de protocolo incompatível")
    return config


def validate_pilot_protocol_document(
    document: object,
    config: PilotConfig,
    margins: StabilityMargins,
) -> None:
    persisted_config = pilot_config_from_protocol_document(document, margins)
    if persisted_config != config:
        raise ValueError("documento de protocolo incompatível")
