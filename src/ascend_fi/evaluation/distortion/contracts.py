"""Small JSON contracts extending EvaluationResult through its extra fields."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from ...contracts._base import JsonContract, extra_fields

V1_FROZEN_SHA256 = "abfc3c9a26597e3eefc7204306553dd381353c45dfe5b3d135632bbd8856d502"


def content_hash(data: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in data.items() if k != "hash"}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FaultConfigurationRecord(JsonContract):
    capability: str | None = None
    strategy: str | None = None
    module_families: tuple[str, ...] = ()
    actual_module_names: tuple[str, ...] = ()
    targeted_layers: tuple[int, ...] = ()
    available_layers: int | None = None
    targeted_ranks: tuple[int, ...] = ()
    world_size: int | None = None
    strength: dict[str, Any] = field(default_factory=dict)
    hit_policy: dict[str, Any] = field(default_factory=dict)
    persistence: dict[str, Any] = field(default_factory=dict)
    runtime_hash: str | None = None
    config_hash: str | None = None
    model_fingerprint: Any = None
    topology: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FaultConfigurationRecord":
        known = {f.name for f in fields(cls) if f.name != "extra"}
        values = {k: data[k] for k in known if k in data}
        for key in ("module_families", "actual_module_names", "targeted_layers", "targeted_ranks"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values, extra=extra_fields(data, known))


@dataclass(frozen=True)
class DistortionProbe(JsonContract):
    probe_id: str
    category: str
    prompt: str
    generation_config: dict[str, Any]
    expected: dict[str, Any]
    applicable_detectors: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistortionProbe":
        known = {f.name for f in fields(cls) if f.name != "extra"}
        return cls(str(data["probe_id"]), str(data["category"]), str(data["prompt"]),
                   dict(data["generation_config"]), dict(data.get("expected", {})),
                   tuple(data["applicable_detectors"]), extra_fields(data, known))


@dataclass(frozen=True)
class DetectorResult(JsonContract):
    detector: str
    triggered: bool | None
    baseline_value: Any
    fault_value: Any
    severity: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: str = "DETERMINISTIC_RULE"
    source: str = "recorded_response"
    phenotype: str | None = None
    baseline_failed: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectorResult":
        known = {f.name for f in fields(cls) if f.name != "extra"}
        return cls(**{k: data[k] for k in known if k in data}, extra=extra_fields(data, known))


def load_suite(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("hash") != content_hash(data) or data.get("hash") != V1_FROZEN_SHA256:
        raise ValueError("probe suite hash mismatch: frozen probes cannot be retuned during evaluation")
    probes = [DistortionProbe.from_dict(p) for p in data["probes"]]
    if len(probes) != 16 or len({p.probe_id for p in probes}) != 16:
        raise ValueError("DISTORTION_PROBE_SUITE_V1 requires 16 unique probes")
    if data.get("probe_count") != 16 or data.get("version") != "DISTORTION_PROBE_SUITE_V1":
        raise ValueError("unsupported probe suite")
    from .detectors import DETECTORS
    if any(set(p.applicable_detectors) - set(DETECTORS) for p in probes):
        raise ValueError("unknown detector in suite")
    return data
