"""Native execution strategy values. No framework import or device operation.

Context fields describe a real invocation supplied by its adapter/caller. In
particular request_count is never inferred from the leading tensor dimension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f'{label} must be an integer >= {minimum}')
    return value


def number(value: Any, label: str, *, minimum: float = 0) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f'{label} must be finite and >= {minimum}')
    return float(value)


def sha256(value: Any, label: str = 'SHA256') -> str:
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            f'{label} must be 64 lowercase hexadecimal characters')
    return value


def json_value(value: Any, label: str = 'value') -> Any:
    """Validate/copy JSON data without coercing NaN, arbitrary objects or keys."""
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        require(math.isfinite(value), f'{label} must not contain non-finite numbers')
        return value
    if isinstance(value, Mapping):
        require(all(type(k) is str for k in value), f'{label} keys must be strings')
        return {k: json_value(v, f'{label}.{k}') for k, v in value.items()}
    if type(value) in (list, tuple):
        return [json_value(v, label) for v in value]
    raise ValueError(f'{label} must contain only JSON values')


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(json_value(value), sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class InvocationContext:
    operator: str
    request_count: int | None = None
    token_count: int | None = None
    matrix_dims: Mapping[str, int] = field(default_factory=dict)
    seq_lens: tuple[int, ...] | None = None
    dtype: str | None = None
    layout: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(isinstance(self.operator, str) and bool(self.operator), 'operator is required')
        for key in ('request_count', 'token_count'):
            if getattr(self, key) is not None:
                integer(getattr(self, key), key)
        require(isinstance(self.matrix_dims, Mapping), 'matrix_dims must be a mapping')
        dims = dict(self.matrix_dims)
        for key, value in dims.items():
            require(isinstance(key, str) and bool(key), 'matrix dimension names must be nonempty strings')
            integer(value, f'matrix_dims.{key}')
        object.__setattr__(self, 'matrix_dims', dims)
        if self.seq_lens is not None:
            require(type(self.seq_lens) in (list, tuple), 'seq_lens must be explicit sequence lengths or None')
            object.__setattr__(self, 'seq_lens', tuple(integer(v, 'sequence length') for v in self.seq_lens))
        for key in ('dtype', 'layout'):
            require(getattr(self, key) is None or isinstance(getattr(self, key), str), f'{key} must be a string or None')
        require(isinstance(self.attributes, Mapping), 'attributes must be a mapping')
        object.__setattr__(self, 'attributes', json_value(self.attributes, 'attributes'))

    def as_dict(self) -> dict[str, Any]:
        return {'operator': self.operator, 'request_count': self.request_count, 'token_count': self.token_count,
                'matrix_dims': dict(self.matrix_dims), 'seq_lens': None if self.seq_lens is None else list(self.seq_lens),
                'dtype': self.dtype, 'layout': self.layout, 'attributes': json_value(self.attributes)}


@dataclass(frozen=True)
class NativeStrategy:
    """Adapter-defined semantic projection, not a request label or raw-buffer hash.

The adapter decides which verified native fields are strategy semantics. Raw
bytes (including padding) are separately bound by StrategyObservation.
"""
    strategy_id: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        require(isinstance(self.strategy_id, str) and bool(self.strategy_id), 'strategy_id is required')
        require(isinstance(self.parameters, Mapping), 'strategy parameters must be a mapping')
        object.__setattr__(self, 'parameters', json_value(self.parameters, 'parameters'))

    def as_dict(self) -> dict[str, Any]:
        return {'strategy_id': self.strategy_id, 'parameters': json_value(self.parameters)}

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class BuildFingerprints:
    source_sha256: str
    build_sha256: str
    certificate_sha256: str | None = None

    def __post_init__(self) -> None:
        sha256(self.source_sha256, 'source_sha256')
        sha256(self.build_sha256, 'build_sha256')
        if self.certificate_sha256 is not None:
            sha256(self.certificate_sha256, 'certificate_sha256')

    def as_dict(self) -> dict[str, str | None]:
        return {'source_sha256': self.source_sha256, 'build_sha256': self.build_sha256,
                'certificate_sha256': self.certificate_sha256}


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        require(isinstance(self.path, str) and bool(self.path), 'Evidence path is required')
        sha256(self.sha256, 'Evidence SHA256')


@dataclass(frozen=True)
class ResourceEstimate:
    """Admission estimate only; the harness enforces actual time/memory limits."""
    device_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def __post_init__(self) -> None:
        number(self.device_seconds, 'estimated device_seconds')
        integer(self.peak_memory_bytes, 'estimated peak_memory_bytes')


@dataclass(frozen=True)
class ResourceUsage:
    device_seconds: float | None = None
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.device_seconds is not None:
            number(self.device_seconds, 'actual device_seconds')
        if self.peak_memory_bytes is not None:
            integer(self.peak_memory_bytes, 'actual peak_memory_bytes')


@dataclass(frozen=True)
class StrategyObservation:
    context: InvocationContext
    strategy: NativeStrategy
    fingerprints: BuildFingerprints
    evidence: tuple[EvidenceRef, ...]
    raw_sha256: str | None = None
    resources: ResourceUsage = field(default_factory=ResourceUsage)

    def __post_init__(self) -> None:
        require(isinstance(self.context, InvocationContext) and isinstance(self.strategy, NativeStrategy), 'Invalid native observation')
        require(isinstance(self.fingerprints, BuildFingerprints), 'Observation must bind source/build fingerprints')
        require(bool(self.evidence) and all(isinstance(v, EvidenceRef) for v in self.evidence), 'Native observation needs evidence references')
        object.__setattr__(self, 'evidence', tuple(self.evidence))
        if self.raw_sha256 is not None:
            sha256(self.raw_sha256, 'raw_sha256')
        require(isinstance(self.resources, ResourceUsage), 'Invalid resource usage')


@dataclass(frozen=True)
class Assessment:
    """Independent outcomes. None means unknown/unmeasured, never False/True."""
    strategy_switched: bool | None = None
    numerics_observed: bool | None = None
    model_impact: bool | None = None

    def __post_init__(self) -> None:
        for value in (self.strategy_switched, self.numerics_observed, self.model_impact):
            require(value is None or type(value) is bool, 'Assessment values must be bool or None')

    def as_dict(self) -> dict[str, bool | None]:
        return {'strategy_switched': self.strategy_switched, 'numerics_observed': self.numerics_observed,
                'model_impact': self.model_impact}


def assess_strategy_switch(before: StrategyObservation, after: StrategyObservation) -> Assessment:
    """Only assess a same-context, same-build native observation pair."""
    require(before.context.as_dict() == after.context.as_dict(), 'Strategy switch pair must have the same actual context')
    require(before.fingerprints == after.fingerprints, 'Strategy switch pair must have the same source/build/certificate')
    return Assessment(strategy_switched=before.strategy.semantic_sha256 != after.strategy.semantic_sha256)
