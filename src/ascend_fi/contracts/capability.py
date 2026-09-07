"""Capability manifest contract for reusable Ascend FI implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ._base import JsonContract, as_mapping, enum_value, extra_fields, require


class CapabilityKind(str, Enum):
    INJECTOR = "injector"
    OBSERVER = "observer"
    DISCOVERY = "discovery"
    PROXY = "proxy"


class CapabilityMaturity(str, Enum):
    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    PROBE = "probe"


class ExecutionPlane(str, Enum):
    TORCH = "torch"
    VLLM_BACKEND = "vllm_backend"
    TORCH_NPU = "torch_npu"
    OP_API = "op_api"
    ATB = "atb"
    HCCL = "hccl"
    STATE = "state"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class SemanticMapping(JsonContract):
    candidate_positions: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticMapping":
        known = {"candidate_positions"}
        positions = require(data, "candidate_positions")
        if not isinstance(positions, list) or not all(isinstance(item, str) for item in positions):
            raise ValueError("semantic_mapping.candidate_positions must be a string list")
        return cls(tuple(positions), extra_fields(data, known))


@dataclass(frozen=True)
class CapabilitySupport(JsonContract):
    observe: bool
    lossless_validate: bool
    inject: bool
    restore: bool
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilitySupport":
        known = {"observe", "lossless_validate", "inject", "restore"}
        values = {key: require(data, key) for key in known}
        if not all(isinstance(value, bool) for value in values.values()):
            raise ValueError("all supports fields must be booleans")
        return cls(
            observe=values["observe"],
            lossless_validate=values["lossless_validate"],
            inject=values["inject"],
            restore=values["restore"],
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class CanonicalReferences(JsonContract):
    source: tuple[str, ...]
    build: tuple[str, ...]
    evidence: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalReferences":
        known = {"source", "build", "evidence"}
        result: dict[str, tuple[str, ...]] = {}
        for key in known:
            values = require(data, key)
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError(f"canonical.{key} must be a string list")
            result[key] = tuple(values)
        return cls(
            source=result["source"],
            build=result["build"],
            evidence=result["evidence"],
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class Capability(JsonContract):
    schema_version: int
    id: str
    kind: CapabilityKind
    maturity: CapabilityMaturity
    execution_plane: ExecutionPlane
    mechanism: str
    semantic_mapping: SemanticMapping
    supports: CapabilitySupport
    canonical: CanonicalReferences
    compatibility: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Capability":
        known = {
            "schema_version",
            "id",
            "kind",
            "maturity",
            "execution_plane",
            "mechanism",
            "semantic_mapping",
            "supports",
            "canonical",
            "compatibility",
        }
        schema_version = require(data, "schema_version")
        if not isinstance(schema_version, int):
            raise ValueError("schema_version must be an integer")
        capability_id = require(data, "id")
        mechanism = require(data, "mechanism")
        if not isinstance(capability_id, str) or not capability_id:
            raise ValueError("id must be a non-empty string")
        if not isinstance(mechanism, str) or not mechanism:
            raise ValueError("mechanism must be a non-empty string")
        compatibility = data.get("compatibility", {})
        compatibility = dict(as_mapping(compatibility, "compatibility"))
        return cls(
            schema_version=schema_version,
            id=capability_id,
            kind=enum_value(CapabilityKind, require(data, "kind"), "kind"),
            maturity=enum_value(
                CapabilityMaturity, require(data, "maturity"), "maturity"
            ),
            execution_plane=enum_value(
                ExecutionPlane, require(data, "execution_plane"), "execution_plane"
            ),
            mechanism=mechanism,
            semantic_mapping=SemanticMapping.from_dict(
                as_mapping(require(data, "semantic_mapping"), "semantic_mapping")
            ),
            supports=CapabilitySupport.from_dict(
                as_mapping(require(data, "supports"), "supports")
            ),
            canonical=CanonicalReferences.from_dict(
                as_mapping(require(data, "canonical"), "canonical")
            ),
            compatibility=compatibility,
            extra=extra_fields(data, known),
        )


def load_capability_manifest(path: str | Path) -> Capability:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest root must be an object: {manifest_path}")
    return Capability.from_dict(data)
