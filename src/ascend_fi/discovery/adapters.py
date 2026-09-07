"""Adapters from existing native JSONL traces to :class:`ObservedFact`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ascend_fi.contracts import ExecutionPlane

from .observed import ObservedFact


_ACL_DTYPES = {0: "float32", 1: "float16", 2: "int8", 27: "bf16"}
_KNOWN_SYMBOLS = (
    "aclnnQuantMatmulWeightNz",
    "aclnnMatmulWeightNz",
    "aclnnMatmul",
)


@dataclass(frozen=True)
class TraceContext:
    """Stable request/environment context absent from the native Run event."""

    phase: str | None = None
    rank: int | None = None
    model: str | None = None
    topology: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    evidence_ref: str | None = None


def _symbol_from_api_info(api_info: str) -> str | None:
    for symbol in _KNOWN_SYMBOLS:
        if symbol in api_info:
            return symbol
    return None


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def adapt_qmm_run_trace(
    record: Mapping[str, Any], context: TraceContext | None = None
) -> ObservedFact | None:
    """Convert one existing ``qmm_run_trace`` event.

    Non-QMM records return ``None`` so a mixed runtime JSONL can be streamed.
    Malformed QMM records raise ``ValueError`` rather than fabricating proof.
    """

    if record.get("event") != "qmm_run_trace":
        return None
    context = context or TraceContext()
    api_info = record.get("api_info")
    if not isinstance(api_info, str):
        raise ValueError("qmm_run_trace.api_info must be a string")
    symbol = _symbol_from_api_info(api_info)
    if symbol != "aclnnQuantMatmulWeightNz":
        return None
    x1 = record.get("x1")
    slots = record.get("storage_order_slots")
    proof = record.get("proof")
    if not isinstance(x1, Mapping) or not isinstance(slots, Mapping) or not isinstance(proof, Mapping):
        raise ValueError("qmm_run_trace requires x1, storage_order_slots, and proof objects")
    raw_shape = x1.get("shape")
    if not isinstance(raw_shape, list) or not raw_shape or not all(
        isinstance(item, int) for item in raw_shape
    ):
        raise ValueError("qmm_run_trace.x1.shape must be a non-empty integer list")
    dtype_raw = x1.get("dtype")
    dtype = _ACL_DTYPES.get(dtype_raw, f"acl_dtype_{dtype_raw}")
    run_slot = slots.get("x1")
    if not isinstance(run_slot, int):
        raise ValueError("qmm_run_trace.storage_order_slots.x1 must be an integer")
    formal_ordinal = record.get("x1_param_ordinal")
    semantic_slot_proven = (
        formal_ordinal == 0
        and run_slot == 0
        and all(
            proof.get(name) is True
            for name in (
                "param_order_exact",
                "address_count_exact",
                "empty_slots_are_null",
                "nonempty_slots_are_nonnull",
                "no_shared_storage",
                "x1_storage_unique",
                "x1_slot_proven",
            )
        )
    )
    normalized_proof = dict(proof)
    normalized_proof["semantic_slot_proven"] = semantic_slot_proven
    environment = dict(context.runtime)
    if context.model is not None:
        environment["model"] = context.model
    if context.topology is not None:
        environment["topology"] = context.topology
    fingerprint_payload = {
        "execution_plane": ExecutionPlane.OP_API.value,
        "symbol": symbol,
        "phase": context.phase,
        "shape": raw_shape,
        "dtype": dtype,
        "tensor_role": "input_activation",
        "runtime_slot": run_slot,
        "proof": normalized_proof,
    }
    refs = () if context.evidence_ref is None else (context.evidence_ref,)
    return ObservedFact(
        execution_plane=ExecutionPlane.OP_API,
        symbol=symbol,
        rank=context.rank if context.rank is not None else record.get("rank", record.get("device")),
        phase=context.phase if context.phase is not None else record.get("phase"),
        call_index=record.get("run_index"),
        call_fingerprint=_fingerprint(fingerprint_payload),
        stream=record.get("stream"),
        shape=tuple(raw_shape),
        dtype=dtype,
        runtime_slots={str(key): int(value) for key, value in slots.items() if isinstance(value, int)},
        tensor_roles={
            "x1": {
                "semantic_role": "input_activation",
                "formal_ordinal": formal_ordinal,
                "runtime_slot": run_slot,
                "format": x1.get("format"),
                "strides": x1.get("strides"),
                "contiguous": x1.get("contiguous"),
                "view_offset": x1.get("view_offset"),
                "view_bytes": x1.get("view_bytes"),
                "storage_bytes": x1.get("storage_bytes"),
            }
        },
        proof=normalized_proof,
        address_metadata={
            "runtime_address": record.get("x1_runtime_address"),
            "raw_addr": record.get("x1_raw_addr"),
            "storage_addr": record.get("x1_storage_addr"),
            "data_addr": record.get("x1_data_addr"),
            "direct_gws_address_match": (
                record.get("direct_gws_address_match", {}).get("x1")
                if isinstance(record.get("direct_gws_address_match"), Mapping)
                else None
            ),
        },
        environment=environment,
        evidence_refs=refs,
    )


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        yield value


def adapt_qmm_jsonl(path: str | Path, context: TraceContext | None = None) -> list[ObservedFact]:
    return [fact for record in read_jsonl(path) if (fact := adapt_qmm_run_trace(record, context))]
