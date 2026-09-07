"""Capability-manifest matching and fail-closed Candidate resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Any

from ascend_fi.contracts import (
    Candidate,
    CandidateBinding,
    CandidateProof,
    CandidateRecommendation,
    CandidateStatus,
    CandidateValidation,
    Capability,
    CapabilityKind,
    ProofLevel,
    ObservedTarget,
    load_capability_manifest,
)

from .observed import ObservedFact
from .ranking import RankedCandidate, rank_candidate


def load_capabilities(directory: str | Path) -> tuple[Capability, ...]:
    """Load a manifest snapshot; this is not a live or product registry."""

    root = Path(directory)
    return tuple(load_capability_manifest(path) for path in sorted(root.glob("*.json")))


def _downstream_symbols(capability: Capability) -> tuple[str, ...]:
    downstream = capability.extra.get("downstream", {})
    if not isinstance(downstream, Mapping):
        return ()
    values = downstream.get("symbols")
    if isinstance(values, list):
        return tuple(str(item) for item in values)
    trigger = downstream.get("trigger_symbol")
    return () if trigger is None else (str(trigger),)


def _proof_level(fact: ObservedFact) -> ProofLevel:
    if fact.proof.get("semantic_slot_proven") is True:
        return ProofLevel.SEMANTIC_SLOT_PROVEN
    if fact.proof.get("x1_slot_proven") is True and fact.runtime_slots.get("x1", -1) >= 0:
        return ProofLevel.SLOT_PROVEN
    if fact.symbol and fact.execution_plane:
        return ProofLevel.STRUCTURAL_MATCH
    return ProofLevel.OBSERVED


def _candidate_id(capability_id: str, fact: ObservedFact) -> str:
    seed = f"{capability_id}|{fact.call_fingerprint}|input_activation|{fact.runtime_slots.get('x1')}"
    return "candidate-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def _compatibility_reasons(capability: Capability, fact: ObservedFact) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    rejected: list[str] = []
    if capability.execution_plane is fact.execution_plane:
        accepted.append(f"execution_plane={fact.execution_plane.value}")
    else:
        rejected.append("execution plane mismatch")
    symbols = _downstream_symbols(capability)
    if fact.symbol in symbols:
        accepted.append(f"symbol={fact.symbol}")
    else:
        rejected.append("symbol not declared by capability")
    models = capability.compatibility.get("models")
    model = fact.environment.get("model")
    if isinstance(models, list) and model is not None:
        if model in models:
            accepted.append(f"model={model}")
        else:
            rejected.append(f"model {model} is outside compatibility")
    if capability.id.endswith("residual_stomp_proxy"):
        if not fact.producer_metadata.get("live_residual_relation"):
            rejected.append("live residual relation not observed")
        if not fact.producer_metadata.get("allocation_boundary_relation"):
            rejected.append("allocation boundary relation not observed")
    return accepted, rejected


def resolve_candidate(fact: ObservedFact, capability: Capability) -> Candidate | None:
    accepted, rejected = _compatibility_reasons(capability, fact)
    if "symbol not declared by capability" in rejected or "execution plane mismatch" in rejected:
        return None
    proof_level = _proof_level(fact)
    role = fact.tensor_roles.get("x1", {})
    slot = fact.runtime_slots.get("x1")
    if capability.kind is CapabilityKind.INJECTOR and capability.supports.lossless_validate:
        if role.get("semantic_role") != "input_activation":
            rejected.append("x1 input_activation role is not proven")
        if proof_level not in {
            ProofLevel.SEMANTIC_SLOT_PROVEN,
            ProofLevel.LOSSLESS_VALIDATED,
            ProofLevel.E2E_VALIDATED,
        }:
            rejected.append("semantic runtime slot proof is insufficient")
    status = CandidateStatus.ACCEPTED if not rejected else CandidateStatus.REJECTED
    reason = "; ".join(rejected) if rejected else "; ".join(accepted)
    observed = ObservedTarget(
        execution_plane=fact.execution_plane,
        symbol=fact.symbol,
        rank=fact.rank,
        phase=fact.phase,
        shape=fact.shape,
        dtype=fact.dtype,
        stream=fact.stream,
        call_fingerprint=fact.call_fingerprint,
        extra={
            "call_index": fact.call_index,
            "environment": fact.environment,
            "runtime_proof": fact.proof,
        },
    )
    recommendation = "lossless_substitution" if capability.supports.lossless_validate else "observe"
    return Candidate(
        candidate_id=_candidate_id(capability.id, fact),
        capability_id=capability.id,
        observed=observed,
        binding=CandidateBinding(
            tensor_role=role.get("semantic_role"),
            runtime_slot=slot,
            storage_role="x1",
            extra={
                "formal_ordinal": role.get("formal_ordinal"),
                "format": role.get("format"),
                "strides": role.get("strides"),
                "contiguous": role.get("contiguous"),
                "view_offset": role.get("view_offset"),
                "view_bytes": role.get("view_bytes"),
                "storage_bytes": role.get("storage_bytes"),
                "address_metadata": fact.address_metadata,
            },
        ),
        proof=CandidateProof(level=proof_level, evidence_refs=fact.evidence_refs, extra={"matching_reasons": accepted}),
        validation=CandidateValidation(status=status, reason=reason),
        recommended=CandidateRecommendation(
            strategy=recommendation,
            params={
                "symbol": fact.symbol,
                "phase": fact.phase,
                "shape": list(fact.shape) if fact.shape is not None else None,
                "dtype": fact.dtype,
                "tensor_role": role.get("semantic_role"),
                "runtime_slot": slot,
                "proof_mode": proof_level.value,
            },
        ),
        extra={"matching_reasons": accepted, "rejection_reasons": rejected},
    )


def resolve_and_rank(fact: ObservedFact, capabilities: Iterable[Capability]) -> tuple[RankedCandidate, ...]:
    by_id = {capability.id: capability for capability in capabilities}
    candidates = [candidate for capability in by_id.values() if (candidate := resolve_candidate(fact, capability))]
    ranked = [rank_candidate(candidate, by_id[candidate.capability_id]) for candidate in candidates]
    return tuple(sorted(ranked, key=lambda item: (-item.score, item.candidate.capability_id)))
