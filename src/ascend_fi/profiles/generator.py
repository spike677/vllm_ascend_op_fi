"""Generate a portable model capability profile after validated E2E evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ascend_fi.contracts import (
    Candidate,
    Capability,
    EvaluationResult,
    EvaluationStatus,
    ValidationResult,
)


@dataclass(frozen=True)
class ModelCapabilityProfile:
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.data, ensure_ascii=False, indent=indent, sort_keys=True)

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _portable_ref(reference: str, repository_root: Path | None) -> str:
    path = Path(reference)
    if not path.is_absolute() or repository_root is None:
        return reference
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return reference


def generate_model_profile(
    capability: Capability,
    candidate: Candidate,
    validation: ValidationResult,
    evaluation: EvaluationResult,
    *,
    repository_root: str | Path | None = None,
) -> ModelCapabilityProfile:
    if candidate.capability_id != capability.id:
        raise ValueError("candidate/capability id mismatch")
    if not validation.destructive_injection_allowed:
        raise ValueError("profile generation requires a passed lossless gate")
    if evaluation.status is not EvaluationStatus.PASS:
        raise ValueError("profile generation requires a passed evaluation")
    if not evaluation.mechanism.hit_count or evaluation.mechanism.hit_count <= 0:
        raise ValueError("profile generation requires a recorded fault hit")
    environment = dict(candidate.observed.extra.get("environment", {}))
    stable_identity = {
        "model": environment.get("model"),
        "topology": environment.get("topology"),
        "symbol": candidate.observed.symbol,
        "shape": list(candidate.observed.shape or ()),
        "dtype": candidate.observed.dtype,
        "phase": candidate.observed.phase,
        "tensor_role": candidate.binding.tensor_role,
    }
    fingerprint = hashlib.sha256(
        json.dumps(stable_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    root = None if repository_root is None else Path(repository_root)
    refs = _unique(
        _portable_ref(ref, root)
        for ref in (
            candidate.proof.evidence_refs
            + validation.evidence_refs
            + evaluation.evidence_refs
        )
    )
    profile_id = f"model-capability-{fingerprint[:16]}-{capability.id.replace('.', '-')}"
    return ModelCapabilityProfile(
        {
            "schema_version": 0,
            "profile_id": profile_id,
            "model_fingerprint": fingerprint,
            "model_identity": stable_identity,
            "runtime_environment": environment,
            "capability": {
                "id": capability.id,
                "execution_plane": capability.execution_plane.value,
                "mechanism": capability.mechanism,
                "maturity": capability.maturity.value,
            },
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "binding": {
                    "tensor_role": candidate.binding.tensor_role,
                    "runtime_slot": candidate.binding.runtime_slot,
                    "storage_role": candidate.binding.storage_role,
                    "stability": "rediscover_per_environment",
                },
                "proof_level": candidate.proof.level.value,
                "recommended_strategy": candidate.recommended.strategy,
            },
            "validation": {
                "status": validation.status.value,
                "lossless_passed": validation.lossless_passed,
                "model_output_equal": validation.model_output_equal,
                "restore_exact": validation.restore_exact,
                "hit_count": validation.hit_count,
                "reject_count": validation.reject_count,
                "before_hash": validation.before_hash,
                "substituted_hash": validation.substituted_hash,
                "restored_hash": validation.restored_hash,
            },
            "evaluation": {
                "status": evaluation.status.value,
                "hit_count": evaluation.mechanism.hit_count,
                "response_equal": evaluation.model_effect.response_equal,
                "token_equal": evaluation.model_effect.token_equal,
                "first_divergent_token": evaluation.model_effect.first_divergent_token,
                "recovery_exact": evaluation.extra.get("recovery_exact"),
            },
            "known_limitations": list(capability.extra.get("limitations", [])),
            "evidence_refs": refs,
        }
    )
