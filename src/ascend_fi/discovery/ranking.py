"""Deterministic candidate ranking for validation eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ascend_fi.contracts import Candidate, CandidateStatus, Capability, CapabilityMaturity, ProofLevel


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNSAFE_UNKNOWN = "UNSAFE/UNKNOWN"


_PROOF_SCORE = {
    ProofLevel.OBSERVED: 5,
    ProofLevel.STRUCTURAL_MATCH: 20,
    ProofLevel.SLOT_PROVEN: 40,
    ProofLevel.SEMANTIC_SLOT_PROVEN: 60,
    ProofLevel.LOSSLESS_VALIDATED: 80,
    ProofLevel.E2E_VALIDATED: 100,
}


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    score: int
    confidence: Confidence
    auto_validation_allowed: bool
    reasons: tuple[str, ...]


def rank_candidate(candidate: Candidate, capability: Capability) -> RankedCandidate:
    if candidate.validation.status is CandidateStatus.REJECTED:
        return RankedCandidate(candidate, 0, Confidence.UNSAFE_UNKNOWN, False, (candidate.validation.reason or "candidate rejected",))
    score = _PROOF_SCORE[candidate.proof.level]
    reasons = [f"proof={candidate.proof.level.value}"]
    if capability.maturity is CapabilityMaturity.VALIDATED:
        score += 15
        reasons.append("capability maturity validated")
    if candidate.observed.shape is not None:
        score += 5
        reasons.append("shape observed")
    if candidate.observed.dtype is not None:
        score += 5
        reasons.append("dtype observed")
    if candidate.observed.phase is not None:
        score += 5
        reasons.append("phase observed")
    if candidate.binding.runtime_slot is not None:
        score += 5
        reasons.append("runtime slot bound")
    if capability.supports.lossless_validate:
        score += 5
        reasons.append("lossless validation supported")
    if capability.supports.restore:
        score += 5
        reasons.append("restore supported")
    score = min(score, 100)
    confidence = Confidence.HIGH if score >= 90 else Confidence.MEDIUM if score >= 65 else Confidence.LOW
    allowed = (
        candidate.proof.level in {ProofLevel.SEMANTIC_SLOT_PROVEN, ProofLevel.LOSSLESS_VALIDATED, ProofLevel.E2E_VALIDATED}
        and capability.supports.lossless_validate
        and candidate.binding.runtime_slot is not None
    )
    return RankedCandidate(candidate, score, confidence, allowed, tuple(reasons))
