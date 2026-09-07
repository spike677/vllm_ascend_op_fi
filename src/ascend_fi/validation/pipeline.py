"""Fail-closed lossless validation of a resolved Candidate."""

from __future__ import annotations

from ascend_fi.contracts import (
    Candidate,
    CandidateStatus,
    ProofLevel,
    ValidationResult,
    ValidationStatus,
)

from .evidence import LosslessEvidence, LosslessExecutor


_VALIDATABLE_PROOFS = {
    ProofLevel.SEMANTIC_SLOT_PROVEN,
    ProofLevel.LOSSLESS_VALIDATED,
    ProofLevel.E2E_VALIDATED,
}


def _rejected(candidate: Candidate, reason: str) -> ValidationResult:
    return ValidationResult(
        candidate_id=candidate.candidate_id,
        resolved=False,
        lossless_attempted=False,
        lossless_passed=False,
        status=ValidationStatus.REJECTED,
        reason=reason,
    )


def validate_target(candidate: Candidate, executor: LosslessExecutor) -> ValidationResult:
    """Run only the lossless gate; this function never requests a destructive mode."""

    if candidate.validation.status is not CandidateStatus.ACCEPTED:
        return _rejected(candidate, candidate.validation.reason or "candidate is not accepted")
    if candidate.proof.level not in _VALIDATABLE_PROOFS:
        return _rejected(candidate, "candidate proof is below SEMANTIC_SLOT_PROVEN")
    if candidate.recommended.strategy != "lossless_substitution":
        return _rejected(candidate, "candidate does not recommend lossless_substitution")
    try:
        evidence = executor.execute_lossless(candidate)
    except Exception as error:
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            resolved=False,
            lossless_attempted=True,
            lossless_passed=False,
            status=ValidationStatus.FAILED,
            reason=f"lossless executor failed: {error}",
        )
    restore_exact = _restore_is_exact(evidence)
    statuses_ok = all(
        status == 0
        for status in (evidence.copy_status, evidence.run_status, evidence.sync_status)
    )
    passed = (
        evidence.target_resolved
        and evidence.expected_api_slot_hit
        and evidence.substitution_executed
        and evidence.hit_count > 0
        and statuses_ok
        and evidence.temporary_freed is True
        and evidence.model_output_equal is True
        and restore_exact
    )
    reason = (
        "lossless substitution, model equivalence, and exact recovery proven"
        if passed
        else "lossless gate evidence is incomplete or inconsistent"
    )
    return ValidationResult(
        candidate_id=candidate.candidate_id,
        resolved=evidence.target_resolved,
        lossless_attempted=True,
        lossless_passed=passed,
        hit_count=evidence.hit_count,
        reject_count=evidence.reject_count,
        reject_reasons=evidence.reject_reasons,
        before_hash=evidence.before_hash,
        substituted_hash=evidence.substituted_hash,
        restored_hash=evidence.restored_hash,
        model_output_equal=evidence.model_output_equal,
        restore_exact=restore_exact,
        status=(
            ValidationStatus.LOSSLESS_VALIDATED if passed else ValidationStatus.FAILED
        ),
        reason=reason,
        evidence_refs=evidence.evidence_refs,
        extra={
            "expected_api_slot_hit": evidence.expected_api_slot_hit,
            "substitution_executed": evidence.substitution_executed,
            "original_storage_untouched": evidence.original_storage_untouched,
            "copy_status": evidence.copy_status,
            "run_status": evidence.run_status,
            "sync_status": evidence.sync_status,
            "temporary_freed": evidence.temporary_freed,
        },
    )


def _restore_is_exact(evidence: LosslessEvidence) -> bool:
    if evidence.before_hash is not None or evidence.restored_hash is not None:
        return (
            evidence.before_hash is not None
            and evidence.restored_hash is not None
            and evidence.before_hash == evidence.restored_hash
        )
    # The existing QMM lossless backend does not overwrite the original x1:
    # it binds a distinct temporary for the real Run and frees it after sync.
    return evidence.original_storage_untouched is True and evidence.temporary_freed is True
