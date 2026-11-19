"""Wire bounds for ordinary task evidence, independent of Host history size.

An overflow is a failed/incomplete report, never a truncated success. Preserve
owned-resource control state so abort/clean/unregister can still release it.
Private provider observations are not modified by this wire projection.
"""
import hashlib
import json

MAX_EVIDENCE_BYTES=512*1024
MAX_PROVIDER_STATE_BYTES=2*1024*1024
_CONTROL=('execution_boundary','state_source','hook_state_available','hook_owned',
    'owner','generation','native_loaded','native_registered','registered','enabled',
    'inflight','leases','owned_hooks','replay_pending','replay_failed','recovery_validation_pending',
    'ordinary_task','mechanism','simulation','current_request_phase','deployment_acceptance_use')


def encoded(value):
    return json.dumps(value,sort_keys=True,allow_nan=False).encode('utf-8')


def bound_state(state,*,ordinary=False):
    if not ordinary and state.get('ordinary_task') is not True:return state
    evidence=state.get('evidence',state.get('serving',{}))
    evidence_bytes=encoded(evidence);state_bytes=encoded(state)
    if len(evidence_bytes)<=MAX_EVIDENCE_BYTES and len(state_bytes)<=MAX_PROVIDER_STATE_BYTES-512:
        return dict(state,evidence_complete=True,evidence_limits=dict(
            evidence_bytes=len(evidence_bytes),provider_state_bytes_before_limits=len(state_bytes),
            max_evidence_bytes=MAX_EVIDENCE_BYTES,max_provider_state_bytes=MAX_PROVIDER_STATE_BYTES))
    result={k:state[k] for k in _CONTROL if k in state}
    diagnostic=dict(status='ORDINARY_EVIDENCE_LIMIT_EXCEEDED',complete=False,
        matched_off_and_health_confirmed=False,qualification_granted=False,
        evidence_bytes=len(evidence_bytes),evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        provider_state_bytes=len(state_bytes),max_evidence_bytes=MAX_EVIDENCE_BYTES,
        max_provider_state_bytes=MAX_PROVIDER_STATE_BYTES,
        payload_omitted=True,original_provider_observations_retained=True)
    result.update(evidence=diagnostic,evidence_complete=False,evidence_limit_error=True)
    return result
