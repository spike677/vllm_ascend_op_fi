"""Offline consistency audit of saved coverage; never runtime qualification.

The call ledger is checked independently of its saved counters and boolean
verdict. Referenced artifacts must exist with their exact byte hashes. Neither
this consistency check nor a supplied artifact authenticates an observer's
claim about the real model universe or proves Native execution semantics.
Those remain the integrator's separate source/runtime review.
"""
import hashlib
import json
from pathlib import Path
import re


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def index_artifacts(paths):
    """Hash explicit evidence paths; no implicit search of tensor directories."""
    result = {}
    for path in paths:
        path = Path(path).resolve(strict=True)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result.setdefault(digest.hexdigest(), []).append(dict(path=str(path), bytes=path.stat().st_size))
    return result


def _worker(target):
    prefix = [target.get('process_id'), target.get('process_start_identity'), target.get('worker_id'), target.get('rank')]
    return prefix + ([target.get('device_type')] if 'boundary' in target else []) + [target.get('logical_device')]


def audit_saved_coverage(report, *, artifact_index=None, c2_report=None):
    """Recompute coverage/recovery joins; do not manufacture a live C2 proof."""
    errors, references = [], set()
    def check(condition, name):
        if not condition:
            errors.append(name)
    def reference(value, name):
        check(_valid_sha(value), 'INVALID_ARTIFACT_SHA:' + name)
        if _valid_sha(value):
            references.add(value)
    scope = report.get('scope', {})
    discovery, observation = report.get('discovery') or {}, report.get('observation') or {}
    recovery = report.get('recovery') or {}
    check(report.get('schema') == 'demoncat.coverage_session.v1', 'UNSUPPORTED_COVERAGE_SCHEMA')
    check(report.get('public_qualification_granted') is False, 'SAVED_REPORT_CANNOT_GRANT_QUALIFICATION')
    check(report.get('selection') == 'all_eligible', 'ALL_ELIGIBLE_REQUIRED')
    check(report.get('trigger') in ('once', 'every_match'), 'UNKNOWN_TRIGGER')
    check(report.get('contract_satisfied') is True and report.get('rejection_reasons') == [], 'CONTRACT_NOT_SATISFIED')
    check(discovery.get('complete') is True and discovery.get('source') in (
        'independent_native_trace', 'independent_workload_observer'), 'INDEPENDENT_DISCOVERY_INCOMPLETE')
    check(discovery.get('workload_sha256') == scope.get('workload_sha256'), 'DISCOVERY_WORKLOAD_MISMATCH')
    for name in ('workload_sha256', 'environment_sha256', 'implementation_sha256'):
        check(_valid_sha(scope.get(name)), 'MISSING_SCOPE:' + name)
    check(scope.get('evidence_kind') in ('natural_worker', 'natural_pretrained_model', 'constructed_npu', 'cpu_contract'), 'INVALID_EVIDENCE_KIND')
    if scope.get('evidence_kind') == 'natural_pretrained_model':
        check(all(_valid_sha(scope.get(k)) for k in ('model_weights_sha256', 'model_source_sha256')), 'PRETRAINED_MODEL_IDENTITY_MISSING')
    check(report.get('real_worker_evidence') == (scope.get('evidence_kind') == 'natural_worker'), 'WORKER_SCOPE_LABEL_MISMATCH')
    check(observation.get('complete') is True and observation.get('dropped_calls') == 0, 'INDEPENDENT_OBSERVATION_INCOMPLETE')
    reference(discovery.get('artifact_sha256'), 'discovery')
    reference(observation.get('artifact_sha256'), 'observation')
    check(recovery.get('complete') is True and recovery.get('state') == 'DISABLED_CONFIRMED', 'GROUP_RECOVERY_INCOMPLETE')
    check(recovery.get('owner') == report.get('owner') and recovery.get('generation') == report.get('generation'), 'GROUP_RECOVERY_OWNER_MISMATCH')
    check(type(report.get('generation')) is int and report['generation'] > 0 and bool(report.get('owner')), 'INVALID_OWNER_GENERATION')
    check(type(recovery.get('inflight')) is int and recovery['inflight'] == 0, 'GROUP_INFLIGHT_NOT_ZERO')
    for key in ('held_leases', 'pending_drains', 'pending_leases', 'errors'):
        check(recovery.get(key) == [], 'GROUP_RECOVERY_' + key.upper())
    leases = {}
    for row in recovery.get('targets', []):
        lease = row.get('lease', {}); target = lease.get('target', {}); identity = _sha(target)
        check(identity not in leases, 'DUPLICATE_RECOVERED_TARGET')
        leases[identity] = lease
        check(all(row.get(key) is True for key in ('drained', 'released', 'off_exact'))
              and row.get('pending_reason') is None, 'TARGET_RECOVERY_INCOMPLETE')
        check(lease.get('owner') == report.get('owner') and lease.get('generation') == report.get('generation'), 'TARGET_OWNER_MISMATCH')
        check(bool(target.get('process_start_identity')), 'TARGET_LIFETIME_MISSING')
        check(_valid_sha(row.get('off_output_sha256')), 'TARGET_OFF_HASH_MISSING')
        reference(row.get('proof_sha256'), 'target_recovery')
        if scope.get('evidence_kind') == 'natural_pretrained_model':
            check(target.get('scope_kind') == 'MODEL_LOCAL' and target.get('device_type') == 'npu', 'MODEL_LOCAL_NPU_SCOPE_MISMATCH')
        if scope.get('evidence_kind') == 'natural_worker' and 'boundary' in target:
            check(target.get('scope_kind') == 'INITIALIZED_WORKER' and target.get('device_type') == 'npu', 'NATURAL_WORKER_SCOPE_MISMATCH')
    check(bool(leases) and report.get('planned_targets') == len(leases), 'RECOVERED_TARGET_DENOMINATOR_MISMATCH')
    workers = {_sha(_worker(lease['target'])) for lease in leases.values()}
    expected_workers = discovery.get('expected_workers', [])
    check(workers == {_sha(w) for w in expected_workers}
          and len(workers) == len(expected_workers)
          and workers == {_sha(w) for w in discovery.get('observed_workers', [])}, 'WORKER_DENOMINATOR_MISMATCH')
    health_workers = set()
    for row in recovery.get('workers', []):
        identity = _sha(row.get('worker')); check(identity not in health_workers, 'DUPLICATE_HEALTH_WORKER'); health_workers.add(identity)
        check(row.get('owner') == report.get('owner') and row.get('generation') == report.get('generation'), 'HEALTH_OWNER_MISMATCH')
        check(_valid_sha(row.get('baseline_output_sha256')) and row.get('baseline_output_sha256') == row.get('off_output_sha256'), 'SAME_PROCESS_HEALTH_OUTPUT_CHANGED')
        check(all(type(row.get(k)) is int and row[k] == 0 for k in
            ('active_leases', 'device_bytes', 'host_bytes', 'inflight_calls', 'owned_hooks')), 'WORKER_RESOURCES_NOT_ZERO')
        reference(row.get('proof_sha256'), 'worker_health')
    check(health_workers == workers, 'WORKER_HEALTH_DENOMINATOR_MISMATCH')
    calls = report.get('calls', []); keys = set(); tickets = []; hits = {k:0 for k in leases}; eligible_count = 0; hit_count = 0
    for sequence, row in enumerate(calls):
        decision, receipt = row.get('decision') or {}, row.get('receipt') or {}
        ticket = decision.get('ticket') or {}; call = ticket.get('call') or {}; target = call.get('target') or {}
        tickets.append(ticket)
        identity = _sha(target); key = (_sha(row.get('worker')), row.get('call_id'))
        check(key not in keys, 'DUPLICATE_OBSERVED_CALL'); keys.add(key)
        check(ticket.get('sequence') == sequence and ticket.get('observer_id') == discovery.get('observer_id'), 'OBSERVATION_SEQUENCE_OR_IDENTITY_MISMATCH')
        check(row.get('target_id') == identity and row.get('worker') == _worker(target), 'CALL_TARGET_OR_WORKER_MISMATCH')
        for public, internal in (('call_id','call_id'), ('forward_id','forward_context'), ('request_ids','request_ids'), ('phase','phase')):
            check(row.get(public) == call.get(internal), 'CALL_JOIN_MISMATCH:' + public)
        check(row.get('operator') == target.get('operator_api') and row.get('model_instance') == target.get('model_instance_id')
              and row.get('rank') == target.get('rank') and row.get('device') == target.get('logical_device'), 'CALL_METADATA_MISMATCH')
        check(row.get('owner') == report.get('owner') and row.get('generation') == report.get('generation'), 'CALL_OWNER_MISMATCH')
        check(receipt.get('decision') == decision and decision.get('session_id') == report.get('session_id'), 'RECEIPT_DECISION_MISMATCH')
        check(type(receipt.get('original_calls')) is int and receipt['original_calls'] == 1, 'ORIGINAL_CALL_COUNT_MISMATCH')
        check(row.get('eligible') is ticket.get('eligible') and type(row.get('eligible')) is bool
              and row.get('propagation_added') is ticket.get('propagation_added'), 'CALL_CLASSIFICATION_MISMATCH')
        eligible = row.get('eligible') is True and (not row.get('propagation_added') or identity in leases) and row.get('phase') != 'OFF_RECOVERY'
        if eligible:
            eligible_count += 1
            check(identity in leases and bool(row.get('request_ids')), 'ELIGIBLE_CALL_TARGET_OR_REQUEST_UNPROVEN')
        if row.get('propagation_added'):
            check(row.get('phase') == 'POST_FIRST_ANOMALY' and identity not in leases, 'INVALID_PROPAGATION_ADDITION')
        if receipt.get('outcome') == 'HIT':
            hit_count += 1
            check(eligible and decision.get('admitted') is True and receipt.get('native_applied') is True, 'NON_TARGET_OR_UNADMITTED_HIT')
            check(decision.get('lease') == leases.get(identity), 'HIT_LEASE_MISMATCH')
            if identity in hits: hits[identity] += 1
            reference(receipt.get('execution_sha256'), 'execution'); reference(receipt.get('effect_sha256'), 'effect')
        else:
            check(receipt.get('native_applied') is False, 'UNREPORTED_NATIVE_INTERVENTION')
            check(receipt.get('outcome') in ('PASSTHROUGH', 'REJECTED', 'RESOURCE_LIMITED', 'STOPPED'), 'CALL_FAILED_OR_OUTCOME_MISSING')
            if report.get('trigger') == 'every_match' and eligible:
                errors.append('ELIGIBLE_CALL_NOT_HIT')
    check(observation.get('calls_sha256') == _sha(tickets), 'OBSERVATION_LEDGER_HASH_MISMATCH')
    check(report.get('observed_calls') == len(calls) == observation.get('observed_calls'), 'OBSERVED_CALL_COUNT_MISMATCH')
    check(report.get('observed_eligible_calls') == eligible_count > 0, 'ELIGIBLE_CALL_COUNT_MISMATCH')
    check(report.get('hits') == hit_count and report.get('planned_targets_hit') == sum(v > 0 for v in hits.values()), 'HIT_COUNTER_MISMATCH')
    check(report.get('missing_outcomes') == [], 'MISSING_OUTCOMES')
    check(report.get('propagation_added_calls') == [ticket for ticket in tickets if ticket.get('propagation_added')],
          'PROPAGATION_ADDITIONS_MISMATCH')
    check(all(v == 1 if report.get('trigger') == 'once' else v > 0 for v in hits.values()), 'TARGET_COVERAGE_INCOMPLETE')
    if report.get('trigger') == 'every_match':
        check(isinstance(c2_report, dict), 'MATCHING_SAVED_C2_MISSING')
        if isinstance(c2_report, dict):
            c2 = audit_saved_coverage(c2_report, artifact_index=artifact_index)
            check(c2['report_consistent'] and c2_report.get('trigger') == 'once', 'MATCHING_SAVED_C2_INVALID')
            check(c2_report.get('scope') == scope and c2_report.get('owner') == report.get('owner')
                  and c2_report.get('generation', 0) < report.get('generation', 0)
                  and c2['target_ids'] == sorted(leases)
                  and c2_report.get('discovery', {}).get('candidates_sha256') == discovery.get('candidates_sha256'), 'C2_SCOPE_OR_LIFETIME_MISMATCH')
    missing = sorted(references - set(artifact_index or {}))
    return dict(schema='demoncat.saved_coverage_audit.v1', report_sha256=_sha(report),
        report_consistent=not errors, errors=sorted(set(errors)), target_ids=sorted(leases),
        recomputed=dict(targets=len(leases), calls=len(calls), eligible_calls=eligible_count, hits=hit_count),
        referenced_artifacts=len(references), missing_artifact_sha256=missing,
        referenced_artifact_bytes_verified=bool(references) and not missing, scope=scope,
        public_qualification_granted=False, c2_qualification_granted=False, c3_qualification_granted=False,
        remaining_review=['actual independent observer universe and eligibility', 'lossless target proofs',
            'Native intervention and effect semantics', 'real device drain and same-process restoration',
            'actual source/build/loaded-library identity', 'live C2 proof admission for C3'])
