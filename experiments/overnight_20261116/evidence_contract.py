"""Offline consistency checks for sprint reports, not proof or runtime admission.

These checks cannot authenticate a supplied receipt or grant production support.
Existing TargetIdentity, QualifiedNativeSession and Worker admission remain the
runtime authority. In particular, a CPU simulation can never validate an NPU run.
"""
import hashlib
import json
import re


class EvidenceError(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise EvidenceError(reason)


def scope_hash(scope):
    return hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_execution(record, expected_scope, *, require_model=False):
    """Reject scope transfer, mock promotion and missing recovery observations."""
    from fault_injection.catalog import resolve
    resolve(record['mechanism'])
    _require(record.get('execution_kind') == 'npu', 'CPU_OR_MOCK_IS_NOT_NATIVE')
    _require(record.get('scope') == expected_scope, 'EVIDENCE_SCOPE_MISMATCH')
    for field in ('entrypoint', 'operator', 'cann', 'soc', 'worker_start_id', 'model_instance_id'):
        _require(isinstance(expected_scope.get(field), str) and bool(expected_scope[field]), 'MISSING_SCOPE_' + field)
    build=expected_scope.get('build_sha256')
    _require(isinstance(build,str) and re.fullmatch('[0-9a-f]{64}',build), 'MISSING_LOADED_BUILD')
    _require(type(expected_scope.get('rank')) is int and expected_scope['rank'] >= 0, 'INVALID_RANK')
    identity=scope_hash(expected_scope)
    for stage in ('G5','G6','G7','G8'):
        row=record.get('gates',{}).get(stage,{})
        _require(row.get('status') == 'PASS' and row.get('scope_sha256') == identity, 'MISSING_OR_FOREIGN_' + stage)
    native=record['gates']['G6']
    _require(type(native.get('actual_calls')) is int and native['actual_calls'] > 0, 'NO_ACTUAL_NATIVE_CALL')
    _require(native.get('actual_execution_verified') is True, 'HIT_COUNT_IS_NOT_EXECUTION')
    recovery=record['gates']['G8']
    _require(recovery.get('drained') is True and recovery.get('off_verified') is True
             and recovery.get('pending') == 0 and type(recovery.get('pending')) is int, 'RECOVERY_UNPROVEN')
    if require_model:
        _require(record.get('pretrained_model') is True, 'CONSTRUCTED_MODULE_IS_NOT_PRETRAINED_MODEL')
        _require(record.get('request_mapping_verified') is True, 'MODEL_REQUEST_MAPPING_UNPROVEN')
        stats=validate_statistics(record.get('independent_statistics',{}))
        _require(stats['numerical_localization_complete'], 'STATISTICAL_LOCALIZATION_INCOMPLETE')
        return 'MODEL_CLOSED'
    return 'NATIVE_VERIFIED'


def validate_statistics(report):
    _require(report.get('task')=='statistics' and report.get('level')=='mix'
             and report.get('summary_mode')=='statistics', 'NOT_RAW_STATISTICS_MIX')
    _require(report.get('rtol')==1e-5 and report.get('atol')==1e-8
             and report.get('reference')=='baseline', 'FIXED_TOLERANCE_CHANGED')
    _require(report.get('native_logs_read') is False and report.get('private_tensors_read') is False,
             'FIRST_DIFFERENCE_NOT_INDEPENDENT')
    _require(report.get('aligned') is True, 'CALL_OR_CONFIG_ALIGNMENT_FAILED')
    steps=report.get('baseline_steps')
    _require(isinstance(steps,list) and bool(steps) and all(type(s) is int and s>=0 for s in steps)
             and steps==sorted(set(steps)) and steps==report.get('fault_steps'), 'ACTUAL_STEPS_MISMATCH')
    rows=report.get('steps',[])
    _require([r.get('step') for r in rows]==steps, 'MISSING_STEP_RESULT')
    for r in rows:
        for key in ('unobservable_count','incomparable_count'):
            _require(type(r.get(key)) is int and r[key]>=0, 'MISSING_UNOBSERVABLE_ACCOUNTING')
        _require(r.get('status') in ('DIFFERENCE','NO_OBSERVABLE_DIFFERENCE','INCOMPARABLE'), 'INVALID_STATISTICS_RESULT')
        if r['incomparable_count']:
            _require(r['status']=='INCOMPARABLE', 'NAN_CANNOT_BE_EQUAL')
    return dict(numerical_localization_complete=all(r['status']!='INCOMPARABLE' for r in rows),
                nulls_preserved=sum(r['unobservable_count'] for r in rows),
                observed_change=any(r['status']=='DIFFERENCE' for r in rows))


def validate_coverage(report):
    """Neither positive hit counts nor a zero denominator prove full coverage."""
    for name in ('eligible_calls','executed_calls','completed_calls','restored_calls'):
        _require(type(report.get(name)) is int and report[name]>=0, 'INVALID_COVERAGE_COUNT')
    n=report['eligible_calls']
    _require(n>0 and all(report[k]==n for k in ('executed_calls','completed_calls','restored_calls')),
             'INCOMPLETE_COVERAGE_OR_RECOVERY')
    _require(report.get('denominator_source')=='independent_observer', 'HITS_CANNOT_DEFINE_DENOMINATOR')
    _require(report.get('discovery_completed') is True and report.get('group_recovery_verified') is True,
             'GROUP_QUALIFICATION_MISSING')
    _require(report.get('resource_budget_verified') is True and report.get('rank_isolation_verified') is True,
             'COVERAGE_BUDGET_OR_RANK_UNPROVEN')
    return 'COVERAGE_VERIFIED_WITHIN_RECORDED_SCOPE'
