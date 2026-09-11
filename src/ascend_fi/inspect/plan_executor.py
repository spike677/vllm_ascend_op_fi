"""One model-independent runner, delegating mutation and cleanup to its owner."""
from __future__ import annotations

import copy
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate, rpc_workers
from .effect_stats import validate_stats
from .plan_builder import verify_plan
from .validation import ExistingOwnerIdentityValidator


def plan_window_gate(plan, fault, *, execution_coverage=None, identity=False):
    """Call counters are necessary but insufficient for token/phase coverage."""
    failures, measured = [], []
    workers = rpc_workers(fault.get('fault_transaction', {}).get('measurement_query')) or []
    arm_workers = rpc_workers(fault.get('fault_transaction', {}).get('arm')) or []
    disarm_workers = rpc_workers(fault.get('fault_disarm')) or []
    if (len(arm_workers) != len(plan['workers']) or not all(w.get('ok') is True for w in arm_workers)
            or len(disarm_workers) != len(plan['workers']) or not all(w.get('ok') is True for w in disarm_workers)):
        failures.append('ARM_OR_DISARM_NOT_CONFIRMED_ON_ALL_WORKERS')
    if len(workers) != len(plan['workers']):
        failures.append('WORKER_COUNT_MISMATCH')
    seen = set()
    for raw in workers:
        ranks = {m.get('tp_rank') for m in raw.get('matched_modules', [])}
        if len(ranks) != 1 or None in ranks:
            failures.append('WORKER_RANK_UNRESOLVED'); continue
        rank = next(iter(ranks))
        expected = next((w for w in plan['workers'] if w['rank'] == rank), None)
        if expected is None or rank in seen:
            failures.append('WORKER_RANK_MISMATCH'); continue
        seen.add(rank)
        armed = raw.get('armed_config', {})
        expected_arm = {**plan['prospective_arm'], 'request_generation': fault.get('request_generation')}
        if identity:
            expected_arm['mode'] = 'IDENTITY'
        if (type(fault.get('request_generation')) is not int
                or any(armed.get(k) != v for k, v in expected_arm.items())):
            failures.append('ARMED_PLAN_OR_REQUEST_GENERATION_MISMATCH')
        names = {t['module_name'] for t in expected['targets']}
        reject = raw.get('reject_reasons', {})
        contract = raw.get('input_contract', {})
        if (contract.get('installed') is not True or contract.get('plan_sha256') != plan['plan_sha256']
                or contract.get('violations') != 0 or contract.get('handles') != len(names)
                or contract.get('calls_checked', 0) < raw.get('hook_seen', 0)):
            failures.append('RUNTIME_INPUT_CONTRACT_NOT_PROVEN')
        if {m['module'] for m in raw.get('matched_modules', [])} != names:
            failures.append('INSTALLED_TARGET_SET_MISMATCH')
        if expected['targeted']:
            stats = validate_stats(raw, names, {t['layer'] for t in expected['targets']},
                                   rank, plan['prospective_arm'], identity=identity)
            if stats['status'] != 'PASS':
                failures.extend(stats['failures'])
            hits = raw.get('hit_count', 0)
            if not 0 < hits < plan['request_window']['max_hits'] or any(reject.values()):
                failures.append('HARD_CAP_OR_REJECTION')
            if raw.get('candidate_seen') != hits or raw.get('hook_seen') != hits:
                failures.append('TARGET_CALL_WINDOW_INCOMPLETE')
            for name in names:
                m = raw.get('per_module', {}).get(name, {})
                if not m.get('hit_count') or m.get('hit_count') != m.get('candidate_seen') or m.get('hit_count') != m.get('hook_seen'):
                    failures.append('MODULE_CALL_COVERAGE_INCOMPLETE')
        else:
            stats = None
            if raw.get('hit_count') != 0 or not raw.get('hook_seen') or reject != {'rank_filter': raw.get('hook_seen')}:
                failures.append('EXCLUDED_RANK_POLICY_VIOLATION')
        measured.append({'rank': rank, 'hit_count': raw.get('hit_count'), 'reject': reject,
                         'effect': stats, 'per_module': raw.get('per_module', {})})
    if (fault.get('http_status') != 200 or fault.get('request_window_evidence', {}).get('single_request_completed') is not True):
        failures.append('REQUEST_COMPLETION_UNPROVEN')
    # Coverage must come from an independent execution-step source, not hook/shape guesses.
    coverage_ok = bool(not failures and execution_coverage and execution_coverage.get('source') == 'scheduler_execution_steps'
        and execution_coverage.get('request_generation') == fault.get('request_generation')
        and execution_coverage.get('service_epochs') == sorted(w['service_epoch'] for w in plan['workers'])
        and execution_coverage.get('per_worker_expected_calls') ==
            {str(w['rank']): {t['module_name']: next((m['per_module'][t['module_name']]['hit_count']
              for m in measured if m['rank'] == w['rank']), None) for t in w['targets']}
             for w in plan['workers'] if w['targeted']})
    return {'mechanism': 'PASS' if not failures else 'INVALID_FAULT_RUN',
            'target_call_window': 'PASS' if not failures else 'WINDOW_INCOMPLETE',
            'full_request_window': 'PASS' if not failures and coverage_ok else 'UNPROVEN',
            'failures': failures, 'workers': measured,
            'coverage_gap': None if coverage_ok else 'INDEPENDENT_EXECUTION_STEP_COVERAGE_MISSING'}


class TargetPlanExecutor:
    """Owner must supply configure_target_plan plus the existing leased owner API.

    A fresh instance/plan needs fresh IDENTITY. It never consumes a caller-provided
    boolean 'identity_pass'. No launcher, NPU lock acquisition or new QDQ math here.
    """
    def __init__(self, owner, plan, current_descriptors):
        verify_plan(plan)
        self.owner, self.plan = owner, copy.deepcopy(plan)
        self.current_descriptors = current_descriptors
        self.identity_receipt = None
        self.config = None

    def rebind(self):
        verify_plan(self.plan)
        current = self.current_descriptors()
        expected = {(w['service_epoch'], w['runtime_fingerprint'], w['model_fingerprint']) for w in self.plan['workers']}
        actual = {(d['service_epoch'], d['runtime_fingerprint'], d['model_fingerprint']) for d in current}
        if actual != expected or len(current) != len(expected):
            raise ValueError('STALE_MODEL_RUNTIME_OR_WORKER_EPOCH')
        check = self.owner.preflight()
        if check.get('status') != 'PASS' or check.get('exclusive_lock_owned') is not True:
            raise ValueError('EXCLUSIVE_OWNER_REQUIRED')
        if cleanup_gate(self.owner.clean_query(), self.owner.service_healthy(), len(expected))['status'] != 'PASS':
            raise ValueError('OWNER_NOT_CLEAN')

    def validate_identity(self, probe):
        self.identity_receipt = None
        self.rebind()
        self.owner.configure_target_plan(copy.deepcopy(self.plan))
        try:
            self.config = self.owner.install()
        finally:
            self.owner.disarm_and_remove()
        targeted = [w for w in self.plan['workers'] if w['targeted']]
        selected = self.plan['selected_modules']
        bridge = {'candidate_ids': [self.plan['plan_sha256']], 'service_epoch': targeted[0]['service_epoch'],
                  'runtime_fingerprint': targeted[0]['runtime_fingerprint'],
                  'prospective_arm': self.plan['prospective_arm'], 'selected_modules': selected,
                  'ranks': [w['rank'] for w in targeted], 'unique_float_input_proven': True}
        def identity():
            self.rebind()
            return bridge['service_epoch'], bridge['runtime_fingerprint']
        result = ExistingOwnerIdentityValidator(self.owner, self.config, probe, identity).validate_identity(bridge)
        raw = result.to_dict().get('raw', {})
        pieces = raw.get('baseline', {}).get('token_pieces')
        exact = pieces is not None and all(raw.get(k, {}).get('token_pieces') == pieces for k in ('disabled', 'identity', 'recovery'))
        gate = plan_window_gate(self.plan, raw.get('identity', {}), identity=True)
        if result.lossless_passed and result.restore_exact and exact and gate['mechanism'] == 'PASS':
            self.identity_receipt = {'plan_sha256': self.plan['plan_sha256'], 'token_pieces_exact': True}
        else:
            raise RuntimeError('MODEL_AWARE_IDENTITY_GATE_FAILED')
        return {'validation': result.to_dict(), 'identity_receipt': self.identity_receipt, 'window': gate}

    def run_pair(self, probe, cycle=1):
        if not self.identity_receipt or self.identity_receipt['plan_sha256'] != self.plan['plan_sha256']:
            raise ValueError('SAME_PLAN_IDENTITY_REQUIRED')
        self.rebind()
        if self.owner.arm != self.plan['prospective_arm']:
            raise ValueError('OWNER_ARM_CHANGED_AFTER_IDENTITY')
        baseline = self.owner.request(probe, 'baseline', cycle)
        try:
            config = self.owner.install()
            if set(config.actual_module_names) != set(self.plan['selected_modules']):
                raise ValueError('INSTALLED_MODULE_SET_MISMATCH')
            fault = self.owner.fault_request(probe, config, cycle)
        except BaseException:
            self.identity_receipt = None
            raise
        finally:
            self.owner.disarm_and_remove()
        after = cleanup_gate(self.owner.clean_query(), self.owner.service_healthy(), len(self.plan['workers']))
        if after['status'] != 'PASS':
            raise RuntimeError('RECOVERY_CLEAN_STATE_FAILED; stop subsequent requests')
        recovery = self.owner.request(probe, 'recovery', cycle)
        equal = (baseline.get('http_status') == recovery.get('http_status') == 200
                 and baseline.get('response_text') is not None
                 and baseline.get('response_text') == recovery.get('response_text')
                 and baseline.get('token_pieces') is not None
                 and baseline.get('token_pieces') == recovery.get('token_pieces'))
        gate = plan_window_gate(self.plan, fault)
        if not equal or gate['mechanism'] != 'PASS':
            self.identity_receipt = None  # Stop subsequent faulty requests after any failed gate.
        return {'pair': {'baseline': baseline, 'fault': fault, 'recovery': recovery},
                'window': gate, 'cleanup': after, 'recovery_exact': equal,
                'status': 'PASS' if equal and gate['mechanism'] == 'PASS' else 'PARTIAL'}
