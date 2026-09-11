"""Explicit bridge to the existing lifecycle owner, not a second controller."""
from __future__ import annotations

import re

from ascend_fi.contracts import ValidationResult, ValidationStatus
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate, rpc_workers


class ExistingOwnerIdentityValidator:
    """Owner API: preflight, clean_query, request, install, fault_request(identity=True),
    disarm_and_remove, service_healthy. Caller supplies its already-acquired lease.
    """
    def __init__(self, owner, config, probe, current_identity):
        self.owner, self.config, self.probe, self.current_identity = owner, config, probe, current_identity

    def validate_identity(self, plan):
        cid = ','.join(plan['candidate_ids'])
        current = self.current_identity()
        if current != (plan['service_epoch'], plan['runtime_fingerprint']):
            return ValidationResult(cid, False, False, False, status=ValidationStatus.REJECTED, reason='STALE_SERVICE_EPOCH')
        owner = self.owner
        pre = owner.preflight()
        if pre.get('status') != 'PASS' or pre.get('exclusive_lock_owned') is not True:
            raise RuntimeError('EXISTING_OWNER_LEASE_REQUIRED')
        before = cleanup_gate(owner.clean_query(), owner.service_healthy(), self.config.world_size)
        if before['status'] != 'PASS':
            raise RuntimeError('EXISTING_OWNER_NOT_CLEAN')
        # This bridge cannot silently reconfigure the frozen lifecycle owner.
        arm = owner.arm
        planned = plan['prospective_arm']
        for key in ('mode', 'rank_filter', 'scale_provider', 'scale_multiplier', 'max_hits',
                    'interval', 'eps', 'runtime_scale_attr', 'fixed_scale'):
            if arm.get(key) != planned.get(key):
                raise ValueError('EXISTING_OWNER_CONFIG_MISMATCH:' + key)
        if planned['tensor_selector'] != arm.get('tensor_selector'):
            # first_float is only equivalent when discovery proved a unique floating input.
            if arm.get('tensor_selector') != 'first_float' or not plan.get('unique_float_input_proven'):
                raise ValueError('EXISTING_OWNER_SELECTOR_NOT_EQUIVALENT')
        elif arm.get('tensor_index') != planned.get('tensor_index'):
            raise ValueError('EXISTING_OWNER_SELECTOR_NOT_EQUIVALENT')
        baseline = owner.request(self.probe, 'inspect_baseline', 0)
        identity = disabled = None
        try:
            actual = owner.install(self.config)
            names = actual.to_dict().get('actual_module_names', [])
            if set(names) != set(plan['selected_modules']):
                raise ValueError('EXISTING_OWNER_TARGET_SET_NOT_EQUIVALENT')
            if any(not re.search(arm['module_regex'], n) for n in names):
                raise ValueError('EXISTING_OWNER_PATTERN_MISMATCH')
            disabled = owner.request(self.probe, 'inspect_disabled', 0)
            identity = owner.fault_request(self.probe, actual, 0, identity=True)
        finally:
            owner.disarm_and_remove()
        clean = cleanup_gate(owner.clean_query(), owner.service_healthy(), self.config.world_size)
        if clean['status'] != 'PASS':
            raise RuntimeError('IDENTITY_CLEANUP_NOT_PROVEN')
        recovery = owner.request(self.probe, 'inspect_recovery', 0)
        transaction = identity.get('fault_transaction', {})
        workers = rpc_workers(transaction.get('measurement_query')) or []
        targeted, good, ranks_seen = 0, len(workers) == self.config.world_size, set()
        for worker in workers:
            histories = worker.get('tensor_effect_history', [])
            rank_set = {m.get('tp_rank') for m in worker.get('matched_modules', [])}
            if len(rank_set) != 1 or None in rank_set or ranks_seen.intersection(rank_set):
                good = False
                continue
            rank = next(iter(rank_set))
            ranks_seen.add(rank)
            if rank in plan['ranks'] and worker.get('hit_count', 0):
                targeted += worker['hit_count']
                if 'effect_summary' in worker:
                    from .effect_stats import validate_stats
                    checked = validate_stats(worker, plan['selected_modules'], self.config.targeted_layers,
                                             rank, owner.arm, identity=True)
                    good = good and checked['status'] == 'PASS'
                else:
                    good = good and len(histories) == worker['hit_count'] and all(
                        h.get('mode') == 'IDENTITY' and h.get('changed_elements') == 0
                        and h.get('rank') == rank and h.get('module') in plan['selected_modules'] for h in histories)
                    good = good and {h.get('module') for h in histories} == set(plan['selected_modules'])
            elif rank in plan['ranks'] or worker.get('hit_count') != 0:
                good = False
        def answer(row):
            return row.get('response_text', row.get('response', row.get('text')))
        equal = answer(baseline) is not None and answer(baseline) == answer(disabled) == answer(identity) == answer(recovery)
        ok = good and targeted > 0 and equal and before.get('pids') == clean.get('pids') and all(
            r.get('http_status') == 200 for r in (baseline, disabled, identity, recovery))
        return ValidationResult(cid, True, True, ok, hit_count=targeted, model_output_equal=equal,
            restore_exact=equal and clean['status'] == 'PASS',
            status=ValidationStatus.LOSSLESS_VALIDATED if ok else ValidationStatus.FAILED,
            reason='Existing lifecycle IDENTITY gate; no numeric transform or native substitution',
            extra={'validation_kind': 'MODULE_IDENTITY', 'native_slot_proven': False,
                   'raw': {'baseline': baseline, 'disabled': disabled, 'identity': identity, 'recovery': recovery}, 'cleanup': clean})
