"""Control-plane confirmation for explicit-selection plans; no collectives in hooks."""
from ascend_fi.evaluation.distortion.mechanism import rpc_workers


def confirm_workers(plan, query, *, ack=None, expected_arm=None):
    rows = rpc_workers(query) or []
    if len(rows) != len(plan['workers']):
        raise ValueError('NECESSARY_WORKER_MISSING')
    seen, pids = set(), set()
    for row in rows:
        contract = row.get('input_contract', {})
        identity = contract.get('worker_identity', {})
        epoch, rank, pid = identity.get('service_epoch'), identity.get('global_rank'), row.get('pid')
        expected = next((w for w in plan['workers'] if w['service_epoch'] == epoch), None)
        if (expected is None or epoch in seen or rank != expected['rank']
                or type(pid) is not int or pid <= 0 or pid in pids
                or identity.get('reference_rank') != rank or identity.get('module_tp_ranks') != [rank]):
            raise ValueError('WORKER_IDENTITY_OR_REFERENCE_RANK_NOT_PROVEN')
        names = [m.get('module') for m in row.get('matched_modules', [])]
        if (row.get('installed') is not True or sorted(names) != sorted(plan['selected_modules'])
                or contract.get('installed') is not True or contract.get('plan_sha256') != plan['plan_sha256']
                or contract.get('handles') != len(names) or contract.get('violations') != 0):
            raise ValueError('WORKER_INSTALL_OR_INPUT_BINDING_INCOMPLETE')
        if expected_arm is None:
            if row.get('enabled') is not False:
                raise ValueError('PRE_ARM_WORKER_NOT_DISABLED')
        elif (row.get('enabled') is not True
                or any(row.get('armed_config', {}).get(k) != v for k, v in expected_arm.items())):
            raise ValueError('NECESSARY_WORKER_ARM_NOT_CONFIRMED')
        seen.add(epoch)
        pids.add(pid)
    if ack is not None:
        confirmations = rpc_workers(ack) or []
        if (len(confirmations) != len(rows) or not all(r.get('ok') is True for r in confirmations)
                or {r.get('pid') for r in confirmations} != pids):
            raise ValueError('NECESSARY_WORKER_ACK_MISSING_OR_DUPLICATED')
    return {'status': 'PASS', 'service_epochs': sorted(seen), 'worker_pids': sorted(pids),
            'participating_global_ranks': sorted(w['rank'] for w in plan['workers']),
            'targeted_global_ranks': plan['prospective_arm']['rank_filter']}
