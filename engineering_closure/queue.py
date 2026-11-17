"""Dependency planning and handoff to the existing single NPU controller."""
import json
from pathlib import Path

from fault_injection.catalog import CAPABILITIES
from fault_injection.evidence import digest
from experiments.c2c3_closure_20261117.queue_contract import validate_cases


def load_batches(directory):
    batches = [json.loads(path.read_bytes()) for path in sorted(Path(directory).glob('batch[1-5].json'))]
    if len(batches) != 5:
        raise ValueError('FIVE_BATCHES_REQUIRED')
    jobs = [job for batch in batches for job in batch['jobs']]
    if sorted(job['mechanism'] for job in jobs) != sorted(item.name for item in CAPABILITIES):
        raise ValueError('EXACT_MECHANISM_ACCOUNTING_REQUIRED')
    indexed = {job['id']: job for job in jobs}
    if len(indexed) != len(jobs):
        raise ValueError('DUPLICATE_JOB_ID')
    visiting, done = set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError('DEPENDENCY_CYCLE:' + key)
        if key in done:
            return
        if key not in indexed:
            raise ValueError('UNKNOWN_JOB_DEPENDENCY:' + key)
        visiting.add(key)
        for parent in indexed[key].get('after', []):
            visit(parent)
        visiting.remove(key)
        done.add(key)

    for job in jobs:
        visit(job['id'])
        if not job.get('gap') or not job.get('requires') or not job.get('scope'):
            raise ValueError('EXPLICIT_GAP_SCOPE_AND_GATES_REQUIRED')
    return jobs


def plan_queue(directory, *, receipts=(), evidence_root=None):
    """Receipts are explicit review decisions with hashed backing files.

    This checks prerequisites for preparation only. The existing host controller
    still owns actual device scheduling, and the Provider owns live admission.
    """
    jobs = load_batches(directory)
    root = Path(evidence_root or directory).resolve()
    known = {}
    for receipt in receipts:
        key = (receipt['job'], receipt['gate'])
        if key in known:
            raise ValueError('DUPLICATE_GATE_RECEIPT')
        path = (root / receipt['artifact']).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or digest(path) != receipt.get('sha256')):
            raise ValueError('GATE_ARTIFACT_MISSING_OR_CHANGED')
        if not receipt.get('reviewer'):
            raise ValueError('GATE_REVIEWER_REQUIRED')
        known[key] = receipt
    rows = []
    for job in jobs:
        blockers = []
        for gate in job['requires']:
            receipt = known.get((job['id'], gate), {})
            if receipt.get('status') != 'PASS' or receipt.get('scope') != job['scope']:
                blockers.append(gate)
        for parent in job.get('after', []):
            receipt = known.get((parent, 'complete'), {})
            parent_job = next(row for row in jobs if row['id'] == parent)
            if receipt.get('status') != 'PASS' or receipt.get('scope') != parent_job['scope']:
                blockers.append('job:' + parent)
        status = 'READY_TO_PREPARE' if not blockers else 'PENDING_PREREQUISITES'
        if not job['entrypoint']:
            status = 'IMPLEMENTATION_PENDING'
        rows.append(dict(id=job['id'], mechanism=job['mechanism'], batch=job['batch'],
            status=status,
            missing_gates=blockers, gap=job['gap'], inherited=job['inherited'],
            stage_goal=job['stage_goal'], remaining_after_stage=job['remaining_after_stage'],
            scope=job['scope'], entrypoint=job['entrypoint'],
            preparation_entrypoint=job.get('preparation_entrypoint'),
            pre_gate_actions=job.get('pre_gate_actions', []),
            task_kind=job['task_kind'],
            actual_npu_execution_authorized=False))
    return dict(schema='demoncat.engineering.queue_plan.v1', jobs=rows,
                controller='experiments/c2c3_closure_20261117/run_queue_124.py',
                queued_npu_processes=0)


def prepare_cases(kind, options):
    """Reuse reviewed case factories for runnable controller input."""
    if kind == 'worker18-statistics':
        from experiments.c2c3_closure_20261117.prepare_tensor_natural_cases import cases
        result = cases(task=kind, **options)
    elif kind in ('39.output_proxy', '41.tail_column'):
        if kind == '39.output_proxy':
            from experiments.c2c3_postwrite_20261117.prepare_coverage_cases import cases
        else:
            from experiments.c2c3_postwrite_20261117.prepare41_coverage_cases import cases
        result = cases(**options)
    else:
        raise ValueError('USE_REVIEWED_ENTRYPOINT_LISTED_IN_BATCH:' + kind)
    cards = result[0]['review']['physical_cards']
    return validate_cases(result, cards)
