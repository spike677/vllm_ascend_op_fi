"""Concrete factory delegating to the project's already verified leased R1 owner.

Only the project tools/run_normal_diagnostic_live.py owner is imported. It retains
its lock checks, immutable manifest, per-request arm, finally-disarm and cleanup.
This adapter adds no fault algorithm or alternative mutation control path.
"""
import importlib.util
import re
from pathlib import Path

from ascend_fi.evaluation.distortion.contracts import FaultConfigurationRecord
from .client import InspectClient
from .validation import ExistingOwnerIdentityValidator


def create_owner(session, suite):
    source = Path(__file__).resolve().parents[3] / 'tools/run_normal_diagnostic_live.py'
    spec = importlib.util.spec_from_file_location('inspect_existing_r1_owner', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VerifiedOwner(Path(session), suite)


def factory(args):
    owner = create_owner(args['session_dir'], args['suite'])
    run_id = args.get('validation_run', 'identity_adapter_v1')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', run_id):
        raise ValueError('INVALID_VALIDATION_RUN_ID')
    owner.rawroot = Path(args['session_dir']) / 'inspect_identity' / run_id
    if owner.rawroot.exists():
        raise ValueError('PRESERVE_PRIOR_VALIDATION; choose a new validation_run')
    config = FaultConfigurationRecord.from_dict(args['fault_configuration'])
    client = InspectClient('http://127.0.0.1:18031')
    target_epoch = args['plan']['service_epoch']

    def identity():
        rows = client.call('status')
        all_workers = [r['result'] for r in rows if not r.get('skipped')]
        if any(w['active'] or w['installed_handles'] for w in all_workers):
            raise RuntimeError('STOP_INSPECT_OBSERVER_ON_ALL_WORKERS_BEFORE_IDENTITY')
        active = [w for w in all_workers if w['service_epoch'] == target_epoch]
        if len(active) != 1:
            raise RuntimeError('EXACT_CURRENT_WORKER_REQUIRED')
        if active[0]['active'] or active[0]['installed_handles']:
            raise RuntimeError('STOP_INSPECT_OBSERVER_BEFORE_IDENTITY')
        return active[0]['service_epoch'], active[0]['runtime_fingerprint']

    return ExistingOwnerIdentityValidator(owner, config, args['probe'], identity)
