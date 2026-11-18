"""One disposable public Worker, three actual model statistics windows.

The existing public probe drives all ten real enqueued requests and recovery.
This entry composes its drive function only; no Core/provider/registry callable
is replaced. Numerical comparison is a separate raw-only command.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'fault39/src')]
from experiments.c2c3_closure_20261117 import natural_tensor_worker_core_probe as original
from experiments.worker_core_v11_20261118.core_tensor_statistics_capture import (
    PublicTensorStatisticsSidecar, RUNTIME_FILES, require, sha)

ATTRIBUTE = '_demoncat_experimental_public_rms_statistics'
EXTRA_FILES = ('experiments/worker_core_v11_20261118/core_tensor_statistics_capture.py',
               'experiments/worker_core_v11_20261118/collector_packet_compat.py',
               'experiments/worker_core_v11_20261118/core_tensor_statistics_probe.py')
ORIGINAL_DRIVE = original.drive


def source_pins(path, expected_sha256):
    path = Path(path)
    require(sha(path) == expected_sha256, 'FROZEN_SOURCE_MANIFEST_CHANGED')
    manifest = json.loads(path.read_text())
    require(manifest.get('schema') == 'demoncat.c2c3.source.v1'
            and type(manifest.get('files')) is list, 'EXACT_FROZEN_SOURCE_MANIFEST_REQUIRED')
    entries = {row['path']: row for row in manifest['files']}
    require(len(entries) == len(manifest['files']), 'DUPLICATE_SOURCE_MANIFEST_PATH')
    for name in RUNTIME_FILES + EXTRA_FILES:
        require(name in entries and sha(ROOT/name) == entries[name]['sha256'], 'CAPTURE_SOURCE_BYTES_NOT_FROZEN:'+name)
    return {str((ROOT/name).resolve()): entries[name]['sha256'] for name in RUNTIME_FILES}


def compact_capture(sidecar):
    sidecar.save()
    report = sidecar.report()
    path = sidecar.folder/'collector_sidecar.json'
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
        phase_steps=report['phase_steps'], error=report['error'], profile_owned=report['profile_owned'],
        capture_active=report['capture_active'], hook_snapshot_retained=report['hook_snapshot_retained'],
        session_capture_attached=report['session_capture_attached'],
        global_api_wrappers_restored=False, disposable_worker_required=True,
        sampler_captured=False, root_module_record_absent=True, qualification_granted=False, e5=False)


def install_worker_capture(worker, expected, ticket, folder, manifest_path, manifest_sha256):
    worker = original.actual(worker)
    identity = worker.demoncat_worker_identity()
    require(identity.get('status') == 'WORKER_IDENTIFIED' and identity.get('worker') == expected
            and identity.get('quarantined') is False, 'CAPTURE_ACTUAL_WORKER_IDENTITY_CHANGED')
    runtime = getattr(worker, '_demoncat_worker_fault_runtime', None)
    lease = runtime._leases.get('18.api_half') if runtime is not None else None
    require(lease is not None and lease.ticket() == ticket, 'CAPTURE_PUBLIC_RUNTIME_OWNER_CHANGED')
    require(not hasattr(worker, ATTRIBUTE), 'COLLECTOR_SIDECAR_ALREADY_OWNED')
    require(ticket['mechanism'] == '18.api_half' and lease.provider.session is not None,
            'ENABLED_PUBLIC_RMS_SESSION_REQUIRED')
    session = lease.provider.session
    if not session.provider.simulation:
        require(Path(folder).is_absolute() and Path(folder).resolve().is_relative_to(Path('/evidence')),
                'PRIVATE_EVIDENCE_OUTPUT_REQUIRED')
    sidecar = PublicTensorStatisticsSidecar(session, folder,
        source_pins=source_pins(manifest_path, manifest_sha256))
    sidecar.public_ticket = deepcopy(ticket)
    sidecar.expected_worker = deepcopy(expected)
    setattr(worker, ATTRIBUTE, sidecar)
    return compact_capture(sidecar)


def close_worker_capture(worker, expected, ticket):
    worker = original.actual(worker)
    identity = worker.demoncat_worker_identity()
    require(identity.get('worker') == expected and identity.get('quarantined') is False,
            'CAPTURE_CLEANUP_WORKER_CHANGED')
    sidecar = getattr(worker, ATTRIBUTE, None)
    require(type(sidecar) is PublicTensorStatisticsSidecar and sidecar.public_ticket == ticket
            and sidecar.expected_worker == expected and sidecar.session.worker is worker,
            'EXACT_CAPTURE_OWNER_REQUIRED')
    runtime = getattr(worker, '_demoncat_worker_fault_runtime', None)
    lease = runtime._leases.get('18.api_half') if runtime is not None else None
    require(lease is None or lease.ticket() == ticket, 'NEW_PUBLIC_OWNER_MUST_NOT_BE_TOUCHED')
    sidecar.close()  # Core remains responsible for an outstanding completion/recovery.
    report = compact_capture(sidecar)
    require(not any(report[key] for key in ('profile_owned', 'capture_active', 'hook_snapshot_retained',
                                            'session_capture_attached')), 'CAPTURE_OWNERSHIP_RETAINED')
    delattr(worker, ATTRIBUTE)
    return report


class StatisticsClient:
    """Delegate unchanged public RPCs; install only after a successful enable."""
    def __init__(self, client, llm, *, out, report, manifest_path, manifest_sha256):
        self.client, self.llm, self.out, self.report = client, llm, Path(out), report
        self.manifest_path, self.manifest_sha256 = str(manifest_path), manifest_sha256
        self.ticket = self.worker = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    def enable(self):
        snapshot = self.client.enable()
        if snapshot['status'] == 'ACTION_COMPLETED':
            self.ticket, self.worker = deepcopy(snapshot['ticket']), deepcopy(snapshot['worker'])
            rows = self.llm.collective_rpc(install_worker_capture, args=(self.worker, self.ticket,
                str(self.out/'raw_model_capture'), self.manifest_path, self.manifest_sha256), timeout=300)
            require(len(rows) == 1, 'TP1_COLLECTOR_OWNER_REQUIRED')
            self.report['collector_install'] = rows[0]
        return snapshot

    def close_capture(self):
        if self.ticket is None:
            return
        rows = self.llm.collective_rpc(close_worker_capture, args=(self.worker, self.ticket), timeout=300)
        require(len(rows) == 1, 'TP1_COLLECTOR_CLEANUP_REQUIRED')
        self.report['collector_final'] = rows[0]
        require(rows[0]['error'] is None and [(r['phase'], r['actual_step'], r['completed'])
            for r in rows[0]['phase_steps']] == [('baseline', 0, True), ('baseline_repeat', 1, True),
                                                ('c2_fault', 2, True)], 'THREE_ACTUAL_COMPLETE_WINDOWS_REQUIRED')


def drive(llm, client, *, prompt, params, out, report, manifest_path, manifest_sha256):
    require(client.config.mechanism == '18.api_half', 'THIS_EXPERIMENT_ONLY_REVIEWS_RMS18')
    proxy = StatisticsClient(client, llm, out=out, report=report,
        manifest_path=manifest_path, manifest_sha256=manifest_sha256)
    report.update(statistics_adapter_sha256=sha(__file__), source_manifest_sha256=manifest_sha256,
        statistics_scope='actual model root entry through logits processor return',
        sampler_captured=False, native_root_module_record_absent=True, public_e5=False,
        global_collector_api_wrappers_restored=False, disposable_worker_required=True)
    result = ORIGINAL_DRIVE(llm, proxy, prompt=prompt, params=params, out=out, report=report)
    try:
        proxy.close_capture()
        require(result['status'] == original.SUCCESS, 'PUBLIC_PROTOCOL_FAILED_BEFORE_STATISTICS_REVIEW')
        report['statistics_status'] = 'THREE_ORIGINAL_WINDOWS_READY_FOR_INDEPENDENT_RAW_REVIEW'
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED', statistics_error=type(error).__name__+': '+str(error),
                      statistics_status='PARTIAL_OR_FAILED_CAPTURE_RETAINED')
    original.write_json(Path(out)/'result.json', report)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    args, rest = parser.parse_known_args(argv)
    source_pins(args.source_manifest, args.source_manifest_sha256)
    require('--mechanism' in rest and rest[rest.index('--mechanism')+1] == '18.api_half',
            'EXPLICIT_RMS18_PUBLIC_PROBE_REQUIRED')
    # Composition is limited to the experiment entry. Runtime classes,
    # provider factories, registry and exporter retain their original objects.
    saved = original.drive
    require(saved is ORIGINAL_DRIVE, 'PUBLIC_PROBE_DRIVE_ALREADY_REPLACED')
    original.drive = lambda *a, **kw: drive(*a, **kw, manifest_path=args.source_manifest,
                                          manifest_sha256=args.source_manifest_sha256)
    try:
        return original.main(rest)
    finally:
        original.drive = saved


if __name__ == '__main__':
    raise SystemExit(main())
