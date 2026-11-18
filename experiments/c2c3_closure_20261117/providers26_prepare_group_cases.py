"""Prepare at most eight root-reviewed candidate or complete-set v2 cases."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'native', 'probe', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--max-targets', type=int, default=8)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--worker-core', action='store_true')
    parser.add_argument('--source', help='Frozen source root for the new public Core C1 probe')
    parser.add_argument('--stages', nargs='+', choices=('inventory', 'lossless', 'effects', 'coverage', 'c1_single_effects'), default=['inventory'])
    parser.add_argument('--mechanisms', nargs='+', choices=('26.residual_stomp', '26.fia_destination'),
                        default=None)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.worker_core and (args.stages != ['c1_single_effects'] or args.max_targets != 1 or args.statistics
            or args.source is None or not args.source.startswith('/')
            or any(c in args.source for c in ('\n', '\r', '\0', ','))):
        parser.error('Public Core needs --source, only c1_single_effects, max-targets=1, and no unadapted statistics')
    if args.worker_core:
        source = Path(args.source).resolve(strict=True)
        expected_probe = source/'experiments/c2c3_closure_20261117/providers26_worker_core_probe.py'
        if Path(args.probe).resolve(strict=True) != expected_probe.resolve(strict=True):
            parser.error('Public Core requires the reviewed providers26_worker_core_probe.py entry')
    if 'c1_single_effects' in args.stages and args.mechanisms is None:
        parser.error('c1_single_effects requires explicit --mechanisms 26.residual_stomp and/or 26.fia_destination')
    if args.mechanisms is None:
        args.mechanisms = ['26.residual_stomp', '26.fia_destination']
    if (not re.fullmatch('[0-9a-f]{7,40}', args.source_commit) or not 1 <= args.max_targets <= 128
            or len(set(args.stages)) != len(args.stages) or len(set(args.mechanisms)) != len(args.mechanisms)
            or any(not path.startswith('/') or any(c in path for c in ('\n', '\r', '\0', ','))
                   for path in (args.installed, args.native, args.probe))):
        parser.error('Frozen paths, source and bounded distinct phases required')
    root = Path(args.native).resolve(strict=True)
    manifest = root / 'build.json'
    build = json.loads(manifest.read_text())
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if (build.get('schema') != 'demoncat.writer26.group_deployment.v2' or build.get('actual_npu_calls') != 0
            or not build.get('legacy_writers_absent') or not re.fullmatch(r'sha256:[0-9a-f]{64}', build.get('actual_image_id', ''))):
        raise RuntimeError('ACTUAL_V2_SDK_BUILD_IMAGE_ID_REQUIRED')
    for name, expected in build['files'].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('NATIVE_BUNDLE_CHANGED:' + name)
    if 'c1_single_effects' in args.stages and '26.fia_destination' in args.mechanisms:
        from fault_injection.deployments.writer26_fia_typed import profile
        profile(build.get('fia_typed_profile'))
    cases = []
    for mechanism in args.mechanisms:
        fia = mechanism == '26.fia_destination'
        model = '/home/fault39-integration-20261116-a1/models-smollm2-135m' if fia else '/home/haoran/models/DeepSeek-V2-Lite-W8A8'
        for stage in args.stages:
            coverage = stage == 'coverage'
            single_site = stage == 'c1_single_effects'
            name = ('providers26-core-' if args.worker_core else 'providers26-v2-') + ('fia-smol-' if fia else 'qmm-deepseek-') + stage
            command = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
                'export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend\n'
                'export LD_PRELOAD=/native/libdcat26_group_v2.so:/native/libfault26_readonly_trace_abi0.so:'
                '/native/libfault26_fia_opapi_proxy.so:/native/libopapi.so:/native/libcust_opapi.so\n'
                'export LD_LIBRARY_PATH=/native:/native/real:$LD_LIBRARY_PATH\n'
                'export PYTHONPATH=/installed:/msprobe_packages'+(':/source' if args.worker_core else '')+':$PYTHONPATH\n'
                'exec python3 ' + ('/source/experiments/c2c3_closure_20261117/providers26_worker_core_probe.py'
                    if args.worker_core else '/probe.py') + ' --mechanism ' + mechanism + ' --manifest-sha256 ' + sha + ' --stage ' + stage
                + ' --max-targets ' + str(args.max_targets) + ' --memory ' + ('.25' if fia else '.55')
                + (' --statistics' if args.statistics else '')
                + (' --source-root /source > /evidence/worker-core26.stdout.log 2> /evidence/worker-core26.stderr.log'
                   if args.worker_core else ''))
            cases.append(dict(id=name, timeout=1800, image=build['actual_image_id'],
                mounts=([[args.source, '/source', 'ro']] if args.worker_core else []) +
                    [[args.installed, '/installed', 'ro'], [args.native, '/native', 'ro'], [args.probe, '/probe.py', 'ro'],
                    [model, '/model', 'ro'], [args.native + '/libopapi_nn.so',
                    '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_nn.so', 'ro']],
                env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                    VLLM_ASCEND_ENABLE_NZ='0' if fia else '2', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
                    VLLM_SERVER_DEV_MODE='1', TASK_QUEUE_ENABLE='0', OMP_NUM_THREADS='4', VLLM_HOST_IP='127.0.0.1',
                    GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120',
                    FAULT26_TRACE_ENABLED='1', FAULT26_TRACE_ALL_APIS='1', FAULT26_TRACE_CAPTURE_TENSORS='1',
                    FAULT26_WRITER_TOPOLOGY_ENABLED='1', FAULT26_TRACE_MAX_EVENTS='100000',
                    FAULT26_TRACE_LOG_PATH='/evidence/trace.jsonl', DCAT26_GROUP_V2_ALLOWED='1'), command=command,
                review=dict(experiment_id=name, provider=mechanism + (' v2 all_eligible once then live every_match' if coverage
                    else ' v2 one actual healthy-resolved ' + ('typed FIA' if fia else 'QMM') + ' site C1 effects (not C2)' if single_site
                    else ' v2 serial candidate experiment (not C2)'),
                    model=model, cann='8.5.1', soc='Ascend910B4/A2; verify actual Worker', physical_cards=[args.card], ranks=[0],
                    public_entry='WorkerFaultClient+FaultConfig(26,variant)' if args.worker_core else 'legacy callable experiment RPC',
                    public_core_new_source_requires_acceptance=args.worker_core,
                    previous_direct_entry_qualification_inherited=False,
                    worker='standard TP1 eager BF16; single fresh prefill/request; fixed64 KV blocks; no decode claim',
                    build_fingerprint=sha, source_commit=args.source_commit,
                    fia_typed_bridge=build.get('fia_typed_profile') if fia else None,
                    fia_stock_gate=('Two complete unmodified public-output byte samples; new owned-descriptor '
                        'Reference must match both completely before lossless/OFF qualification or any F. '
                        'Actual format 47 stays NCL; original descriptor reads stay inside GWS; every selected '
                        'execute uses an owned descriptor with complete current typed snapshots. No old reference reuse.'
                        if fia and build.get('fia_typed_profile') else None),
                    metadata_startup='Before importing Torch/vLLM, verify the immutable bundle and explicitly load '
                        'its missing split opapi with RTLD_NOW|RTLD_LOCAL; retain its handle. Then require all four '
                        'already-loaded counters with independent instances and exact mapped/dladdr identities.',
                    expected_hits=('C2: one per complete frozen eligible target; C3: every existing-target call in two fresh requests; '
                        'no subset, only after actual live C2' if coverage else
                        'exactly one F after the selected current site passes reference/lossless/OFF; no fallback or all-eligible claim'
                        if single_site else 0 if stage != 'effects'
                        else 'one per completely observed and independently lossless-qualified candidate; never C2'),
                    resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16, independent_events=32768,
                        max_targets=args.max_targets, planned_single_site_targets=1 if single_site else None,
                        max_targets_selects_subset=False, native_device_bytes=64 * 1024 * 1024, native_host_bytes=512 * 1024 * 1024,
                        fia_stock_host_bytes=512 * 1024 * 1024 if fia and build.get('fia_typed_profile') else 0),
                    recovery_plan=('Every original executes once. Two complete healthy censuses plus independent target '
                        'reference/lossless/OFF qualify the full set. Any unresolved, overlapping or over-budget set blocks '
                        'all group faults. Every Native call is independently observed before admission; frozen eligible failures '
                        'remain in the denominator. New post-propagation lifetimes pass unchanged and are separately recorded. '
                        'Stop ALL, drain ALL before ANY target release; exact restoration receipts for every target and '
                        'same-process full-vocab health plus zero Native/host/hooks precede the live C2 proof. C3 requires '
                        'that original issued proof object in the same Worker, owner, process lifetime and scope with a new '
                        'generation. Copying/relabeling proof is rejected before Native fault sealing. Failed backups/leases '
                        'remain recoverable; absent actual public pointer liveness refuses restoration writes. Container '
                        'restart is not OFF. All raw evidence remains on 124; raw statistics E5 is separate.' if coverage else
                        'Eight standard requests: two complete healthy censuses, selected reference/lossless/OFF, F/OFF, '
                        'then same-Worker health after Native release. Keep every actual Native call and module alias. '
                        'Choose the minimum actual twice-healthy resolved target key once; never truncate the inventory, '
                        'choose a user layer, skip failed qualification, or fall back to another target. The separate '
                        'single-site plan has one target; max-targets retains its legacy complete-set budget meaning and '
                        'does not select this site. Recheck actual object/formal/storage/allocator adjacency and consumer '
                        'on every admission; failure stops the plan, keeps its first receipt and uses existing '
                        'stop/drain/restore/readback/release only while public storage owners remain valid. '
                        'Final full-vocab health and zero Native/host/hooks are required. All raw evidence stays on 124; '
                        'C1 evidence awaits independent review, C2/C3/all-eligible remain false, and E5 is separate.' if single_site else
                        'Every original executes once, one explicit target per request. Retain full Native denominator '
                        'and every UNVERIFIED row. Budget overflow stops the whole resolved target plan, never chooses the first target. '
                        'Reference/lossless/OFF qualify every candidate before any F. Actual next consumer must retain the exact '
                        'public residual and writer output. Stop/drain/restore/readback/release retains failed backups and descriptors; '
                        'unproven pointer liveness refuses recovery writes and quarantines the Worker. Final full-vocab health runs '
                        'after Native resource release in the same Worker, then remove hooks and verify zero resources. '
                        'Container restart is not OFF. All raw evidence remains on 124; no C2/C3 claim.'))))
    with args.out.open('x') as stream:
        json.dump(cases, stream, indent=2)


if __name__ == '__main__':
    main()
