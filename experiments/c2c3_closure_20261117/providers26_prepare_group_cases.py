"""Prepare six separate root-reviewed v2 inventory/lossless/effects candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'native', 'probe', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--max-targets', type=int, default=8)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--stages', nargs='+', choices=('inventory', 'lossless', 'effects'), default=['inventory'])
    parser.add_argument('--mechanisms', nargs='+', choices=('26.residual_stomp', '26.fia_destination'),
                        default=['26.residual_stomp', '26.fia_destination'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
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
    cases = []
    for mechanism in args.mechanisms:
        fia = mechanism == '26.fia_destination'
        model = '/home/fault39-integration-20261116-a1/models-smollm2-135m' if fia else '/home/haoran/models/DeepSeek-V2-Lite-W8A8'
        for stage in args.stages:
            name = 'providers26-v2-' + ('fia-smol-' if fia else 'qmm-deepseek-') + stage
            command = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
                'export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend\n'
                'export LD_PRELOAD=/native/libdcat26_group_v2.so:/native/libfault26_readonly_trace_abi0.so:'
                '/native/libfault26_fia_opapi_proxy.so:/native/libopapi.so:/native/libcust_opapi.so\n'
                'export LD_LIBRARY_PATH=/native:/native/real:$LD_LIBRARY_PATH\n'
                'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
                'exec python3 /probe.py --mechanism ' + mechanism + ' --manifest-sha256 ' + sha + ' --stage ' + stage
                + ' --max-targets ' + str(args.max_targets) + ' --memory ' + ('.25' if fia else '.55')
                + (' --statistics' if args.statistics else ''))
            cases.append(dict(id=name, timeout=1800, image=build['actual_image_id'],
                mounts=[[args.installed, '/installed', 'ro'], [args.native, '/native', 'ro'], [args.probe, '/probe.py', 'ro'],
                    [model, '/model', 'ro'], [args.native + '/libopapi_nn.so',
                    '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_nn.so', 'ro']],
                env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                    VLLM_ASCEND_ENABLE_NZ='0' if fia else '2', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
                    VLLM_SERVER_DEV_MODE='1', TASK_QUEUE_ENABLE='0', OMP_NUM_THREADS='4', VLLM_HOST_IP='127.0.0.1',
                    GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120',
                    FAULT26_TRACE_ENABLED='1', FAULT26_TRACE_ALL_APIS='1', FAULT26_TRACE_CAPTURE_TENSORS='1',
                    FAULT26_WRITER_TOPOLOGY_ENABLED='1', FAULT26_TRACE_MAX_EVENTS='100000',
                    FAULT26_TRACE_LOG_PATH='/evidence/trace.jsonl', DCAT26_GROUP_V2_ALLOWED='1'), command=command,
                review=dict(experiment_id=name, provider=mechanism + ' v2 serial candidate experiment (not C2)',
                    model=model, cann='8.5.1', soc='Ascend910B4/A2; verify actual Worker', physical_cards=[args.card], ranks=[0],
                    worker='standard TP1 eager BF16; single fresh prefill/request; fixed64 KV blocks; no decode claim',
                    build_fingerprint=sha, source_commit=args.source_commit,
                    expected_hits=0 if stage != 'effects' else 'one per completely observed and independently lossless-qualified candidate; never C2',
                    resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16, independent_events=32768,
                        max_targets=args.max_targets, native_device_bytes=64 * 1024 * 1024, native_host_bytes=512 * 1024 * 1024),
                    recovery_plan='Every original executes once, one explicit target per request. Retain full Native denominator '
                        'and every UNVERIFIED row. Budget overflow stops the whole resolved target plan, never chooses the first target. '
                        'Reference/lossless/OFF qualify every candidate before any F. Actual next consumer must retain the exact '
                        'public residual and writer output. Stop/drain/restore/readback/release retains failed backups and descriptors; '
                        'unproven pointer liveness refuses recovery writes and quarantines the Worker. Final full-vocab health runs '
                        'after Native resource release in the same Worker, then remove hooks and verify zero resources. '
                        'Container restart is not OFF. All raw evidence remains on 124; no C2/C3 claim.')))
    with args.out.open('x') as stream:
        json.dump(cases, stream, indent=2)


if __name__ == '__main__':
    main()
