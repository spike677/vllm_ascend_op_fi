"""Write one reviewed QMM group case for the root queue; never acquire devices."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'native', 'probe', 'build-sha', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model', default='/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--c3', action='store_true')
    parser.add_argument('--mapped-libraries', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (not re.fullmatch('[0-9a-f]{64}', args.build_sha)
            or not re.fullmatch('[0-9a-f]{7,40}', args.source_commit)
            or any(not value.startswith('/') or any(c in value for c in ('\n', '\r', '\0', ','))
                   for value in (args.installed, args.native, args.probe, args.model))):
        parser.error('Actual build SHA, source commit and immutable absolute inputs required')
    native = PurePosixPath(args.native)
    name = 'providers20-deepseek-c2c3' if args.c3 else 'providers20-deepseek-c2'
    case = dict(id=name, timeout=1800, image='quay.io/ascend/vllm-ascend:v0.18.0',
        mounts=[[args.installed, '/installed', 'ro'], [args.native, '/native', 'ro'],
                [args.probe, '/probe.py', 'ro'], [args.model, '/model', 'ro']],
        env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            VLLM_ASCEND_ENABLE_NZ='2', OMP_NUM_THREADS='4', VLLM_HOST_IP='127.0.0.1',
            GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120',
            TASK_QUEUE_ENABLE='0', VLLM_SERVER_DEV_MODE='1', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
            DEMONCAT_F20_REQUIRE_TYPED_GWS='1', DEMONCAT_F20_DIAGNOSTICS_PATH='/evidence/current-call.jsonl'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
            'export LD_PRELOAD=/native/libdcat20_qmm.so:/native/libopapi.so\n'
            'export LD_LIBRARY_PATH=/native:$LD_LIBRARY_PATH\n'
            'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
            'exec python3 /probe.py' + (' --statistics' if args.statistics else '') + (' --c3' if args.c3 else ''),
        review=dict(experiment_id=name, provider='20.buffer_substitute / exact-target-group-v3', model=args.model,
            cann='8.5.1', soc='Ascend910B4/A2; recheck actual Worker', physical_cards=[args.card], ranks=[0],
            worker='new standard vLLM 0.18.0 TP1 eager BF16; fixed natural Hello. prompt <=4 actual tokens; fresh KV each request',
            build_fingerprint=args.build_sha, source_commit=args.source_commit,
            expected_hits='C2: all independently observed and losslessly qualified native QMM targets once; '
                'C3 only after live C2: same target set on every call in two bounded requests; structural count is not qualification',
            resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16, native_targets=512,
                public_api_calls=4096, scoped_native_calls=16384, group_seconds=1200,
                native_x1_max_bytes=65536, group_reserved_device_bytes=8 * 1024**3, group_reserved_host_bytes=8 * 1024**3),
            recovery_plan='Every native scoped continuation executes exactly once and releases temporary allocations after '
                'synchronization. Observe all original targets in OFF then a disabled OFF in the same Worker; exact per-target '
                'reference input/output and full-model logits equality required. Stop all admission, drain all, release callbacks '
                'only after the whole barrier; remove all owned public hooks. Failed or missing target remains C2/C3 false. '
                'Uncertain native failure retains quarantine; root disposes only its owned temporary container, never labels a restart as OFF.'))
    for library in ('libopapi_nn.so', 'libopapi_math.so'):
        case['mounts'].append([str(native / library), '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/' + library, 'ro'])
    if args.mapped_libraries:
        snapshot = json.loads(args.mapped_libraries.read_text())
        if (snapshot.get('schema') != 'demoncat.private_mapped_libraries.v1'
                or snapshot.get('status') != 'SNAPSHOT_ONLY_REQUIRES_ACTUAL_MAPPED_IDENTITY'
                or snapshot.get('actual_npu_calls') != 0
                or not re.fullmatch('sha256:[0-9a-f]{64}', snapshot.get('image_id', ''))
                or not isinstance(snapshot.get('libraries'), list) or not 1 <= len(snapshot['libraries']) <= 16):
            raise ValueError('Reviewed identical-byte image snapshot required')
        destinations = {row[1] for row in case['mounts']}
        for row in snapshot['libraries']:
            source = Path(row['source'])
            if (row['destination'] in destinations or not row['destination'].startswith('/')
                    or source.is_symlink() or not source.is_file() or source.stat().st_size != row['bytes']
                    or hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']):
                raise ValueError('Mapped library snapshot changed or mount conflicts')
            case['mounts'].append([str(source), row['destination'], 'ro'])
            destinations.add(row['destination'])
        case['image'] = snapshot['image_id']
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
