"""Write one reviewed queue case; never start Docker or acquire NPU devices."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'native', 'probe', 'build-sha', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model', default='/home/gyl/models/Qwen3-0.6B')
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--c3', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for value in (args.installed, args.native, args.probe, args.model):
        if not value.startswith('/') or any(c in value for c in ('\n', '\r', '\0', ',')):
            parser.error('Absolute immutable remote input paths required')
    if not re.fullmatch('[0-9a-f]{64}', args.build_sha) or not re.fullmatch('[0-9a-f]{7,40}', args.source_commit):
        parser.error('Actual native build SHA and source commit required')
    name = 'providers39-qwen-c2c3' if args.c3 else 'providers39-qwen-c2'
    native = PurePosixPath(args.native)
    case = dict(id=name, timeout=1800, image='quay.io/ascend/vllm-ascend:v0.18.0',
        mounts=[[args.installed, '/installed', 'ro'], [args.native, '/native', 'ro'],
            [str(native / 'libopapi_nn.so'), '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_nn.so', 'ro'],
            [args.probe, '/probe.py', 'ro'], [args.model, '/model', 'ro']],
        env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            VLLM_ASCEND_ENABLE_NZ='0', OMP_NUM_THREADS='4', VLLM_HOST_IP='127.0.0.1',
            GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120',
            VLLM_ALLOW_INSECURE_SERIALIZATION='1'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
            'export LD_PRELOAD=/native/libfault39_native_v1.so:/native/libopapi.so\n'
            'export LD_LIBRARY_PATH=/native:$LD_LIBRARY_PATH\n'
            'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
            'exec python3 /probe.py' + (' --statistics' if args.statistics else '') + (' --c3' if args.c3 else ''),
        review=dict(experiment_id=name, provider='39.matmul_range / exact-cache-v2', model=args.model,
            cann='8.5.1', soc='Ascend910B4/A2; recheck actual Worker', physical_cards=[args.card], ranks=[0],
            worker='new standard vLLM 0.18.0 TP1 eager; full root plus actual LogitsProcessor',
            build_fingerprint=args.build_sha, source_commit=args.source_commit,
            expected_hits='C2: independently discovered complete eligible target count, once each; '
                'C3 if requested: same targets every actual matching call in two bounded fresh-prefill requests, only after live C2 proof',
            resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16,
                dispatch_calls_per_request=20000, group_seconds=1200, max_native_storage_bytes=536870912,
                group_reserved_device_bytes=8 * 1024**3, group_reserved_host_bytes=8 * 1024**3),
            recovery_plan='Disable each exact native call and verify inflight/leases zero. Run all-target observed OFF and disabled OFF '
                'in same process; remove every owned hook; group stop-all/drain-all barrier then native clean; per-target '
                'output hashes and same-Worker complete logits restore required. On failure retain failed/quarantined evidence; '
                'root controller stops/removes only this owned temporary container. No NPU reuse marked healthy without recovery.'))
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
