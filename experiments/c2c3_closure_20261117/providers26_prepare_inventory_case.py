"""Prepare one root-queued natural writer census, with read-only inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'native', 'probe', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--mechanism', choices=('26.residual_stomp', '26.fia_destination'), required=True)
    parser.add_argument('--model')
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    fia = args.mechanism == '26.fia_destination'
    args.model = args.model or ('/home/fault39-integration-20261116-a1/models-smollm2-135m' if fia
                               else '/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    if (not re.fullmatch('[0-9a-f]{7,40}', args.source_commit)
            or any(not value.startswith('/') or any(c in value for c in ('\n', '\r', '\0', ','))
                   for value in (args.installed, args.native, args.probe, args.model))):
        parser.error('Actual source commit and immutable absolute inputs required')
    manifest = Path(args.native) / 'build.json'
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    build = json.loads(manifest.read_text())
    if build.get('schema') != 'demoncat.writer26.deployment.v1' or build.get('actual_npu_calls') != 0:
        raise ValueError('Fresh SDK-built and staged native26 bundle required')
    for name, sha in build['files'].items():
        path = (Path(args.native) / name).resolve(strict=True)
        if not path.is_relative_to(Path(args.native).resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Native build file changed: ' + name)
    name = 'providers26-' + ('fia-smol' if fia else 'qmm-deepseek') + '-native-inventory'
    case = dict(id=name, timeout=1800, image='quay.io/ascend/vllm-ascend:v0.18.0',
        mounts=[[args.installed, '/installed', 'ro'], [args.native, '/native', 'ro'],
            [args.probe, '/probe.py', 'ro'], [args.model, '/model', 'ro'],
            [args.native + '/libopapi_nn.so', '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_nn.so', 'ro']],
        env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            VLLM_ASCEND_ENABLE_NZ='0' if fia else '2', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
            TASK_QUEUE_ENABLE='0', OMP_NUM_THREADS='4', VLLM_HOST_IP='127.0.0.1', GLOO_SOCKET_IFNAME='lo',
            HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120', FAULT26_TRACE_ENABLED='1',
            FAULT26_TRACE_ALL_APIS='1', FAULT26_TRACE_CAPTURE_TENSORS='1', FAULT26_WRITER_TOPOLOGY_ENABLED='1',
            FAULT26_TRACE_MAX_EVENTS='100000', FAULT26_TRACE_LOG_PATH='/evidence/trace.jsonl',
            FAULT26_QMM_UNDERFLOW_ALLOWED='1', FAULT26_FIA_WRITER_PROXY_ALLOWED='1',
            FAULT26_QMM_UNDERFLOW_LOG_PATH='/evidence/native-writer.jsonl', FAULT26_FIA_WRITER_LOG_PATH='/evidence/native-writer.jsonl'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
            'export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend\n'
            'export LD_PRELOAD=/native/libfault26_readonly_trace_abi0.so:/native/libfault26_fia_writer_proxy_abi0.so:'
                '/native/libfault26_qmm_output_underflow_abi0.so:/native/libfault26_fia_opapi_proxy.so:/native/libopapi.so:/native/libcust_opapi.so\n'
            'export LD_LIBRARY_PATH=/native:/native/real:$LD_LIBRARY_PATH\n'
            'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
            'exec python3 /probe.py --mechanism ' + args.mechanism + ' --manifest-sha256 ' + manifest_sha
                + ' --memory ' + ('.25' if fia else '.55'),
        review=dict(experiment_id=name, provider=args.mechanism + ' independent native writer/producer/consumer inventory',
            model=args.model, cann='8.5.1', soc='Ascend910B4/A2; verify actual Worker', physical_cards=[args.card], ranks=[0],
            worker='new standard TP1 eager BF16; two identical single-request fresh prefills, fixed64 KV blocks; no decode coverage',
            build_fingerprint=manifest_sha, source_commit=args.source_commit, expected_hits=0,
            resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16, independent_events=32768),
            recovery_plan='No native fault arm settings are supplied. Every original writer/producer/consumer executes once. '
                'Retain each actual residual only through its next real native consumer and clear terminal references at the '
                'request boundary. Preserve unresolved/missing/unowned native writers and explicit overflow. Verify two complete '
                'full-vocab healthy hashes, remove only owned callbacks/hooks, require no native pending resources, then generate '
                'again in the same Worker. No native qualification or C1/C2/C3 claim. Root removes only its owned temporary '
                'container after failure; process restart is never an OFF proof.'))
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
