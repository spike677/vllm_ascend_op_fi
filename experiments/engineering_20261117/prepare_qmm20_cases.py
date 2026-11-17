"""Freeze one standard Worker QMM trial; no native build or device acquisition."""
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'native', 'probe'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model', default='/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--baseline-control', action='store_true')
    parser.add_argument('--memory', type=float, default=.55)
    parser.add_argument('--mapped-libraries', type=Path)
    args = parser.parse_args()
    for value in (args.installed, args.native, args.probe, args.model):
        if not value.startswith('/') or any(c in value for c in ('\n', '\r', '\x00')):
            parser.error('Absolute remote paths required')
    if not .1 <= args.memory <= .8:
        parser.error('Bounded Worker memory fraction required')
    native = PurePosixPath(args.native)
    case = dict(id='fault20-qmm-' + ('baseline' if args.baseline_control else 'fault'),
        timeout=1800, image='quay.io/ascend/vllm-ascend:v0.18.0',
        mounts=[[args.installed, '/installed', 'ro'], [args.native, '/native', 'ro'],
                [args.probe, '/probe.py', 'ro'], [args.model, '/model', 'ro']],
        env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                 VLLM_ASCEND_ENABLE_NZ='2', OMP_NUM_THREADS='4', VLLM_HOST_IP='127.0.0.1',
                 GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120',
                 VLLM_ALLOW_INSECURE_SERIALIZATION='1', DEMONCAT_F20_REQUIRE_TYPED_GWS='1',
                 DEMONCAT_F20_DIAGNOSTICS_PATH='/evidence/current-call.jsonl'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
                'export LD_PRELOAD=/native/libdcat20_qmm.so:/native/libopapi.so\n'
                'export LD_LIBRARY_PATH=/native:$LD_LIBRARY_PATH\n'
                'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
                'exec python3 /probe.py --mechanism 20.buffer_substitute '
                '--operator aclnnQuantMatmulWeightNz --memory ' + str(args.memory)
                + (' --statistics' if args.statistics else '')
                + (' --baseline-control' if args.baseline_control else ''))
    for name in ('libopapi_nn.so', 'libopapi_math.so'):
        case['mounts'].append([str(native / name), '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/' + name, 'ro'])
    if args.mapped_libraries:
        snapshot = json.loads(args.mapped_libraries.read_text())
        if (snapshot.get('schema') != 'demoncat.private_mapped_libraries.v1'
                or snapshot.get('status') != 'SNAPSHOT_ONLY_REQUIRES_ACTUAL_MAPPED_IDENTITY'
                or snapshot.get('actual_npu_calls') != 0
                or not isinstance(snapshot.get('image_id'), str)
                or not re.fullmatch('sha256:[0-9a-f]{64}', snapshot['image_id'])
                or not isinstance(snapshot.get('libraries'), list)
                or not 1 <= len(snapshot['libraries']) <= 16):
            raise ValueError('Reviewed identical-byte image snapshot required')
        targets = {row[1] for row in case['mounts']}
        for row in snapshot['libraries']:
            source = Path(row['source'])
            if (row['destination'] in targets or not row['destination'].startswith('/')
                    or source.is_symlink() or not source.is_file()
                    or source.stat().st_size != row['bytes']):
                raise ValueError('Mapped library snapshot changed or mount conflicts')
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('Mapped library snapshot bytes changed')
            case['mounts'].append([str(source), row['destination'], 'ro'])
            targets.add(row['destination'])
        case['image'] = snapshot['image_id']
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
