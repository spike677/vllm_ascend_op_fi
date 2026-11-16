"""Run bounded Worker Extension cases on 124 in dedicated containers."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'fault39/tools'))
from deployment_profile import verify_build
from run_124 import idle_gate

IMAGE = 'quay.io/ascend/vllm-ascend:v0.18.0'
MODEL = Path('/home/gyl/models/Qwen3-0.6B')
LOCK = '/var/tmp/fault22_npu_2_5.lock'
EXPERIMENT = 'experiments/worker_extension_20261116'


def now():
    return datetime.now(timezone.utc).isoformat()


def mount(path, target, readonly=True):
    return ['--mount', f'type=bind,src={path.resolve(strict=True)},dst={target}'
            + (',readonly' if readonly else '')]


def verify_source():
    manifest = ROOT / 'worker_extension_source_manifest.json'
    raw = manifest.read_bytes()
    rows = json.loads(raw)
    for name, expected in rows.items():
        file = (ROOT / name).resolve(strict=True)
        if not file.is_relative_to(ROOT) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'SOURCE_MANIFEST_MISMATCH: {name}')
    return hashlib.sha256(raw).hexdigest()


def run_case(args, name, directory, card_ids, build, previous, receipt, save):
    native = verify_build(build) if build else None
    case = {'name': name, 'start': now(), 'cards': card_ids}
    receipt['cases'].append(case)
    case['gate'] = idle_gate(card_ids)
    save()
    if not case['gate']['idle']:
        case['status'] = 'DEVICE_BUSY'
        save()
        return False

    directory.mkdir()
    unique = f'worker-extension-{uuid.uuid4().hex[:12]}'
    label = 'dcat.worker_extension=' + unique
    command = ['docker', 'run', '--name', unique, '--label', label, '--network', 'none',
               '--memory', '48g', '--cpus', '10', '--pids-limit', '1024', '--shm-size', '4g']
    for dev in (*(f'davinci{n}' for n in card_ids), 'davinci_manager', 'devmm_svm', 'hisi_hdc'):
        command += ['--device', '/dev/' + dev]
    for path, target, ro in ((ROOT, '/source', True), (args.wheel, '/pkg/' + args.wheel.name, True),
                             (directory, '/evidence', False), (MODEL, '/model', True),
                             (Path('/usr/local/Ascend/driver'), '/usr/local/Ascend/driver', True),
                             (Path('/usr/local/dcmi'), '/usr/local/dcmi', True),
                             (Path('/dev/null'), '/etc/ld.so.preload', True)):
        command += mount(path, target, ro)
    toolkit_script = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    preload = '-'
    flags = []
    if native:
        toolkit_script = native['environment_script']['path']
        adapter = native.get('adapter', 'matmul-cache-abi0-v1')
        library = {'fault20-isolated-cache-abi0-v1': 'libdcat20_qmm.so',
                   'matmul-cache-abi0-v1': 'libfault39_native_v1.so'}[adapter]
        preflight = subprocess.run([
            'docker', 'run', '--rm', '--runtime=runc', '--network', 'none',
            '--memory', '4g', '--cpus', '2', '-e', 'ASCEND_VISIBLE_DEVICES=void',
            *mount(ROOT, '/source'), *mount(build, '/native'),
            '--entrypoint', '/bin/bash', args.image, '-ec',
            'source "$1" >&2; exec python3 /source/fault39/tools/deployment_profile.py '
            '--native /native --verify-installed --verify-environment',
            'worker-extension-build-check', toolkit_script,
        ], capture_output=True, text=True, check=True, timeout=120)
        case['installed_preflight'] = json.loads(preflight.stdout)
        preload = '/native/' + library + ':/native/libopapi.so'
        command += mount(build, '/native')
        if native['proxy_layout'] == 'main_and_nn':
            command += mount(build / 'libopapi_nn.so', native['toolkit'] + '/lib64/libopapi_nn.so')
        case['native_profile_sha256'] = native['profile_sha256']
    if previous:
        command += mount(previous, '/previous.json')
        flags += ['--previous', '/previous.json']
    for key, value in {
        'VLLM_SERVER_DEV_MODE': '1', 'VLLM_ENABLE_V1_MULTIPROCESSING': '1',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'PYTHONDONTWRITEBYTECODE': '1',
        'TASK_QUEUE_ENABLE': '0', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '2',
    }.items():
        command += ['-e', f'{key}={value}']
    entry = r'''
set -e
source "$1"
unset ASCEND_RT_VISIBLE_DEVICES
preload=$2
wheel=$3
shift 3
python3 -m pip install --disable-pip-version-check --no-index --no-deps --target /tmp/worker-extension "$wheel"
export PYTHONPATH="/tmp/worker-extension:/source:/source/fault39/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$preload" != '-' ]; then
    export LD_PRELOAD="$preload"
    export LD_LIBRARY_PATH="/native:${LD_LIBRARY_PATH}"
fi
exec python3 "$@"
'''
    command += ['--entrypoint', '/bin/bash', args.image, '-c', entry, 'worker-extension',
                toolkit_script, preload, '/pkg/' + args.wheel.name,
                f'/source/{EXPERIMENT}/probe.py', '--tp', str(len(card_ids)),
                '--case', name, '--out', '/evidence/result.json', *flags]
    case['container'] = unique
    case['launch_gate'] = idle_gate(card_ids)
    if not case['launch_gate']['idle']:
        case['status'] = 'DEVICE_BUSY'
        save()
        return False
    case['status'] = 'RUNNING'
    save()
    print(json.dumps({'event': 'CASE_START', 'name': name, 'cards': card_ids}), flush=True)
    try:
        with (directory / 'process.log').open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            try:
                case['exit_code'] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                case['timeout'] = True
    finally:
        inspect = subprocess.run(['docker', 'inspect', unique], capture_output=True, text=True, timeout=30)
        if inspect.returncode == 0:
            info = json.loads(inspect.stdout)[0]
            if info['Config']['Labels'].get('dcat.worker_extension') != unique:
                raise RuntimeError('CONTAINER_OWNERSHIP_CHANGED')
            case['image_id'] = info['Image']
            case['container_state'] = info['State']
            if info['State']['Running']:
                subprocess.run(['docker', 'stop', '--time', '20', unique], check=True, timeout=60)
            subprocess.run(['docker', 'rm', unique], check=True, timeout=30)
        result = directory / 'result.json'
        if result.is_file():
            case['result_sha256'] = hashlib.sha256(result.read_bytes()).hexdigest()
        case['end'] = now()
        case['status'] = ('COMPLETED' if case.get('exit_code') == 0 and result.is_file() else 'FAILED')
        save()
    print(json.dumps({'event': 'CASE_END', 'name': name, 'status': case['status']}), flush=True)
    return case['status'] == 'COMPLETED'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--cards', nargs='+', type=int, choices=(2, 5), required=True)
    parser.add_argument('--cases', nargs='+', choices=('identity', 'restart', 'fault20', 'fault39'), required=True)
    parser.add_argument('--fault20', type=Path)
    parser.add_argument('--fault39', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if len(set(args.cards)) != len(args.cards) or len(set(args.cases)) != len(args.cases):
        parser.error('cards and cases must be distinct')
    source_sha = verify_source()
    args.attempt.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 'demoncat.worker_extension.runner.v1', 'started': now(), 'cases': [],
               'source_manifest_sha256': source_sha, 'status': 'PREFLIGHT',
               'wheel_sha256': hashlib.sha256(args.wheel.read_bytes()).hexdigest()}

    def save():
        target = args.attempt / 'controller.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(receipt, indent=2) + '\n')
        temp.replace(target)

    save()
    with open(LOCK, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            receipt['status'] = 'RESOURCE_LOCK_BUSY'
            save()
            return 3
        receipt['status'] = 'RUNNING'
        save()
        previous = args.previous
        try:
            for case in args.cases:
                build = {'fault20': args.fault20, 'fault39': args.fault39}.get(case)
                if case.startswith('fault') and build is None:
                    parser.error(f'{case} requires a native build path')
                directory = args.attempt / case
                if not run_case(args, case, directory, args.cards, build,
                                previous if case == 'restart' else None, receipt, save):
                    receipt['status'] = receipt['cases'][-1]['status']
                    return 2
                previous = directory / 'result.json'
            receipt['status'] = 'COMPLETED'
            return 0
        except Exception as error:
            receipt['status'] = 'FAILED'
            receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            receipt['finished'] = now()
            save()


if __name__ == '__main__':
    raise SystemExit(main())
