"""Explicit single-NPU demo service lifecycle; never stops an existing service.

Run on the Linux host. New files live in --session; an existing local image is
required. start refuses an occupied NPU/project lease. stop only owns this named
container, and releases the lease only after Docker confirms it has stopped.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import socket
import subprocess
import time
import urllib.request

REFERENCE_SHA = '56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7'


def write(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def command(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)


def stop(session):
    d = json.loads((session / 'deployment.json').read_text())
    lock = Path(d['lock_owner_path'])
    owner = json.loads(lock.read_text())
    if owner.get('lease_id') != d['lease_id'] or owner.get('result_dir') != str(session):
        raise RuntimeError('NOT_OUR_LEASE')
    name = d['container_name']
    info = json.loads(command(['docker', 'inspect', name]))[0]
    if info['Config']['Labels'].get('ascend-fi-lease') != d['lease_id']:
        raise RuntimeError('NOT_OUR_CONTAINER')
    if info['State']['Running']:
        # Graceful server termination, followed by container-stop fallback.
        command(['docker', 'stop', '-t', '60', name])
    info = json.loads(command(['docker', 'inspect', name]))[0]
    if info['State']['Running']:
        raise RuntimeError('SERVICE_STILL_RUNNING; KEEP_LEASE')
    log = subprocess.run(['docker', 'logs', name], capture_output=True, text=True)
    with (session / 'service.log').open('x', encoding='utf-8') as f:
        f.write(log.stdout + log.stderr)
    final = {'container_stopped': True, 'fault_processes_terminated': True,
             'lease_released': True, 'time': dt.datetime.now(dt.timezone.utc).isoformat()}
    # Exact own lock files only, not recursive deletion; evidence remains.
    lock.unlink()
    lock.parent.rmdir()
    write(session / 'service_stopped.json', final)
    print(json.dumps(final))


def start(a):
    session = a.session.resolve()
    source = a.source_root.resolve()
    model = a.model.resolve()
    reference = session / 'reference/fake_requant_reference_extension.py'
    if hashlib.sha256(reference.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise RuntimeError('PINNED_REFERENCE_REQUIRED')
    if (session / 'deployment.json').exists():
        raise RuntimeError('NEW_SESSION_REQUIRED')
    info = command(['npu-smi', 'info'])
    if 'No running processes found' not in info:
        raise RuntimeError('NPU_BUSY_OR_PROCESS_STATE_UNPROVEN')
    image = json.loads(command(['docker', 'image', 'inspect', a.image]))[0]
    devices = (f'/dev/davinci{a.device}', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc')
    for dev in devices:
        if not Path(dev).exists():
            raise RuntimeError('DEVICE_MISSING:' + dev)
    if not (model / 'config.json').is_file() or not (source / 'src/ascend_fi/inspect/experiment_extension.py').is_file():
        raise RuntimeError('MODEL_OR_SOURCE_MISSING')
    with socket.socket() as s:
        s.bind(('127.0.0.1', a.port))
    lock = a.lock_root.resolve() / 'NPU_LOCK'
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir()  # atomic; never removes another owner's lease
    lease = secrets.token_hex(16)
    name = 'demoncat-qwen06b-' + lease[:10]
    owner = {'owner': 'PRIVATE_MODEL_DEMO', 'lease_id': lease,
        'result_dir': str(session), 'device_ids': [a.device],
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    write(lock / 'owner.json', owner)
    token = secrets.token_urlsafe(40)
    with (session / 'access.token').open('x') as f:
        f.write(token)
    os.chmod(session / 'access.token', 0o600)
    arm = {'mode': 'REFERENCE_REQUANT', 'module_regex': r'(?:^|\.)mlp\.down_proj$',
        'layer_filter': None, 'rank_filter': [0], 'tensor_selector': 'tensor_index',
        'tensor_index': 0, 'scale_provider': 'tensor_absmax', 'fixed_scale': None,
        'runtime_scale_attr': None, 'scale_multiplier': 20.0, 'max_hits': 4096,
        'interval': 1, 'eps': 1e-12}
    deployment = {'base_url': f'http://127.0.0.1:{a.port}', 'tp_size': 1,
        'device_ids': [a.device], 'logical_rank_mapping': {'0': a.device},
        'served_model_name': 'qwen3-06b-demo', 'model_path': str(model),
        'reference_path': str(reference), 'source_root': str(source),
        'lock_owner_path': str(lock / 'owner.json'), 'owner': owner['owner'],
        'lease_id': lease, 'container_name': name, 'reference_arm': arm,
        'image_id': image['Id'], 'reference_sha256': REFERENCE_SHA,
        'model_config': json.loads((model / 'config.json').read_text())}
    write(session / 'deployment.json', deployment)
    write(session / 'reference_arm.json', arm)
    for p in ('cache', 'tmp'):
        (session / p).mkdir(exist_ok=True)
    serve = ['vllm', 'serve', str(model), '--served-model-name', deployment['served_model_name'],
        '--host', '127.0.0.1', '--port', str(a.port), '--tensor-parallel-size', '1',
        '--distributed-executor-backend', 'mp', '--enforce-eager', '--dtype', 'bfloat16',
        '--max-model-len', '4096', '--max-num-seqs', '1', '--gpu-memory-utilization', '0.40',
        '--no-enable-prefix-caching', '--no-async-scheduling', '--seed', '42',
        '--worker-extension-cls', 'ascend_fi.inspect.experiment_extension.ExplorationWorkerExtension',
        '--middleware', 'ascend_fi.inspect.private_demo_auth.PrivateDemoAuth']
    argv = ['docker', 'run', '-d', '--name', name, '--runtime=runc', '--net=host', '--shm-size=4g',
        '--label', 'ascend-fi-lease=' + lease, '--log-opt', 'max-size=5m', '--log-opt', 'max-file=2']
    for dev in devices:
        argv += ['--device', dev + ':' + dev]
    for src, dst, mode in ((str(session), str(session), 'rw'), (str(source), str(source), 'ro'),
            (str(lock.parent), str(lock.parent), 'ro'), (str(model), str(model), 'ro'),
            ('/usr/local/Ascend/driver', '/usr/local/Ascend/driver', 'ro'),
            ('/usr/local/dcmi', '/usr/local/dcmi', 'ro'),
            ('/usr/local/bin/npu-smi', '/usr/local/bin/npu-smi', 'ro'),
            ('/etc/ascend_install.info', '/etc/ascend_install.info', 'ro'),
            (str(session / 'tmp'), '/tmp', 'rw')):
        argv += ['-v', f'{src}:{dst}:{mode}']
    env = {'VLLM_SERVER_DEV_MODE': '1', 'ASCEND_RT_VISIBLE_DEVICES': '0',
        'ASCEND_FI_REFERENCE_SHA256': REFERENCE_SHA, 'VLLM_API_KEY': token,
        'PYTHONPATH': str(source / 'src') + ':' + str(reference.parent),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'XDG_CACHE_HOME': str(session / 'cache'), 'TRITON_CACHE_DIR': str(session / 'cache/triton')}
    for k, v in env.items():
        argv += ['-e', f'{k}={v}']
    argv += ['--entrypoint', '/bin/bash', image['Id'], '-lc',
        'source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec ' + ' '.join(shlex.quote(x) for x in serve)]
    redacted = ['VLLM_API_KEY=<session/access.token>' if x.startswith('VLLM_API_KEY=') else x for x in argv]
    write(session / 'launch_command.json', {'argv': redacted, 'npu_before': info})
    try:
        command(argv)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            state = json.loads(command(['docker', 'inspect', name]))[0]['State']
            if not state['Running']:
                raise RuntimeError('SERVICE_EXITED:' + json.dumps(state))
            try:
                with urllib.request.urlopen(deployment['base_url'] + '/health', timeout=3) as r:
                    if r.status == 200:
                        write(session / 'service_ready.json', {'healthy': True, 'container': name})
                        print(json.dumps({'healthy': True, 'session': str(session), 'container': name}), flush=True)
                        return
            except OSError:
                pass
            time.sleep(3)
        raise RuntimeError('STARTUP_TIMEOUT')
    except BaseException:
        # Stop only our newly-created container. Preserve all failure artifacts.
        try:
            stop(session)
        except Exception as e:
            print('KEEP_LEASE_UNTIL_CLEANUP_PROVEN:' + repr(e), flush=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'stop', 'status'])
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--device', type=int, default=5)
    p.add_argument('--port', type=int, default=18131)
    p.add_argument('--image', default='d519927a2b4d')
    p.add_argument('--lock-root', type=Path, default=Path('/home/gyl/demoncat_coordination'))
    a = p.parse_args()
    if a.action == 'start':
        if not a.source_root or not a.model:
            p.error('start requires --source-root and --model')
        start(a)
    elif a.action == 'stop':
        stop(a.session.resolve())
    else:
        d = json.loads((a.session / 'deployment.json').read_text())
        print(command(['docker', 'inspect', d['container_name'], '--format', '{{json .State}}']))


if __name__ == '__main__':
    main()
