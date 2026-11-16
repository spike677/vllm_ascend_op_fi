"""Bounded, fail-closed 124 runner. Never stops foreign workloads.

Run on the host, not in a device container. One shared lock covers cards 2/5.
The lock coordinates cooperating jobs; it is not host isolation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

IMAGE = 'quay.io/ascend/vllm-ascend:v0.18.0'
LABEL = 'dcat.task=fault39-20261116'


def command(args, timeout=20):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def dedicated_atk_reservations(container, cards):
    """Conservatively reserve mapped cards for running, dedicated ATK jobs.

    An idle persistent container's device permissions alone are not a
    reservation. A running ptq_ job, however, can retain driver reservation
    before or between visible compute processes; do not compete with it.
    """
    name = container.get('Name', '')
    if (not isinstance(name, str) or not name.lstrip('/').startswith('ptq_')
            or container.get('State', {}).get('Running') is not True):
        return []
    selected = {f'/dev/davinci{card}': card for card in cards if card in (2, 5)}
    mapped = {dev.get('PathOnHost') for dev in container.get('HostConfig', {}).get('Devices') or []}
    return [{'card': selected[path], 'container': name, 'container_id': container.get('Id'),
             'device': path, 'reason': 'DEDICATED_ATK_CONTAINER_RESERVATION',
             'basis': 'running_ptq_name_and_selected_device_mapping'}
            for path in sorted(mapped.intersection(selected))]


def idle_gate(cards):
    report = {'time': datetime.now(timezone.utc).isoformat(), 'cards': cards, 'blockers': [], 'device_access_grants': []}
    report['npu_smi'] = command(['npu-smi', 'info'])
    rows = report['npu_smi'].splitlines()
    if 'Process id' not in report['npu_smi'] or 'Health' not in report['npu_smi']:
        raise RuntimeError('UNRECOGNIZED_NPU_SMI')
    for card in cards:
        health = next((r for r in rows if re.search(r'\|\s*' + str(card) + r'\s+910', r)), '')
        if not health or not re.search(r'\|\s*OK\s*\|', health):
            report['blockers'].append({'card': card, 'reason': 'HEALTH_UNPROVEN'})
        for row in rows:
            found = re.match(r'\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|', row)
            if found and int(found[1]) == card:
                report['blockers'].append({'card': card, 'pid': int(found[3]), 'reason': 'NPU_PROCESS'})
    devices = {Path('/dev/davinci' + str(c)).stat().st_rdev: c for c in cards}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    info = fd.stat()
                    if stat.S_ISCHR(info.st_mode) and info.st_rdev in devices:
                        report['blockers'].append({'card': devices[info.st_rdev], 'pid': int(proc.name), 'reason': 'OPEN_DEVICE'})
                        break
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    ids = command(['docker', 'ps', '-q']).split()
    for item in json.loads(command(['docker', 'inspect', *ids])) if ids else []:
        report['blockers'].extend(dedicated_atk_reservations(item, cards))
        for dev in item.get('HostConfig', {}).get('Devices') or []:
            path = dev.get('PathOnHost', '')
            if path in ['/dev/davinci' + str(c) for c in cards]:
                report['device_access_grants'].append({'container': item['Name'], 'device': path, 'kind': 'DOCKER_DEVICE_PERMISSION'})
    # OCI device permissions are inventory, not proof of an active reservation.
    # Actual NPU processes and open char-device handles above block execution.
    for config in Path('/run/containerd/io.containerd.runtime.v2.task').glob('*/*/config.json'):
        item = json.loads(config.read_text())
        for dev in item.get('linux', {}).get('devices', []):
            if dev.get('type') == 'c' and os.makedev(dev.get('major', 0), dev.get('minor', 0)) in devices:
                report['device_access_grants'].append({'task': config.parent.name, 'device': dev.get('path'), 'kind': 'OCI_DEVICE_PERMISSION'})
    report['idle'] = not report['blockers']
    return report


def verify_source(source):
    manifest = json.loads((source / 'source_manifest.json').read_text())
    for item in manifest['files']:
        path = (source / item['path']).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('SOURCE_MANIFEST_MISMATCH')
    return hashlib.sha256((source / 'source_manifest.json').read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cards', nargs='+', type=int, choices=[2, 5], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/home/gyl/models/Qwen3-0.6B'))
    p.add_argument('--lossless-receipt', type=Path)
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    cases = json.loads(a.cases.read_text())
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8 or len(set(a.cards)) != len(a.cards):
        p.error('Require 1..8 cases and distinct cards')
    for case in cases:
        if not re.fullmatch(r'[a-z0-9_-]{1,48}', case['id']) or case['tool'] not in ('native_probe.py', 'vllm_validate.py'):
            p.error('Unsupported case')
        if any(not isinstance(x, str) for x in case['args']):
            p.error('Arguments must be strings')
    a.attempt.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 'fault39.runner.v1', 'source_manifest_sha256': verify_source(a.source),
               'image': IMAGE, 'cards': a.cards, 'cases': [], 'status': 'PREPARING'}
    def save():
        (a.attempt / 'controller.json').write_text(json.dumps(receipt, indent=2))
    save()
    with open('/var/tmp/fault22_npu_2_5.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            receipt['status'] = 'RESOURCE_LOCK_BUSY'; save(); return 3
        for case in cases:
            gate = idle_gate(a.cards)
            item = {'id': case['id'], 'gate': gate}; receipt['cases'].append(item); save()
            if not gate['idle']:
                receipt['status'] = 'DEVICE_BUSY'; save(); print(json.dumps({'status': receipt['status'], 'blockers': gate['blockers']}), flush=True); return 3
            out = a.attempt / case['id']; out.mkdir()
            name = 'fault39-' + case['id'] + '-' + str(os.getpid())
            args = ['docker', 'run', '--name', name, '--label', LABEL, '--network', 'none',
                    '--memory', '32g', '--cpus', '8', '--pids-limit', '1024', '--shm-size', '2g']
            for card in a.cards:
                args += ['--device', '/dev/davinci' + str(card)]
            for dev in ('davinci_manager', 'devmm_svm', 'hisi_hdc'):
                args += ['--device', '/dev/' + dev]
            for src, dst, ro in [(a.source, '/source', True), (a.native, '/native', True),
                                 (a.native / 'libopapi_nn.so', '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_nn.so', True),
                                 (out, '/evidence', False), (a.model, '/model', True),
                                 (Path('/usr/local/Ascend/driver'), '/usr/local/Ascend/driver', True),
                                 (Path('/usr/local/dcmi'), '/usr/local/dcmi', True),
                                 (Path('/dev/null'), '/etc/ld.so.preload', True),
                                 (Path('/home/batch-scale-dynamic-msprobe-20261023/python_packages'), '/msprobe_packages', True)]:
                args += ['--mount', 'type=bind,src=' + str(src.resolve()) + ',dst=' + dst + (',readonly' if ro else '')]
            # Device cgroups expose only the selected physical devices. The
            # driver enumerates them as logical 0..N-1; physical IDs here are
            # not valid ASCEND_RT_VISIBLE_DEVICES logical indices.
            env = {'VLLM_SERVER_DEV_MODE': '1',
                   'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PYTHONDONTWRITEBYTECODE': '1',
                   'TASK_QUEUE_ENABLE': '0',
                   'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
            if a.lossless_receipt:
                args += ['--mount', 'type=bind,src=' + str(a.lossless_receipt.resolve()) + ',dst=/validation/lossless.json,readonly']
            for key, value in env.items():
                args += ['-e', key + '=' + value]
            # Preserve the image's matched 8.5.1 dependencies after our two prefixes.
            entry = 'unset ASCEND_RT_VISIBLE_DEVICES; export LD_PRELOAD=/native/libfault39_native_v1.so:/native/libopapi.so; export LD_LIBRARY_PATH=/native:$LD_LIBRARY_PATH; export PYTHONPATH=/source/fault39/src:/msprobe_packages:$PYTHONPATH; exec python3 "$@"'
            args += ['--entrypoint', '/bin/bash', IMAGE, '-c', entry, 'fault39', '/source/fault39/tools/' + case['tool'], *case['args']]
            item['command'] = args; item['started_utc'] = datetime.now(timezone.utc).isoformat(); save()
            print(json.dumps({'status': 'CASE_START', 'id': case['id'], 'cards': a.cards}), flush=True)
            try:
                with (out / 'process.log').open('w') as log:
                    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        item['exit_code'] = proc.wait(timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        item['timeout'] = True
                        # This is exclusively the container created above, never a foreign job.
                        subprocess.run(['docker', 'stop', '--time', '20', name], capture_output=True, timeout=30)
                        item['exit_code'] = proc.wait(timeout=30)
                info = json.loads(command(['docker', 'inspect', name]))[0]
                item['container_state'] = info['State']
                item['image_id'] = info['Image']
            finally:
                try:
                    info = json.loads(command(['docker', 'inspect', name]))[0]
                    if info['Config']['Labels'].get('dcat.task') != 'fault39-20261116':
                        raise RuntimeError('CONTAINER_OWNERSHIP_CHANGED')
                    if info['State']['Running']:
                        command(['docker', 'stop', '--time', '20', name], timeout=30)
                    command(['docker', 'rm', name])
                finally:
                    save()
            print(json.dumps({'status': 'CASE_END', 'id': case['id'], 'exit_code': item.get('exit_code')}), flush=True)
            if item.get('exit_code') != 0:
                receipt['status'] = 'CASE_FAILED'; save(); return 2
        receipt['status'] = 'CASES_COMPLETED'; save(); return 0


if __name__ == '__main__':
    raise SystemExit(main())
