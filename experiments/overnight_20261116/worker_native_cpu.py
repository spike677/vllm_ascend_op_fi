"""Run C++/ctypes host-memory regressions in a device-free owned CPU container."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    source = root / 'source'
    output = root / 'output'
    output.mkdir(exist_ok=False)
    files = json.loads((source / 'cpu_source_manifest.json').read_text())
    for name, expected in files.items():
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('CPU_SOURCE_MISMATCH: ' + name)
    name = 'overnight-worker-cpu-' + uuid.uuid4().hex[:12]
    label = 'dcat.overnight.worker_cpu'
    command = r'''set -e
cd /source
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-error=misleading-indentation -pthread tests/native/test_matmul_auto.cpp -ldl -o /output/matmul_host
/output/matmul_host
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-error=misleading-indentation -pthread -shared -fPIC tests/native/test_matmul_auto.cpp -ldl -o /output/libhost_fixture.so
LD_PRELOAD=/output/libhost_fixture.so python3 -B tests/native/check_matmul_ctypes.py
'''
    create = ['docker', 'create', '--name', name, '--label', label + '=' + name,
              '--runtime=runc', '--network', 'none', '--cpus', '4', '--memory', '4g', '--pids-limit', '256',
              '-e', 'ASCEND_VISIBLE_DEVICES=void', '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0',
              '-e', 'PYTHONDONTWRITEBYTECODE=1', '--mount', f'type=bind,src={source},dst=/source,readonly',
              '--mount', f'type=bind,src={output},dst=/output', '--entrypoint', '/bin/bash',
              'quay.io/ascend/vllm-ascend:v0.18.0', '-ec', command]
    receipt = dict(schema='demoncat.overnight.native_cpu.v1', status='STARTING',
                   actual_npu_calls=0, npu_device_grants=0, source_files=files,
                   inherited_header_warning='Keep accepted runtime_identity.h bytes; misleading-indentation remains visible but is not a build error',
                   container_name=name, root=str(root), command=create)
    path = root / 'receipt.json'
    def save():
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    subprocess.run(create, check=True, stdout=subprocess.PIPE)
    try:
        info = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
        host = info['HostConfig']
        if host.get('Privileged') or host.get('Devices') or host.get('DeviceRequests') or host['Runtime'] != 'runc':
            raise RuntimeError('CPU_CONTAINER_GRANTED_DEVICE_ACCESS')
        receipt.update(device_free_prestart_verified=True, image_id=info['Image'])
        save()
        with (root / 'host_fixture.log').open('xb') as log:
            result = subprocess.run(['docker', 'start', '-a', name], stdout=log, stderr=subprocess.STDOUT, timeout=240)
        state = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]['State']
        receipt.update(returncode=state['ExitCode'], attach_returncode=result.returncode,
                       status='PASS_CPU_ONLY' if state['ExitCode'] == result.returncode == 0 else 'FAIL')
        raw = (root / 'host_fixture.log').read_bytes()
        receipt['log_sha256'] = hashlib.sha256(raw).hexdigest()
        receipt['checks'] = [line for line in raw.decode(errors='replace').splitlines() if line.startswith('PASS')]
    finally:
        info = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
        if info['Config']['Labels'].get(label) != name:
            raise RuntimeError('CPU_CONTAINER_OWNERSHIP_CHANGED')
        if info['State']['Running']:
            subprocess.run(['docker', 'stop', '--time', '5', name], check=True, timeout=20)
        subprocess.run(['docker', 'rm', name], check=True, stdout=subprocess.PIPE)
        receipt['container_removed'] = True
        save()
    print(json.dumps(dict(status=receipt['status'], actual_npu_calls=0, receipt=str(path))))
    return int(receipt['status'] != 'PASS_CPU_ONLY')


if __name__ == '__main__':
    raise SystemExit(main())
