"""Root-owned, device-free rebuild of the existing writer26 trace recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    manifest = json.loads((source/'source_manifest.json').read_bytes())
    for row in manifest['files']:
        p = source/row['path']
        if p.is_symlink() or not p.resolve().is_relative_to(source) or sha(p) != row['sha256']:
            raise RuntimeError('FROZEN_SOURCE_CHANGED:'+row['path'])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    owner = uuid.uuid4().hex
    script = ('set -e\nsource /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
              'export PYTHONPATH=/source\nexec python3 -B -m fault_injection.native_build '
              '--repository /source --cache /build/cache build --mechanism 26.residual_stomp '
              '--operator aclnnQuantMatmulWeightNz --timeout 600')
    command = ['docker', 'create', '--name', 'dcat26-parent-build-'+owner,
        '--label', 'dcat.cpu.owner='+owner, '--runtime', 'runc', '--network', 'none',
        '--cpus', '4', '--memory', '8g', '--pids-limit', '512',
        '-e', 'ASCEND_VISIBLE_DEVICES=void', '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0',
        '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '--mount', 'type=bind,src='+str(source)+',dst=/source,readonly',
        '--mount', 'type=bind,src='+str(out)+',dst=/build',
        '--mount', 'type=bind,src=/dev/null,dst=/etc/ld.so.preload,readonly',
        '--workdir', '/source', '--entrypoint', '/bin/bash', IMAGE, '-c', script]
    receipt = dict(schema='demoncat.root.cpu_build.v1', actual_npu_calls=0,
        status='PREPARING', source_commit=manifest['base_commit'], controller_sha256=sha(__file__),
        source_manifest_sha256=sha(source/'source_manifest.json'), command=command)
    def save():
        (out/'controller.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    container = subprocess.check_output(command, text=True, timeout=30).strip()
    receipt['container_id'] = container
    def owned():
        item, = json.loads(subprocess.check_output(['docker', 'inspect', container], timeout=30))
        host = item['HostConfig']
        if (item['Id'] != container or item['Image'] != IMAGE
                or item['Config']['Labels'].get('dcat.cpu.owner') != owner
                or host['Runtime'] != 'runc' or host['NetworkMode'] != 'none'
                or host['Privileged'] or host.get('Devices') or host.get('DeviceRequests')):
            raise RuntimeError('OWNED_DEVICE_FREE_BUILD_CHANGED')
        return item
    try:
        receipt['devices'] = owned()['HostConfig'].get('Devices') or []
        receipt['status'] = 'BUILDING'
        save()
        with (out/'build.log').open('xb') as log:
            result = subprocess.run(['docker', 'start', '-a', container], stdout=log,
                                    stderr=subprocess.STDOUT, timeout=900)
        receipt.update(returncode=result.returncode,
                       status='CPU_BUILD_COMPLETE' if result.returncode == 0 else 'CPU_BUILD_FAILED')
    except BaseException as error:
        receipt.update(status='CPU_BUILD_CONTROLLER_FAILED', error=repr(error))
        raise
    finally:
        if owned()['State']['Running']:
            subprocess.run(['docker', 'stop', '--time', '10', container], check=True,
                           timeout=30, stdout=subprocess.DEVNULL)
        receipt['final_state'] = owned()['State']
        subprocess.run(['docker', 'rm', container], check=True, timeout=30, stdout=subprocess.DEVNULL)
        receipt['cleanup'] = True
        if (out/'build.log').exists():
            receipt['log_sha256'] = sha(out/'build.log')
        save()
    print(json.dumps({key: receipt[key] for key in ('status', 'returncode', 'actual_npu_calls')}))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
