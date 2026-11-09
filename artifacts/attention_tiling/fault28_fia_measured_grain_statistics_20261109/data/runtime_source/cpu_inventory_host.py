"""Bounded CPU-only container inventory. Never exposes NPU devices."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import re
import uuid

ROOT = Path('/home/fault28-fia-grain-20261109')
IMAGE = 'sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'


def command(argv, timeout=45):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--runtime', choices=('cann91_work', 'cann851_system'), default='cann91_work')
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]{1,40}', args.attempt)
    image = IMAGE if args.runtime=='cann91_work' else 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
    out = ROOT/'outputs'/args.attempt
    out.mkdir(exist_ok=False, parents=True)
    token = uuid.uuid4().hex
    name = 'dcat-f28-fia-cpu-'+token[:12]
    identity = None
    receipt = {'schema': 'fault28.fia_graph.cpu_inventory_host.v1', 'npu_devices': [],
               'token': token, 'name': name, 'image': image, 'runtime':args.runtime, 'status': 'STARTING', 'removed': False}
    try:
        argv = ['docker', 'create', '--name', name, '--label', 'dcat.f28_fia_cpu='+token,
                '--cidfile', str(out/'cid'), '--network', 'none', '--cpus', '2', '--memory', '8g',
                '--pids-limit', '256', '--mount', f'type=bind,src={ROOT},dst=/audit,readonly',
                '--mount', 'type=bind,src=/home/tik-constfold-cann91-overnight-20261022,dst=/work,readonly',
                '--mount', 'type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly',
                '--entrypoint', '/bin/bash', image, '-lc']
        runtime = ('source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /work/venv/bin/activate && '
                   if args.runtime=='cann91_work' else
                   'for f in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh; do '
                   'if test -f "$f"; then source "$f"; break; fi; done; ')
        argv += [runtime+'PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 python3 -B /audit/inspect_cpu.py']
        receipt['command'] = argv
        made = command(argv)
        if made.returncode:
            raise RuntimeError(made.stderr[-3000:])
        identity = made.stdout.strip()
        if len(identity) != 64 or not all(c in '0123456789abcdef' for c in identity):
            raise RuntimeError('Invalid exact container identity')
        result = command(['docker', 'start', '-a', identity], timeout=180)
        (out/'stdout.txt').write_text(result.stdout)
        (out/'stderr.txt').write_text(result.stderr)
        receipt['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:])
        start = result.stdout.find('{')
        inventory, end = json.JSONDecoder().raw_decode(result.stdout[start:])
        receipt['stdout_trailer'] = result.stdout[start+end:]
        (out/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
        receipt.update(status='PASS_CPU_INVENTORY', inventory_sha256=hashlib.sha256((out/'inventory.json').read_bytes()).hexdigest())
    except Exception as exc:
        receipt.update(status='FAIL_CPU_INVENTORY', error=str(exc))
    finally:
        cid = out/'cid'
        if identity is None and cid.is_file():
            identity = cid.read_text().strip()
        if identity:
            found = command(['docker', 'inspect', identity])
            obj = json.loads(found.stdout)[0] if found.returncode == 0 else None
            if obj and obj['Id'] == identity and obj['Name'] == '/'+name and obj['Image'] == image and obj['Config']['Labels'].get('dcat.f28_fia_cpu') == token:
                removed = command(['docker', 'rm', '-f', identity])
                receipt['removed'] = removed.returncode == 0
        (out/'completion.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
    return 0 if receipt['status'] == 'PASS_CPU_INVENTORY' and receipt['removed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
