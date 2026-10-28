"""Read-only post-run resource snapshot; no device contexts or container changes."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed: {result.stderr}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads((args.attempt / 'container_identity.json').read_text())
    completion = json.loads((args.attempt / 'completion.json').read_text())
    ids = run(['docker', 'ps', '-a', '-q', '--no-trunc']).split()
    result = {'epoch': time.time(), 'scope': 'Post-cleanup snapshot; does not reserve devices',
              'owned_container_id': identity['id'], 'owned_container_absent': identity['id'] not in ids,
              'npu_smi': run(['npu-smi', 'info']),
              'running_container_names': run(['docker', 'ps', '--format', '{{.Names}}']).splitlines(),
              'controller_completion_status': completion['status']}
    if not result['owned_container_absent'] or completion['status'] != 'PASS':
        raise RuntimeError('Completed experiment cleanup not established')
    result['status'] = 'PASS'
    with (args.attempt / 'after.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'status': 'PASS', 'owned_container_absent': True}))


if __name__ == '__main__':
    main()
