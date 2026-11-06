"""Read selected host/container metadata without reading container secrets."""
import json
from pathlib import Path
import subprocess


def run(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise RuntimeError(str(argv[:3]) + ': ' + p.stderr[-1500:])
    return p.stdout


def main():
    image = 'sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
    obj = json.loads(run(['docker', 'image', 'inspect', image]))[0]
    result = {'schema': 'fault33.host_inventory.v1', 'npu_calls': 0,
              'image': {'id': obj['Id'], 'tags': obj.get('RepoTags'),
                        'entrypoint': obj['Config'].get('Entrypoint'), 'command': obj['Config'].get('Cmd')},
              'containers': []}
    for obj in json.loads(run(['docker', 'inspect', 'gyl_test', 'chunk_fwd_o_atk_20261031', 'catmonitor-demo-control'])):
        result['containers'].append({'id': obj['Id'], 'name': obj['Name'], 'image': obj['Image'],
            'command': obj['Config'].get('Cmd'), 'entrypoint': obj['Config'].get('Entrypoint'),
            'devices': obj['HostConfig'].get('Devices'), 'running': obj['State']['Running'],
            'started_at': obj['State'].get('StartedAt'),
            'processes': run(['docker', 'top', obj['Id'], '-eo', 'pid,ppid,comm,args'])})
    result['previous_roots'] = [str(p) for p in Path('/home').glob('*fault28*') if p.is_dir()]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
