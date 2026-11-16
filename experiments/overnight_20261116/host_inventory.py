"""Read-only host inventory; no NPU initialization or job intervention."""
import hashlib
import json
from pathlib import Path
import subprocess


def command(args):
    p = subprocess.run(args, text=True, capture_output=True, timeout=30)
    return dict(argv=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def main():
    models = []
    for root in ('/home/gyl/models', '/home/ws/models', '/home/models', '/home/haoran/models'):
        for p in Path(root).glob('*/config.json'):
            value = json.loads(p.read_text())
            models.append(dict(path=str(p.parent), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                               model_type=value.get('model_type'), architectures=value.get('architectures')))
    historical = []
    for p in (Path('/home/haoran/fault8_systemic_mapping_20260915_v3/systemic_extension.py'),
              Path('/home/haoran/fault8_systemic_mapping_20260915_v3/strategy.py')):
        historical.append(dict(path=str(p), exists=p.is_file(),
                               sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    out = dict(schema='demoncat.overnight.host_inventory.v1', actual_npu_calls=0,
               inventory=[command(x) for x in (['uname', '-m'], ['npu-smi', 'info'],
                ['docker', 'ps', '--format', '{{json .}}'],
                ['docker', 'images', '--format', '{{json .}}'])],
               models=models, historical_sources=historical,
               host_ascend_paths=[str(p) for p in Path('/usr/local/Ascend').iterdir()],
               prior_image_installs='/home/provider-adaptation-20261116/build-{851,90,91,92}-a5/version_profile.json',
               limitations=['Host installation paths are not loaded-worker identity.',
                            'No A3 device was observed; no support inferred.'])
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
