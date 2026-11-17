"""Build in a CPU-only container without exposing NPU devices."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--attempt', required=True)
a = p.parse_args()
root = Path('/home/worker-positive-20261117')
if not all(v.replace('-', '').isalnum() for v in (a.source, a.attempt)):
    raise ValueError('Invalid immutable source/attempt label')
source = root / a.source
source.mkdir()
with tarfile.open(root / (a.source + '.tar.gz')) as archive:
    archive.extractall(source, filter='data')
out = root / a.attempt
out.mkdir()
image = 'quay.io/ascend/vllm-ascend:v0.18.0'
receipt = dict(actual_npu_calls=0, image=image, results=[])
for mechanism, script, variable in (
        ('39', 'fault39/native/build.sh', 'FAULT39_OUTPUT_DIR'),
        ('20', 'fault_injection/native_adapters/build_qmm.sh', 'DEMONCAT_QMM_OUTPUT')):
    cmd = ['docker', 'run', '--rm', '--runtime=runc', '--network=none', '--cpus=8', '--memory=16g',
           '--name', 'worker-positive-build-' + a.attempt + '-' + mechanism,
           '-e', 'ASCEND_VISIBLE_DEVICES=void', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-e', variable + '=/output/build' + mechanism,
           '-e', 'FAULT39_PATCHELF=/build-bin/patchelf',
           '--mount', f'type=bind,src={source},dst=/source,readonly',
           '--mount', f'type=bind,src={out},dst=/output',
           '--mount', 'type=bind,src=/home/provider-adaptation-20261116/bin,dst=/build-bin,readonly',
           '--entrypoint', '/bin/bash', image, '-c',
           'source /usr/local/Ascend/cann-8.5.1/set_env.sh && bash /source/' + script +
           (' && python3 /source/fault39/native/test_serving_inputs.py --output /output/build39/serving_inputs_cpu.json'
            if mechanism == '39' else '')]
    with (out / ('build' + mechanism + '.log')).open('xb') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    receipt['results'].append(dict(mechanism=mechanism, command=cmd, exit_code=result.returncode))
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['results'][-1]), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
