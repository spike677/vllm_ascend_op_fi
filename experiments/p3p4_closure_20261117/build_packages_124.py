"""Build and install all three wheels into an isolated, no-device deployment."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
if not all(x.replace('-','').isalnum() for x in (a.source,a.attempt)):raise ValueError('Invalid labels')
root=Path('/home/p3p4-closure-20261117');source=root/a.source;source.mkdir()
with tarfile.open(root/(a.source+'.tar.gz')) as tar:tar.extractall(source,filter='data')
out=root/a.attempt;out.mkdir()
script='''set -e
cp -a /source /tmp/build-source
python3 -m pip wheel --no-deps --no-build-isolation --no-index /tmp/build-source /tmp/build-source/src /tmp/build-source/fault39 -w /output/wheels
python3 -m pip install --no-deps --no-index --target /output/installed /output/wheels/*.whl
'''
cmd=['docker','run','--rm','--runtime=runc','--network=none','--cpus=4','--memory=8g',
     '-e','ASCEND_VISIBLE_DEVICES=void','-e','PYTHONDONTWRITEBYTECODE=1',
     '--mount',f'type=bind,src={source},dst=/source,readonly',
     '--mount',f'type=bind,src={out},dst=/output','--entrypoint','/bin/bash',
     'quay.io/ascend/vllm-ascend:v0.18.0','-c',script]
with (out/'build.log').open('xb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
(out/'receipt.json').write_text(json.dumps(dict(actual_npu_calls=0,command=cmd,exit_code=r.returncode),indent=2)+'\n')
raise SystemExit(r.returncode)
