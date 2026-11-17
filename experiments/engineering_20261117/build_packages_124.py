"""Install all three source packages into a new device-free deployment."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    script='''set -e
cp -a /source /tmp/build-source
python3 -m pip wheel --no-deps --no-build-isolation --no-index /tmp/build-source /tmp/build-source/src /tmp/build-source/fault39 -w /output/wheels
python3 -m pip install --no-deps --no-index --target /output/installed /output/wheels/*.whl
'''
    cmd=['docker','run','--rm','--runtime=runc','--network=none','--cpus=4','--memory=8g',
         '-e','ASCEND_VISIBLE_DEVICES=void','-e','PYTHONDONTWRITEBYTECODE=1',
         '--mount','type=bind,src='+str(a.source.resolve())+',dst=/source,readonly',
         '--mount','type=bind,src='+str(a.out.resolve())+',dst=/output','--entrypoint','/bin/bash',
         'quay.io/ascend/vllm-ascend:v0.18.0','-c',script]
    with (a.out/'build.log').open('xb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    (a.out/'receipt.json').write_text(json.dumps(dict(actual_npu_calls=0,command=cmd,exit_code=r.returncode),indent=2)+'\n')
    print(json.dumps(dict(out=str(a.out),exit_code=r.returncode,actual_npu_calls=0)))
    return r.returncode


if __name__=='__main__':raise SystemExit(main())
