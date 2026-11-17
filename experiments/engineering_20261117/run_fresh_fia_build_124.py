"""Run one fresh FIA source build without exposing any NPU or network."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--driver',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--dependencies',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    mounts=[(a.driver,'/driver'),(a.source,'/input-source'),(a.receipt,'/policy.json'),
            (a.dependencies,'/input-dependencies'),(Path('/usr/bin/patch'),'/buildtools/patch')]
    cmd=['docker','run','--rm','--runtime=runc','--network=none','--cpus=4','--memory=16g',
         '-e','ASCEND_VISIBLE_DEVICES=void','-e','PYTHONDONTWRITEBYTECODE=1']
    for src,dest in mounts:cmd+=['--mount','type=bind,src='+str(src.resolve())+',dst='+dest+',readonly']
    cmd+=['--mount','type=bind,src='+str(a.out.resolve())+',dst=/build','--entrypoint','bash',
          'cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8','-c',
          'source /usr/local/Ascend/cann-9.1.0/set_env.sh\nexport PATH=/buildtools:$PATH\nexec python3 /driver/build_fresh.py '
          '--source /input-source --policy-receipt /policy.json --dependencies /input-dependencies '
          '--work /build/run --jobs 4']
    with (a.out/'process.log').open('xb') as stream:
        r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
    record=dict(command=cmd,exit_code=r.returncode,actual_npu_calls=0)
    (a.out/'controller.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record));return r.returncode


if __name__=='__main__':raise SystemExit(main())
