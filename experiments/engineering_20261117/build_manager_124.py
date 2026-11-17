"""Run actual builds with no NPU devices, keeping per-recipe build failures."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mechanism',required=True)
    p.add_argument('--operator');p.add_argument('--image',default='quay.io/ascend/vllm-ascend:v0.18.0')
    p.add_argument('--env-script',default='/usr/local/Ascend/cann-8.5.1/set_env.sh')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    # The trusted source recipe controls execution. The host shell never
    # interpolates operator names, paths or user text into a command.
    cmd=['docker','run','--rm','--network','none','--runtime','runc','--cpus','4','--memory','8g',
         '-e','ASCEND_VISIBLE_DEVICES=void','-e','PYTHONDONTWRITEBYTECODE=1',
         '-e','PYTHONPATH=/source/src:/source/fault39/src:/source',
         '--mount','type=bind,src='+str(a.source.resolve())+',dst=/source,readonly',
         '--mount','type=bind,src='+str(a.out.resolve())+',dst=/build',
         '--workdir','/source','--entrypoint','/bin/bash',a.image,'-c',
         'source "$1"\nshift\nexec python3 -m fault_injection.native_build --cache /build/cache build "$@"',
         'build',a.env_script,'--mechanism',a.mechanism]
    if a.operator:cmd+=['--operator',a.operator]
    with (a.out/'process.log').open('xb') as stream:
        result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    record=dict(command=cmd,exit_code=result.returncode,actual_npu_calls=0,
                status='BUILD_COMMAND_COMPLETED' if result.returncode==0 else 'BUILD_REFUSED_OR_FAILED')
    (a.out/'controller.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
