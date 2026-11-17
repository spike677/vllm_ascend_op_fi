"""Root-owned CPU-only controller for actual stock/private library resolution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','native','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    root=Path('/home/17-fault-c2c3-20261117').resolve(strict=True)
    source,native=args.source.resolve(strict=True),args.native.resolve(strict=True)
    output=args.out.resolve()
    if not all(p.is_relative_to(root) for p in (source,native,output)) or output.exists():
        raise RuntimeError('FRESH_ROOT_OWNED_CPU_LOADING_PATH_REQUIRED')
    output.mkdir(parents=True)
    completion=native/'evidence/completion.json'
    completion_sha=hashlib.sha256(completion.read_bytes()).hexdigest()
    image='sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
    name='dcat-attn33-load-cpu-'+uuid.uuid4().hex[:12]
    command=['docker','run','--rm','--init','--runtime=runc','--network=none','--name',name,
        '--cpus=4','--memory=8g','--pids-limit=512','-e','ASCEND_VISIBLE_DEVICES=void',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','LD_PRELOAD=','-e','ASCEND_CUSTOM_OPP_PATH=',
        '-e','PYTHONPATH=/repo:/repo/src','-e','OMP_NUM_THREADS=1',
        '--mount','type=bind,src='+str(source)+',dst=/repo,readonly',
        '--mount','type=bind,src='+str(native)+',dst=/native,readonly',
        '--mount','type=bind,src='+str(output)+',dst=/build','--entrypoint','bash',image,'-lc',
        'source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
        'exec python3 /repo/experiments/c2c3_closure_20261117/attention33_loading_preflight.py '
        '--completion /native/evidence/completion.json --completion-sha256 '+completion_sha+
        ' --vendor /native/stage/vendors/dcat_f33_fai_transformer --out /build/probe']
    report=dict(schema='demoncat.paged33.loading_cpu_controller.v1',image=image,command=command,
                devices=[],actual_npu_tensor_operations=0)
    with (output/'controller.json').open('x') as stream:json.dump(report,stream,indent=2)
    try:
        with (output/'process.log').open('xb') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        report['exit_code']=result.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','--time','5',name],check=False,timeout=30,
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        report['exit_code']=124
    report['log_sha256']=hashlib.sha256((output/'process.log').read_bytes()).hexdigest()
    (output/'controller.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('exit_code','actual_npu_tensor_operations')}))
    return report['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
