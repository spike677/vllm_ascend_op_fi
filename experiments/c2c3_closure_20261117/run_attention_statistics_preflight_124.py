"""Root-owned, device-free actual-package statistics preflight controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/17-fault-c2c3-20261117').resolve(strict=True)
    source = args.source.resolve(strict=True)
    output = args.out.resolve()
    if not source.is_relative_to(root) or not output.is_relative_to(root) or output.exists():
        raise RuntimeError('FRESH_ROOT_OWNED_CPU_PREFLIGHT_PATH_REQUIRED')
    output.mkdir(parents=True)
    image = 'cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8'
    image_id = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()
    name = 'dcat-attn-stats-cpu-'+uuid.uuid4().hex[:12]
    command = ['docker','run','--rm','--init','--runtime=runc','--network=none',
        '--name',name,'--cpus=4','--memory=8g','--pids-limit=512',
        '-e','ASCEND_VISIBLE_DEVICES=void','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','LD_PRELOAD=','-e','ASCEND_CUSTOM_OPP_PATH=',
        '-e','PYTHONPATH=/repo:/repo/src:/msprobe_packages','-e','OMP_NUM_THREADS=1',
        '--mount','type=bind,src='+str(source)+',dst=/repo,readonly',
        '--mount','type=bind,src=/home/batch-scale-dynamic-msprobe-20261023/python_packages,dst=/msprobe_packages,readonly',
        '--mount','type=bind,src='+str(output)+',dst=/build',
        '--entrypoint','bash',image_id,'-lc',
        'source /usr/local/Ascend/cann-9.1.0/set_env.sh\n'
        'exec python3 /repo/experiments/c2c3_closure_20261117/attention_cpu_statistics_preflight.py --out /build/probe']
    report = dict(schema='demoncat.attention.cpu_statistics_controller.v1',image=image,image_id=image_id,
                  command=command,devices=[],actual_npu_tensor_operations=0)
    with (output/'controller.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    try:
        with (output/'process.log').open('xb') as log:
            completed = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        report['exit_code'] = completed.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','--time','5',name],check=False,timeout=30,
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        report['exit_code'] = 124
    report['log_sha256'] = hashlib.sha256((output/'process.log').read_bytes()).hexdigest()
    (output/'controller.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('image_id','exit_code','actual_npu_tensor_operations')}))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
