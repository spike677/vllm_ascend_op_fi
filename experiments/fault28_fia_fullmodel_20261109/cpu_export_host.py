"""CPU-only bounded export container, exact owner cleanup, no device mount."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

ROOT=Path('/home/fault28-fia-fullmodel-20261109')
IMAGE='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'


def run(argv,seconds=30):
    return subprocess.run(argv,capture_output=True,text=True,timeout=seconds)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True)
    a=p.parse_args();assert a.attempt.replace('_','').isalnum()
    out=ROOT/'outputs'/a.attempt;out.mkdir(parents=True,exist_ok=False)
    token=uuid.uuid4().hex;name='f28-fia-full-cpu-'+token[:12];identity=None
    result={'status':'RUNNING','npu_device_mounts':[],'token':token,'name':name,'removed':False}
    try:
        command=['docker','create','--name',name,'--label','demoncat.cpu_owner='+token,
            '--network','none','--memory','12g','--cpus','4','--pids-limit','512']
        for source,dest,ro in [(ROOT,'/audit',True),(out,'/cpu_output',False),
            ('/home/tik-constfold-cann91-overnight-20261022','/work',True),
            ('/usr/local/Ascend/driver','/usr/local/Ascend/driver',True),
            ('/home/gyl/models/Qwen3-0.6B','/models/Qwen3-0.6B',True)]:
            command += ['--mount',f'type=bind,src={source},dst={dest}'+(',readonly' if ro else '')]
        command += ['--entrypoint','/bin/bash',IMAGE,'-lc',
            'source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /work/venv/bin/activate && '
            'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/audit OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 '
            'TRANSFORMERS_OFFLINE=1 python3 -B /audit/inspect_cpu.py']
        result['command']=command;r=run(command);assert r.returncode==0,r.stderr
        identity=r.stdout.strip();assert len(identity)==64
        r=run(['docker','start','-a',identity],300)
        (out/'stdout.txt').write_text(r.stdout);(out/'stderr.txt').write_text(r.stderr)
        result['returncode']=r.returncode
        assert r.returncode==0,r.stderr[-3000:]
        data=json.loads((out/'inventory.json').read_bytes())
        assert data['status']=='PASS_CPU_FULLGRAPH_EXPORT_NOT_NPU_ACCEPTED'
        result.update(status=data['status'],inventory_sha256=hashlib.sha256((out/'inventory.json').read_bytes()).hexdigest())
    except BaseException as e:
        result.update(status='FAIL_CPU_EXPORT',error=str(e))
    finally:
        if identity:
            r=run(['docker','inspect',identity])
            if r.returncode==0:
                obj=json.loads(r.stdout)[0]
                assert obj['Id']==identity and obj['Image']==IMAGE and obj['Name']=='/'+name
                assert obj['Config']['Labels']['demoncat.cpu_owner']==token
                result['removed']=run(['docker','rm','-f',identity]).returncode==0
        (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return 0 if result['status'].startswith('PASS_') and result['removed'] else 1


if __name__=='__main__':raise SystemExit(main())
