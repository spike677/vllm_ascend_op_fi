"""Bounded deployment cases with the shared resource lock and per-case idle gate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path('/home/overnight-17-20261116')
sys.path.insert(0,str(ROOT/'source-a1/fault39/tools'))
from run_124 import idle_gate
LABEL='overnight-17-20261116'

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--cards',nargs='+',type=int,choices=[2,5],required=True)
    a=p.parse_args()
    if not re.fullmatch('[a-z0-9_-]{1,80}',a.attempt):p.error('Invalid attempt')
    cases=json.loads(a.cases.read_text())
    if not 1<=len(cases)<=8 or len(set(a.cards))!=len(a.cards):p.error('1..8 cases and distinct cards required')
    out=ROOT/a.attempt;out.mkdir()
    receipt={'cards':a.cards,'cases':[],'status':'WAITING_FOR_LOCK'}
    def save():(out/'controller.json').write_text(json.dumps(receipt,indent=2))
    save()
    with open('/var/tmp/fault22_npu_2_5.lock','a') as lock:
        deadline=time.monotonic()+900
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise RuntimeError('RESOURCE_LOCK_BUSY')
                time.sleep(2)
        for case in cases:
            if not re.fullmatch('[a-z0-9_-]{1,48}',case['id']):raise ValueError('Invalid case')
            gate=idle_gate(a.cards); row={'id':case['id'],'gate':gate};receipt['cases'].append(row);save()
            if not gate['idle']:
                receipt['status']='DEVICE_BUSY';save();return 3
            folder=out/case['id'];folder.mkdir()
            name=f'provider-{case["id"]}-{os.getpid()}'
            cmd=['docker','run','--name',name,'--label','dcat.task='+LABEL,'--network','none',
                 '--memory','80g','--cpus','16','--pids-limit','4096','--shm-size','4g']
            for device in [*(f'davinci{x}' for x in a.cards),'davinci_manager','devmm_svm','hisi_hdc']:
                cmd+=['--device','/dev/'+device]
            mounts=[('/usr/local/Ascend/driver','/usr/local/Ascend/driver','ro'),
                    ('/usr/local/dcmi','/usr/local/dcmi','ro'),('/dev/null','/etc/ld.so.preload','ro'),
                    (str(folder),'/evidence','rw'),
                    ('/home/batch-scale-dynamic-msprobe-20261023/python_packages','/msprobe_packages','ro')]
            mounts+=case.get('mounts',[])
            for source,dest,mode in mounts:
                if mode not in ('ro','rw') or not Path(source).exists():raise ValueError('Invalid mount')
                cmd+=['--mount',f'type=bind,src={source},dst={dest}'+(',readonly' if mode=='ro' else '')]
            env={'PYTHONDONTWRITEBYTECODE':'1','TASK_QUEUE_ENABLE':'0','VLLM_SERVER_DEV_MODE':'1',
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
            env.update(case.get('env',{}))
            for key,value in env.items():cmd+=['-e',key+'='+value]
            cmd+=['--entrypoint','/bin/bash',case['image'],'-c',case['command']]
            row['command']=cmd;save();print(json.dumps({'status':'START','case':case['id']}),flush=True)
            try:
                with (folder/'process.log').open('xb') as log:
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    try:row['exit_code']=proc.wait(timeout=min(case.get('timeout',900),1800))
                    except subprocess.TimeoutExpired:
                        row['timeout']=True
                        subprocess.run(['docker','stop','--time','20',name],capture_output=True,timeout=30)
                        row['exit_code']=proc.wait(timeout=30)
                info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
                row['container_state']=info['State'];row['image_id']=info['Image']
            finally:
                info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
                if info['Config']['Labels'].get('dcat.task')!=LABEL:raise RuntimeError('OWNERSHIP_CHANGED')
                if info['State']['Running']:subprocess.run(['docker','stop','--time','20',name],check=True,timeout=30)
                subprocess.run(['docker','rm',name],check=True,capture_output=True)
                save()
            print(json.dumps({'status':'END','case':case['id'],'exit_code':row['exit_code']}),flush=True)
            if row['exit_code']:
                receipt['status']='CASE_FAILED';save();return 2
        receipt['status']='COMPLETED';save();return 0

if __name__=='__main__':raise SystemExit(main())
