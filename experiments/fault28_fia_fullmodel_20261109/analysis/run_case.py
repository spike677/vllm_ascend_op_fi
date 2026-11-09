"""Run one preregistered case on idle card5; no external cleanup capability."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/fault28-fia-fullmodel-20261109')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--matrix',default='matrix_v5.json',choices=tuple('matrix_v'+str(i)+'.json' for i in range(1,6)));a=p.parse_args()
    matrix=json.loads((ROOT/'setup'/a.matrix).read_bytes())
    assert a.case in matrix['case_ids']
    assert sha(ROOT/'setup'/(a.case+'.json'))==matrix['case_sha256'][a.case]
    for name,digest in matrix['runtime_sha256'].items():assert sha(ROOT/name)==digest,name
    sys.path.insert(0,str(ROOT));import controller
    command=[sys.executable,'-B',str(ROOT/'controller.py'),'--card','5','--probe','graph_probe.py',
        '--case-file','setup/'+a.case+'.json','--attempt',a.attempt,'--capture','native-logs',
        '--case-seconds','900','--max-seconds','1200','--max-log-mib','8192']
    for identity,row in controller.REVIEWED_IDLE_CONTAINERS.items():
        if row['name'] in ('gyl_test','catmonitor-demo-control'):
            command += ['--allow-idle-container-ID',identity]
    raise SystemExit(subprocess.call(command))


if __name__=='__main__':main()
