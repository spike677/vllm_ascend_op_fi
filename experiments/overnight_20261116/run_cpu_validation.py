"""Execute all available Python suites without granting NPU acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=os.pathsep.join(str(ROOT/x) for x in ('src','fault39/src','.')))
    rows=[]
    for name,directory in [('unified','tests'),('strategy','native_execution_strategy/tests'),
                           ('fault39','fault39/tests'),('fault39_tools','fault39/tools'),
                           ('statistics_diagnostics','experiments/overnight_20261116')]:
        start=time.monotonic();cmd=[sys.executable,'-B','-m','unittest','discover','-s',directory,'-v']
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        raw=result.stdout;(a.output/(name+'.log')).write_bytes(raw)
        text=raw.decode(errors='replace');counts=re.findall(r'Ran (\d+) tests?',text)
        row=dict(suite=name,returncode=result.returncode,tests=int(counts[-1]) if counts else 0,
                 skipped=bool(re.search(r'\bskipped\b',text)),seconds=round(time.monotonic()-start,3),
                 log_sha256=hashlib.sha256(raw).hexdigest())
        rows.append(row);print(json.dumps(row),flush=True)
    out=dict(schema='demoncat.overnight.cpu.v1',runs=rows,actual_npu_calls=0,
             status='PASS_CPU_ONLY' if all(r['returncode']==0 and r['tests'] and not r['skipped'] for r in rows) else 'FAILED_OR_INCOMPLETE')
    (a.output/'receipt.json').write_text(json.dumps(out,indent=2)+'\n')
    return int(out['status']!='PASS_CPU_ONLY')


if __name__=='__main__':raise SystemExit(main())
