"""CPU regression receipt with explicit skips; never grants NPU qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];a.out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=os.pathsep.join(str(root/x) for x in ('src','fault39/src','.','tests')))
    suites=[]
    for name,directory in [('unified','tests'),('worker_extension','tests/worker_extension'),
                           ('fault39','fault39/tests'),('native_strategy','native_execution_strategy/tests')]:
        command=[sys.executable,'-B','-m','unittest','discover','-s',directory,'-v']
        r=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        raw=r.stdout;text=raw.decode('utf-8',errors='replace');(a.out/(name+'.log')).write_bytes(raw)
        counts=re.findall(r'Ran (\d+) tests?',text);skip=re.findall(r"(test[^\r\n]+\.\.\. skipped [^\r\n]+)",text)
        suites.append(dict(suite=name,command=command[1:],exit_code=r.returncode,
            tests=int(counts[-1]) if counts else 0,skips=skip,log_sha256=hashlib.sha256(raw).hexdigest()))
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines()
    new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines()
    hashes={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sorted(set(changed+new))
            if (root/s).is_file() and Path(s).suffix in ('.py','.cpp','.inc','.h','.sh','.toml')}
    okay=all(s['exit_code']==0 and s['tests']>0 for s in suites)
    report=dict(schema='demoncat.engineering.cpu_validation.v1',status='CPU_REGRESSION_PASSED_WITH_EXPLICIT_SKIPS'
        if okay and any(s['skips'] for s in suites) else 'CPU_REGRESSION_PASSED' if okay else 'CPU_REGRESSION_FAILED',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        actual_npu_calls=0,tests=sum(s['tests'] for s in suites),suites=suites,
        source_sha256=hashes,runtime_admission_granted=False)
    (a.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
    return 0 if okay else 1


if __name__=='__main__':raise SystemExit(main())
