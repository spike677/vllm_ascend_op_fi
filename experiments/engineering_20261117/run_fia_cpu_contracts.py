"""Persist the Linux-only fresh-vendor gate tests without importing torch/NPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    cmd=[sys.executable,'-B','-m','unittest','discover','-s',str(a.source/'tests'),'-p','test_fresh_fia_qualification.py','-v']
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    log=a.out/'tests.log';log.write_bytes(result.stdout)
    text=result.stdout.decode('utf-8',errors='replace')
    passed=result.returncode==0 and 'Ran 7 tests' in text and 'skipped' not in text
    files=('experiments/engineering_20261117/fresh_fia_qualification.py','tests/test_fresh_fia_qualification.py')
    receipt=dict(status='CPU_LINUX_SYMLINK_CONTRACTS_PASSED' if passed else 'FAILED',
        command=cmd,exit_code=result.returncode,tests=7,skipped=0 if passed else None,
        source_sha256={name:hashlib.sha256((a.source/name).read_bytes()).hexdigest() for name in files},
        log=dict(path=str(log),sha256=hashlib.sha256(result.stdout).hexdigest()),
        actual_npu_calls=0,runtime_admission_granted=False)
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
