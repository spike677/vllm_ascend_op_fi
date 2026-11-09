"""One bounded full-model case, with existing shared lock and per-call gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/fault28-m-only-model-20261109')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=('main_b1','main_b8'))
    p.add_argument('--attempt',required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    manifest=json.loads((ROOT/'setup/manifest.json').read_bytes())
    row=next(c for c in manifest['cases'] if c['case_id']==a.case)
    path=ROOT/'setup'/row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    case=json.loads(path.read_bytes())
    assert case['execution_config']['selection']['target']=='query_tile'
    assert case['mode_sequence']==['calibration','native','controlled','controlled','off']
    command=[sys.executable,'-B',str(ROOT/'controller.py'),'--card','5','--attempt',a.attempt,
             '--case-file','setup/'+a.case+'.json','--probe','model_probe.py','--capture','native-logs',
             '--case-seconds','3600','--max-seconds','4000','--max-log-mib','16384']
    for identity in ('a8eeda3a5db34389fbcc83fdd8cabc99d687ac1db5aa4011a0586f2a9f0ddc4c',
                     'db21bd452c30c64dfd61152b6cf0a98bebfcea468a0f1c8a18452d2cb0e99de1'):
        command+=['--allow-idle-container-ID',identity]
    if a.prepare_only:command.append('--prepare-only')
    print(json.dumps({'command':command,'task':'M-only complete model; no external cleanup'}),flush=True)
    return subprocess.call(command)


if __name__=='__main__':raise SystemExit(main())
