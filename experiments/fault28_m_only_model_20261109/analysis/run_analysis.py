"""Analyze a finished full-model case: coverage, log census and independent stats."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/fault28-m-only-model-20261109')
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    attempt=ROOT/'outputs'/a.attempt;folder=attempt/a.case
    assert json.loads((attempt/'completion.json').read_bytes())['status']=='CAPTURE_COMPLETED'
    common=['--case-dir',str(folder),'--completion',str(attempt/'completion.json')]
    commands=[['audit_model_coverage.py',*common,'--model-config',str(folder/'model_config.json'),
               '--output',str(a.output/'framework.json')],
              ['compare_model_statistics.py',*common,'--output',str(a.output/'statistics')],
              ['native_launch_census.py','--log',str(attempt/(a.case+'.log')),'--case-dir',str(folder),
               '--output',str(a.output/'native_census.json')]]
    environment=dict(os.environ,PYTHONPATH=str(ROOT/'runtime')+':'+str(ROOT),OPENBLAS_NUM_THREADS='2')
    for command in commands:
        print(json.dumps({'analysis':command[0]}),flush=True)
        subprocess.run([sys.executable,'-B',str(HERE/command[0]),*command[1:]],check=True,env=environment)
    print(json.dumps({'status':'ANALYSIS_COMPLETED','output':str(a.output)}),flush=True)


if __name__=='__main__':main()
