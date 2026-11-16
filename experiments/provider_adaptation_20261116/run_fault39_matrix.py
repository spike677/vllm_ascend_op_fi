"""Run bounded, serialized B/F/OFF cases through the existing lock and gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from prepare_cpu_builds import IMAGES

ROOT = Path('/home/provider-adaptation-20261116')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--attempt',required=True)
    p.add_argument('--source',default='a5')
    p.add_argument('--tags',nargs='+',choices=list(IMAGES),default=list(IMAGES))
    a=p.parse_args()
    if not a.attempt.isalnum() or not a.source.isalnum(): p.error('Alphanumeric attempts required')
    records=[]
    for tag in a.tags:
        output=ROOT/f'npu-{tag}-{a.attempt}'
        args=[sys.executable,str(ROOT/f'source-{a.source}/fault39/tools/run_124.py'),
              '--cards','2','--source',str(ROOT/f'source-{a.source}'),
              '--native',str(ROOT/f'build-{tag}-{a.source}'),
              '--attempt',str(output),'--image',IMAGES[tag],
              '--cases',str(ROOT/'fault39_operator_cases.json'),'--timeout','600']
        rc=subprocess.run(args).returncode
        row={'version_tag':tag,'runner_exit':rc,'attempt':str(output)}
        records.append(row)
        print(json.dumps(row),flush=True)
    path=ROOT/f'npu-matrix-{a.attempt}.json'
    with path.open('x') as f: json.dump(records,f,indent=2)
    return int(any(row['runner_exit'] for row in records))

if __name__=='__main__': raise SystemExit(main())
