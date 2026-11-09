"""CPU-only final review after BOTH controller attempts have cleanly completed.

No waiting, device operation, controller launch, certificate issuance or
automatic publication. Existing review outputs are never overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/fault28-m-only-model-20261109')
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--b1-attempt',required=True)
    p.add_argument('--b8-attempt',required=True);p.add_argument('--b1-analysis',required=True)
    p.add_argument('--b8-analysis',required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'runtime')+':'+str(ROOT),OPENBLAS_NUM_THREADS='2')
    def run(name,*arguments):
        command=[sys.executable,'-B',str(HERE/name),*map(str,arguments)]
        print(json.dumps({'cpu_review_step':name}),flush=True)
        subprocess.run(command,check=True,env=env)
    for name in (a.b1_attempt,a.b8_attempt):
        assert '/' not in name and json.loads((ROOT/'outputs'/name/'completion.json').read_bytes())['status']=='CAPTURE_COMPLETED'
    run('prepare_review.py','--b1-attempt',a.b1_attempt,'--b8-attempt',a.b8_attempt,
        '--b1-analysis',a.b1_analysis,'--b8-analysis',a.b8_analysis,'--output',a.review)
    for suffix,attempt,case in (('b1',a.b1_attempt,'main_b1'),('b8',a.b8_attempt,'main_b8')):
        folder=ROOT/'outputs'/attempt/case
        for tool,kind in (('summarize_numerics.py','numeric'),('review_exact_logits.py','logits')):
            destination=a.review/f'{kind}_{suffix}.json'
            run(tool,'--case-dir',folder,'--output',destination)
    run('accept_model_execution.py','--inputs',a.review/'suite.json','--output',a.review/'execution_acceptance.json')
    run('review_formal_summary.py','--inputs',a.review/'suite.json','--acceptance',a.review/'execution_acceptance.json',
        '--output',a.review/'formal_statistics_hash_review.json')
    run('compare_cross_batch_target.py','--b1',ROOT/'outputs'/a.b1_attempt/'main_b1',
        '--b8',ROOT/'outputs'/a.b8_attempt/'main_b8','--output',a.review/'cross_batch_target.json')
    run('review_causal_chain.py','--review',a.review,'--output',a.review/'causal_chain.json')
    print(json.dumps({'status':'CPU_FINAL_REVIEW_COMPLETED','review':str(a.review)}),flush=True)


if __name__=='__main__':main()
