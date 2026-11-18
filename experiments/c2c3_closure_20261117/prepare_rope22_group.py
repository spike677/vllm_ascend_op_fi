"""Prepare one root-owned actual Worker group case; never execute it."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.c2c3_closure_20261117.prepare_tensor_natural_cases import cases


def group_cases(source,card=2,image='quay.io/ascend/vllm-ascend:v0.18.0'):
    row,=cases(source,'rope22-census',card,image)
    row['id']='natural-rope22-group'
    row['command']=('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
        'python3 /source/experiments/c2c3_closure_20261117/natural_rope_worker_probe.py '
        '--model /model --out /evidence/probe --memory .35')
    row['review'].update(experiment_id=row['id'],provider='Source-bound actual fused RoPE legal positions input variant',
        expected_hits='All independently observed shared callsites once, then twice using original live C2 proof; no fixed28 assumption',
        recovery_plan='Four full healthy hidden/logits qualifications; C2+OFF then original live proof C3 two requests+OFF; unchanged root positions/cache/non-target rows; owned modes/profile/hooks drained and same-Worker healthy generation. No raw statistics in this group.')
    return [row]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--card',type=int,choices=(2,5),default=2)
    parser.add_argument('--image',default='quay.io/ascend/vllm-ascend:v0.18.0')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    with args.out.open('x') as stream:json.dump(group_cases(args.source,args.card,args.image),stream,indent=2)


if __name__=='__main__':main()
