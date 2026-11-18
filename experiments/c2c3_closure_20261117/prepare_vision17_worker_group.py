"""Prepare one root-owned public MM Torch-parameter group, without execution."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.c2c3_closure_20261117.prepare_vision17_worker_census import cases as census_cases


def cases(source,card=2):
    row=census_cases(source,card)[0]
    row['id']='vision17-public-worker-torch-group'
    row['command']=row['command'].replace('vision17_worker_census_probe.py','vision17_worker_group_probe.py')
    row['review'].update(experiment_id=row['id'],provider='Vision17GroupProvider through existing WorkerFaultRuntime/CoverageSession',
        expected_hits=3,acceptance='C1_AND_SINGLE_TARGET_SET_C2_C3_TORCH_PARAMETER_ONLY_NATIVE_AND_E5_UNOBSERVED',
        recovery_plan='Four fresh healthy bridge requests, once-F/OFF, two every-match F/OFF, cleanup-healthy; public idle MM and encoder cache reset every request, actual scheduled encoder required. On failure retain originals and require real OFF plus healthy request before release.',
        actual_group_targets='Must independently discover exactly one eligible projection; no multi-target qualification claim')
    return [row]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--card',type=int,choices=(2,5),default=2)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    with args.out.open('x') as stream:json.dump(cases(args.source,args.card),stream,indent=2)


if __name__=='__main__':main()
