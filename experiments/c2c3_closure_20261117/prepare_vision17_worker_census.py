"""Prepare one root-queued two-healthy-MM-request case; no device execution."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.c2c3_closure_20261117.prepare_tensor_worker_core import core_cases


def cases(source,card=2,reset_mm_cache=False,image='sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'):
    row=core_cases(source,card,image)[0]
    row['id']='vision17-public-worker-census'
    model='/home/17-fault-c2c3-20261117/models/Qwen3-VL-2B-Instruct'
    row['mounts']=[[source,'/source','ro'],[model,'/model','ro'],['/home/jw/Wan2.2/examples/i2v_input.JPG','/natural-input.JPG','ro']]
    row['command']='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec python3 -u /source/experiments/c2c3_closure_20261117/vision17_worker_census_probe.py --model /model --image /natural-input.JPG --out /evidence/probe --memory .60 > /evidence/vision17.stdout.log 2> /evidence/vision17.stderr.log'
    row['review'].update(experiment_id=row['id'],provider='Vision17ObservationProvider through existing WorkerFaultRuntime',
        model=model,expected_hits=0,acceptance='OBSERVATION_ONLY_NO_C1_C2_C3_NATIVE_OR_E5',
        worker='Actual single-image public enqueue, TP1 BF16 eager, existing pretrained Qwen3-VL; no fixed layer names',
        recovery_plan='Two healthy enqueues only; original scheduler MM feature/placeholder/frame binding, final Torch Conv arguments/first consumer; retain cache skips; remove owned observer and Runtime lease. No F or direct internal cache mutation.',
        raw_evidence=['/evidence/probe/result.json','/evidence/vision17.stdout.log','/evidence/vision17.stderr.log'])
    if reset_mm_cache:
        row['id']='vision17-public-worker-census-mm-reset'
        row['command']=row['command'].replace(' --memory .60', ' --memory .60 --reset-mm-cache')
        row['review'].update(experiment_id=row['id'])
        row['review']['recovery_plan'] += ' Public reset_mm_cache at each idle healthy enqueue; actual scheduled encoder observations remain mandatory.'
    return [row]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--card',type=int,choices=(2,5),default=2)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reset-mm-cache',action='store_true');args=parser.parse_args()
    with args.out.open('x') as stream:json.dump(cases(args.source,args.card,args.reset_mm_cache),stream,indent=2)


if __name__=='__main__':main()
