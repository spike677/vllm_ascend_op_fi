"""Produce explicitly labelled fixture examples and sanitized historical QMM replay.

Offline only. No torch installation, server requests, observers or mutations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ascend_fi.inspect.operators import OperatorEvidenceAdapter
from ascend_fi.discovery.resolver import load_capabilities
from ascend_fi.discovery.adapters import TraceContext
from ascend_fi.validation import validate_target
from ascend_fi.validation.evidence import RecordedLosslessExecutor
from ascend_fi.contracts import Candidate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    adapter=OperatorEvidenceAdapter(load_capabilities(ROOT/'profiles/capabilities'))
    source=ROOT/'artifacts/evaluation/qwen_2card_fault_effect_v1/v4/candidate.json'
    candidate=adapter.recorded_candidate(json.loads(source.read_text(encoding='utf-8')),source_ref=str(source.relative_to(ROOT)))
    trace=json.loads((ROOT/'tests/fixtures/qmm_run_trace.json').read_text(encoding='utf-8').splitlines()[1])
    native=adapter.native_run(trace,context=TraceContext(rank=0,runtime={'provenance':'SYNTHETIC_FIXTURE'},evidence_ref='tests/fixtures/qmm_run_trace.json'))
    native['provenance']='SYNTHETIC_FIXTURE_NOT_LIVE_MODEL'
    typed=adapter.typed_probe({'event':'typed_qmm_call_exit','tensors':{'x1':{'dtype':'int8','shape':[1,8]},
            'x2':{'dtype':'int8','shape':[8,8]},'scale':{'dtype':'float32','shape':[8]},'bias':None},
            'output':{'dtype':'bfloat16','shape':[1,8]}},source_ref='SYNTHETIC_FIXTURE_NOT_LIVE_MODEL')
    c=next(c for c in native['candidates'] if c['capability_id']=='op_api.quant_matmul.input_buffer_transform')
    validation=validate_target(Candidate.from_dict(c),RecordedLosslessExecutor(ROOT/'tests/fixtures/qmm_lossless_validation.json')).to_dict()
    validation.update(origin='SYNTHETIC_FIXTURE_NOT_LIVE_MODEL',current_arm_allowed=False)
    manifest={'schema':'INSPECT_REPLAY_V1','model_requests_sent':0,'native_library_loaded':False,
              'historical_candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'status':'LOCAL_ADAPTERS_TESTED_LIVE_DEFERRED',
              'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'src/ascend_fi/inspect').glob('*.py')}}
    for name,value in [('recorded_candidate',candidate),('native_fixture',native),('typed_probe_fixture',typed),
                       ('fixture_validation_result',validation),('inspect_session_manifest',manifest)]:
        with (a.output/(name+'.json')).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,ensure_ascii=False)
    print(json.dumps({'output':str(a.output),'status':manifest['status']}))


if __name__=='__main__': main()
