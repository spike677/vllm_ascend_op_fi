"""CPU-only preparation for the explicitly constructed B3/B4/B5 research driver."""
import argparse
import json
from pathlib import Path
import sys
from experiments.worker_core_v11_20261118 import prepare_tnd28_prepared_input as base


def prepare(args):
    cases,expected=base.prepare(args);case=cases[0]
    build=json.loads((args.build_root/'evidence/completion.json').read_bytes())
    candidate=build.get('tnd28_two_stage_candidate',{})
    base.E.need(candidate.get('prepare_symbol')=='dcat_tnd28_replay_prepare_gate_v2'
        and candidate.get('copy_launch_symbol')=='dcat_tnd28_copy_launch_read_v1'
        and candidate.get('native_qualified') is False,'TWO_STAGE_CANDIDATE_BUILD_REQUIRED')
    observer=base.E.observer_bundle(args.observer,base.frozen(args.source,args.source_manifest_sha256))
    receipt=json.loads((Path(observer['folder'])/'result.json').read_bytes())
    base.E.need(receipt.get('observer_launch_readback_selftest',{}).get('exit_code')==0,'CURRENT_LAUNCH_OBSERVER_REQUIRED')
    old='natural_tnd28_prepared_input_probe.py';new='natural_tnd28_two_stage_probe.py'
    base.E.need(case['command'].count(old)==1,'ONE_RESEARCH_PROBE_ENTRY')
    case['id']='tnd28-two-stage-two-scopes';case['command']=case['command'].replace(old,new)
    review=case['review'];review.update(experiment_id=case['id'],
        provider='28.fia.denominator_replay / explicit two-stage research provider, not public Runtime',
        worker='same standard Qwen3 TP1 eager Worker; two distinct owners; actual phase mapping',
        expected_hits='Exactly one selected FIA replay in each fault request; two independent scopes; all other calls original',
        research_requested_fault=True,public_fault_authorized=False,
        fault_authorized=False,independent_review_required=True,
        recovery_plan='Actual OFF complete hidden/logits must equal baseline; remove Attention/API interception before '
          'health using three read-only root/logits observers; finish/unregister/detach; preserve owner on uncertainty.')
    review['resource_budget'].update(observed_requests=14,ordinary_health_requests=0,compiled_input_records=192,
        compiled_input_bytes=8*1024**2)
    review['request_plan']=['observe','observe_repeat','baseline','baseline_repeat','fault','off','health']
    review['scope_count']=2
    return base.validate_cases(cases,[args.card]),expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','installed','observer','model','build-root','vendor','libraries','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','build-sha256','libraries-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--card',type=int,choices=(2,5),default=2);p.add_argument('--timeout',type=int,default=1500)
    a=p.parse_args();sys.dont_write_bytecode=True;sys.addaudithook(base.E.prohibit_runtime)
    target=a.out.with_name(a.out.stem+'-expected-libraries.json')
    base.E.need(not a.out.exists() and not target.exists(),'FRESH_RESEARCH_PREPARATION')
    cases,expected=prepare(a)
    for path,value in ((a.out,cases),(target,expected)):
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='TWO_STAGE_RESEARCH_PREPARED_NOT_RUN',cases_sha256=base.E.sha(a.out),
        expected_libraries_sha256=base.E.sha(target),npu_calls=0,qualification_granted=False)))

if __name__=='__main__':main()
