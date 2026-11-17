"""Summarize actual token feedback, KV lifetime and native-qualified scope.

This is runtime evidence, not an input to the independent statistics comparator.
No private arrays, raw statistics or serialized tiling bytes are published here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--fault',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    summary=dict(schema='demoncat.paged33.continuous_summary.v1',arms={},raw_data_published=False,
        common_vllm_worker=False,common_worker_all_eligible=False,
        statistics_first_difference_source=False)
    full={};inputs={}
    for arm,case in (('baseline',a.baseline),('fault',a.fault)):
        result=json.loads((case/'result.json').read_bytes());full[arm]=result
        if result['status']!='PASS_HF_CONTINUOUS_PAGED_GENERATION_NOT_COMMON_WORKER':raise RuntimeError('INCOMPLETE_RUN')
        phases=[]
        for row in result['phases']:
            n=result['decode_steps'];layers=len(row['coverage']['candidates'])
            if (len(row['steps'])!=n or row['native_policy_calls']!=layers
                    or row['stock_followup_calls']!=layers*(n-1)
                    or not row['coverage']['all_discovered_candidates_covered']
                    or not row['coverage']['cleanup_completed'] or not row['inputs_unchanged']):
                raise RuntimeError('INCOMPLETE_RUNTIME_COVERAGE')
            if n>1 and not row['stock_routing']['all_discovered_candidates_routed']:raise RuntimeError('STOCK_ROUTING_INCOMPLETE')
            previous=row['initial_tokens']
            for i,step in enumerate(row['steps']):
                if (step['input_tokens']!=previous or step['cache_length_before']!=16383+i
                        or step['cache_length_after']!=16384+i or not step['finite']):raise RuntimeError('TOKEN_FEEDBACK_OR_CACHE_INVALID')
                previous=step['output_tokens']
            keep=('phase','initial_tokens','steps','native_policy_calls','actual_policy_applied',
                'stock_followup_calls','following_lengths','inputs_unchanged','generated_tokens',
                'generated_text','complete_logits_equal_reference','per_request_max_abs','generated_tokens_equal_reference')
            phases.append({k:row[k] for k in keep if k in row})
            phases[-1].update(fault_scope_candidates=layers,
                fault_scope_forwards=len(row['coverage']['forwards']),
                fault_scope_cleanup_completed=row['coverage']['cleanup_completed'],
                following_scope='STOCK_CALLS_NOT_NEW_FAULT_QUALIFICATION')
            matches=[]
            for receipt in (case/(row['phase']+'_native')).glob('*-request_mapping.json'):
                d=json.loads(receipt.read_bytes())
                if d['details']['cache_layer']==0:matches.append(d['details'])
            if len(matches)!=1:raise RuntimeError('FIRST_LAYER_INPUT_AMBIGUOUS')
            inputs[arm,row['phase']]=matches[0]
        summary['arms'][arm]=dict(actual_model_class=result['actual_model_class'],case=str(case),
            result_sha256=sha(case/'result.json'),input_sha256=result['input_sha256'],
            prefill_length=result['prefill_length'],decode_steps=result['decode_steps'],phases=phases)
    if full['baseline']['input_sha256']!=full['fault']['input_sha256']:raise RuntimeError('B_F_PREFILL_DIFFERS')
    for key in ('query_sha256','kv','table','teacher_tokens','geometry'):
        if inputs['baseline','comparison'][key]!=inputs['fault','comparison'][key]:raise RuntimeError('FIRST_NATIVE_INPUT_DIFFERS:'+key)
    ref_b,ref_f=(full[arm]['phases'][0] for arm in ('baseline','fault'))
    if [x['logits_sha256'] for x in ref_b['steps']]!=[x['logits_sha256'] for x in ref_f['steps']]:raise RuntimeError('B_F_HEALTHY_TRAJECTORY_DIFFERS')
    summary.update(status='PASS_BOUNDED_HF_GENERATION_RECOVERY_NOT_COMMON_WORKER',
        first_actual_paged_ifa_input_equal=True,baseline_reference_trajectory_equal_across_processes=True,
        limits=['Only the already qualified KV16384 calls receive Fault33; KV16385..16387 use stock paged IFA.',
            'Legacy receipt key teacher_tokens contains actual preceding argmax tokens in this experiment.',
            'Statistics, actual native launch and runtime propagation remain separate acceptance layers.',
            'No claim of arbitrary shape, pretrained task accuracy or default SDPA equivalence.'])
    with a.out.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=summary['status'],out=str(a.out))))


if __name__=='__main__':main()
