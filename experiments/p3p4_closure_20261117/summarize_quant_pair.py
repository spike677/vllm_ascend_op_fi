"""Auxiliary exact Q/scale evidence, separate from raw-only first difference.

Run on CPU against our own private small tensors. Does not select a first
difference or consume native hit logs to modify a statistics conclusion.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--fault',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();summary=dict(schema='demoncat.quant.per_request_summary.v1',models={})
    phases={}
    for arm,case in (('baseline',a.baseline),('fault',a.fault)):
        result=json.loads((case/'result.json').read_bytes());runtime=json.loads((case/'serving/runtime/serving.json').read_bytes())
        if result['status']!='PASS_COMMON_B8_QUANT_BFOFF' or not runtime['recovered'] or runtime['error']:raise RuntimeError('INCOMPLETE_REAL_RUN')
        phases[arm]={x['phase']:x for x in runtime['rows']}
        item=dict(actual_model_class=result['setup']['model_class'],model_type=result['setup']['model_type'],
            automatic_selected_module=runtime['healthy']['path'],candidates=runtime['candidate_count'],
            scope=runtime['scope'],capture_scope=runtime.get('capture_scope','generation'),
            native_build=runtime.get('native_build'),core_methods_replaced=runtime['core_methods_replaced'],
            recovered=runtime['recovered'],case=str(case),phases=[],
            hashes={n:sha(case/n) for n in ('result.json','serving/runtime/serving.json')})
        for row,host in zip(runtime['rows'],result['phases']):
            actual=row['decode']['decode'];mapping=host['mapping']
            if set(actual['request_ids'])!=set(host['enqueued']) or set(host['enqueued'])!={x['internal_request_id'] for x in mapping}:
                raise RuntimeError('ACTUAL_SCHEDULER_DRIVER_MAPPING_MISMATCH')
            if {x['request_id'] for x in host['outputs']}!={x['external_request_id'] for x in mapping}:raise RuntimeError('COMPLETION_MAPPING_MISMATCH')
            events=[] if row['native'] is None else [x for x in row['native']['events'] if x['status']=='routed']
            item['phases'].append(dict(phase=row['phase'],top1=row['decode']['top1'],native_calls=len(events),
                native_receipts=[e['native_receipt'] for e in events],
                actual_native_execution_proven=all(e['native_execution_proven'] for e in events) if events else None,
                actual_native_fault_hits=sum(e['native_hit_proven'] for e in events),
                hidden_sha256=row['decode']['hidden_sha256'],logits_sha256=row['decode']['logits_sha256'],
                prefill_logits_sha256=row['prefill']['logits_sha256'],actual_request_mapping=True,
                generated=[dict(tokens=x['tokens'],text=x['text']) for x in host['outputs']]))
        summary['models'][arm]=item
    b,f=phases['baseline']['comparison'],phases['fault']['comparison']
    if b['quant']['input_sha256']!=f['quant']['input_sha256']:raise RuntimeError('B_F_QUANT_INPUT_DIFFERS')
    for key in ('logical_ids','row_token_positions','prefixes','input_ids'):
        if b['quant']['rows'][key]!=f['quant']['rows'][key]:raise RuntimeError('B_F_ROW_MAPPING_DIFFERS:'+key)
    tensors=[]
    for case in (a.baseline,a.fault):tensors.append(torch.load(case/'serving/runtime/comparison-quant.pt',map_location='cpu',weights_only=True))
    bt,ft=tensors
    if list(bt['scale'].shape)!=[8] or bt['q'].shape!=ft['q'].shape:raise RuntimeError('Q_SCALE_GEOMETRY_DIFFERS')
    summary['row_evidence']=[]
    for i in range(8):
        delta=(ft['q'][i].to(torch.int16)-bt['q'][i].to(torch.int16))
        summary['row_evidence'].append(dict(logical_request=b['quant']['rows']['logical_ids'][i],
            actual_tensor_row=i,token_position=b['quant']['rows']['row_token_positions'][i],
            baseline_scale=float(bt['scale'][i]),fault_scale=float(ft['scale'][i]),
            q_changed_elements=int(torch.count_nonzero(delta)),q_elements=delta.numel(),
            q_max_abs_integer_change=int(delta.abs().max())))
    summary.update(input_sha256=b['quant']['input_sha256'],
        fault_scales_all_equal=bool((ft['scale']==ft['scale'][0]).all()),
        fault_scale_equals_baseline_largest=bool((ft['scale']==bt['scale'].max()).all()),
        complete_q_changed=not torch.equal(bt['q'],ft['q']),
        source='AUXILIARY_REAL_Q_SCALE_AND_RUNTIME_MAPPING_NOT_INDEPENDENT_FIRST_DIFFERENCE',
        common_worker_all_eligible=False,raw_data_published=False)
    with a.out.open('x') as file:json.dump(summary,file,indent=2);file.write('\n')
    print(json.dumps(dict(rows=summary['row_evidence'],shared_scale=summary['fault_scales_all_equal'])))


if __name__=='__main__':main()
