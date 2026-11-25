"""Bounded text pages for private DeepSeek noise and read-only observations."""
import argparse,json,sys
from pathlib import Path
from tools.extract_registration_record import canonical,sha,file_id,get

def pages(kind,path):
    raw=Path(path).read_bytes();data=json.loads(raw);base=dict(schema='dcat.dsv4.v030.extract.v1',kind=kind,file=file_id(path,raw),qualification_granted=False)
    payloads=[]
    if kind=='noise':
        # Existing calibration summary; no full hidden/logits arrays leave yellow.
        if data.get('schema')!='demoncat.dsv4_0731.v030.noise.v1':raise ValueError('NOISE_SUMMARY_SCHEMA')
        from fault_injection.dsv4_noise_v030 import identity,METRICS
        error=None
        try:identity(data.get('combination',{}))
        except Exception as e:error=str(e)
        for rank in sorted(set(data.get('bounds',{}))|set(data.get('observed_maxima',{})),key=int):
            maximum=data.get('observed_maxima',{}).get(rank);bounds=data.get('bounds',{}).get(rank)
            rows=lambda x:[[r.get(k) for k in METRICS] for r in x] if isinstance(x,list) else None
            payloads.append(dict(rank=int(rank),combination=data.get('combination'),identity_error=error,
                calibration_sha256=data.get('calibration_sha256'),prompt=[data.get('prompt_sha256'),data.get('prompt_tokens')],
                tokens=data.get('tokens'),runs=data.get('runs'),process_sample_counts=sorted(data.get('process_counts',{}).values()),
                safety_factor=data.get('safety_factor'),maxima=rows(maximum),bounds=rows(bounds),
                finite=data.get('all_finite'),token_stable=data.get('tokens_stable'),
                limitation='Summary fields missing remain null; finite/token stability requires local raw recheck.'))
    elif kind=='binding':
        reports=data if isinstance(data,list) else [data]
        for report in reports:
            rpc=report if report.get('schema')=='dcat.probe.rpc.v2' else {}
            if rpc:report=rpc.get('binding') or rpc.get('result',{})
            if report.get('schema')!='dcat.probe.bindings.v2':raise ValueError('BINDING_SCHEMA')
            common=dict(rank=report.get('rank'),rpc_status=rpc.get('status'),rpc_code=rpc.get('code'),
                observer_state_present=report.get('observer_state_present'),capture_state_present=report.get('capture_state_present'),read_only=report.get('read_only'))
            for key,value in report.get('objects',{}).items():
                payloads.append(dict(common,object_name=key,identity=value,target_paths=report.get('target_paths')))
            for key,value in report.get('bindings',{}).items():
                if value is None:payloads.append(dict(common,method=key,identity=None));continue
                compact={k:v for k,v in value.items() if k not in ('callable_chain','pinned_original_candidates')}
                payloads.append(dict(common,method=key,identity=compact))
                for index,item in enumerate(value.get('callable_chain',[])):
                    payloads.append(dict(common,method=key,chain_index=index,callable=item))
                for index,item in enumerate(value.get('pinned_original_candidates',[])):
                    payloads.append(dict(common,method=key,pin_index=index,pinned_original=item))
    elif kind=='identity':
        reports=data if isinstance(data,list) else [data]
        for report in reports:
            if report.get('schema')=='dcat.probe.rpc.v2':report=report.get('result',{})
            if report.get('schema')!='dcat.probe.d0.v2':raise ValueError('IDENTITY_SCHEMA')
            common=dict(rank=report.get('rank'),worker_pid=report.get('worker_pid'),baseline_approved=False)
            payloads.append(dict(common,parallel_configs=report.get('parallel_configs'),groups=report.get('groups')))
            for key,value in report.get('objects',{}).items():payloads.append(dict(common,object_name=key,identity=value))
            for key,value in report.get('methods',{}).items():payloads.append(dict(common,method=key,identity=value))
    else:
        reports=data if isinstance(data,list) else [data]
        for report in reports:
            if report.get('schema')=='dcat.probe.rpc.v2':report=report.get('result',{})
            if 'files' in report:
                for item in report['files']:payloads.append(dict(rank=report.get('rank'),source_file=item))
                continue
            if report.get('schema')!='demoncat.dsv4_0731.observe.v1':raise ValueError('OBSERVE_SCHEMA')
            for mechanism,effect in report.get('mechanisms',{}).items():
                common=dict(rank=report.get('rank'),mechanism=mechanism,read_only=report.get('read_only'),injected_calls=report.get('injected_calls'),
                    steps=report.get('steps'),unknown_phase_steps=report.get('unknown_phase_steps'),observer_removed=report.get('observer_removed'),
                    totals=[effect.get('observed'),effect.get('eligible')],unavailable_reason=effect.get('unavailable_reason'))
                sites=effect.get('sites',[])
                for i,site in enumerate(sites):
                    # Internal IDs can be long and private; preserve count+canonical hash.
                    compact={k:v for k,v in site.items() if k!='request_ids'}
                    ids=site.get('request_ids');compact['request_association']=[len(ids),sha(canonical(ids))] if isinstance(ids,list) else None
                    payloads.append(dict(common,site_index=i,site_count=len(sites),site=compact))
                if not sites:payloads.append(dict(common,site_count=0,site=None))
            if 'calls' in report:
                payloads.append(dict(rank=report.get('rank'),calls=len(report['calls']),truncated=report.get('truncated'),
                    observation_errors=report.get('observation_errors'),combine_output_row_join=report.get('combine_output_row_join')))
                for call in report['calls']:
                    if not any(x in call['operator'].lower() for x in ('dispatch','combine','routing')):continue
                    compact={k:call.get(k) for k in ('sequence','step','operator','layer','phase','rank','groups','padding','valid_output_rows','dispatch_candidates','combine_output_row_join','join_reason')}
                    rows=call.get('input_rows',{})
                    compact['input_rows']=dict(complete=rows.get('complete'),reason=rows.get('reason'),
                        count=len(rows.get('rows',[])),sha256=sha(canonical(rows.get('rows'))))
                    payloads.append(dict(rank=report.get('rank'),call=compact))
                    for side in ('inputs','outputs'):
                        for tensor in call.get(side) or []:
                            item={k:v for k,v in tensor.items() if k not in ('values',)}
                            item['values_recorded']=tensor.get('values') is not None
                            payloads.append(dict(rank=report.get('rank'),call_sequence=call['sequence'],side=side,tensor=item))
    if not payloads:raise ValueError('NO_EXTRACTABLE_RECORDS')
    output=[]
    for i,payload in enumerate(payloads):
        page=dict(base,page=i,pages=len(payloads),payload=payload);page['extract_sha256']=sha(canonical(page))
        if len(canonical(page))+1>4096:raise ValueError('DSV4_PAGE_EXCEEDS_4096_NO_TRUNCATION')
        output.append(page)
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['noise','observe','identity','binding']);p.add_argument('input');a=p.parse_args()
    # Validate all pages before emitting any.
    result=pages(a.kind,a.input)
    for page in result:sys.stdout.buffer.write(canonical(page)+b'\n')
if __name__=='__main__':main()
