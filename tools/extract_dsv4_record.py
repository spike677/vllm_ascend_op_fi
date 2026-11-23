"""Bounded text pages for private DeepSeek noise and read-only observations."""
import argparse,json,sys
from pathlib import Path
from tools.extract_registration_record import canonical,sha,file_id,get

def pages(kind,path):
    raw=Path(path).read_bytes();data=json.loads(raw);base=dict(schema='dcat.dsv4.extract.v1',kind=kind,file=file_id(path,raw),qualification_granted=False)
    payloads=[]
    if kind=='noise':
        # Existing calibration summary; no full hidden/logits arrays leave yellow.
        if data.get('schema')!='demoncat.dsv4_0731.noise.v1':raise ValueError('NOISE_SUMMARY_SCHEMA')
        from fault_injection.dsv4_noise import identity,METRICS
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
    else:
        reports=data if isinstance(data,list) else [data]
        for report in reports:
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
    if not payloads:raise ValueError('NO_EXTRACTABLE_RECORDS')
    output=[]
    for i,payload in enumerate(payloads):
        page=dict(base,page=i,pages=len(payloads),payload=payload);page['extract_sha256']=sha(canonical(page))
        if len(canonical(page))+1>4096:raise ValueError('DSV4_PAGE_EXCEEDS_4096_NO_TRUNCATION')
        output.append(page)
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['noise','observe']);p.add_argument('input');a=p.parse_args()
    # Validate all pages before emitting any.
    result=pages(a.kind,a.input)
    for page in result:sys.stdout.buffer.write(canonical(page)+b'\n')
if __name__=='__main__':main()
