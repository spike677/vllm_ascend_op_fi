"""Independent T2 receipt review; no injection log is used to infer recovery."""
import argparse,hashlib,json,re
from pathlib import Path


def review(data):
    issues=[]
    if data.get('status')!='REFERENCE_CONTROL_REFUSAL_A2_PASSED':issues.append('RUN_NOT_COMPLETE')
    before=data.get('before',{});ranks=before.get('ranks',[])
    if len(ranks)!=2 or [r.get('rank') for r in ranks]!=[0,1]:issues.append('TP2_RANKS_REQUIRED')
    for r in ranks:
        steps=r.get('steps',[])
        if r.get('removed') is not True or len(steps)!=4:issues.append('FOUR_STEPS_AND_REMOVED_OBSERVER_REQUIRED')
        for s in steps:
            if any(not re.fullmatch('[a-f0-9]{64}',str(s.get(k,''))) for k in ('hidden','logits')):
                issues.append('FULL_ARRAY_HASH_REQUIRED')
    expected={'missing_site':'REFERENCE_NO_REAL_BF16_WEIGHT',
              'identity_drift':'EXTERNAL_DEPLOYMENT_IDENTITY_MISMATCH',
              'range':'PRODUCT_ONLY_ALL_REQUESTS_SUPPORTED',
              'missing_site_repeat':'REFERENCE_NO_REAL_BF16_WEIGHT'}
    checks=data.get('checks',[])
    if [c.get('kind') for c in checks]!=list(expected):issues.append('NEGATIVE_SEQUENCE_REQUIRED')
    for c in checks:
        reply=c.get('reply',{});codes=reply.get('refusal_codes',[reply.get('code')])
        if reply.get('exit_code')!=4 or expected.get(c.get('kind')) not in codes:issues.append('STRUCTURED_REFUSAL_REQUIRED')
        if c.get('kind')!='range' and [r.get('rank') for r in reply.get('refusals',[])]!=[0,1]:issues.append('ALL_RANK_REFUSAL_REQUIRED')
        if c.get('canary_exact') is not True or c.get('after')!=before or c.get('engine_alive') is not True:
            issues.append('POST_REFUSAL_CANARY_DIFFERS')
        audit=c.get('audit',[])
        if len(audit)!=2 or [r.get('rank') for r in audit]!=[0,1] or any(
            type(r.get(k)) is not int or r[k]!=0 for r in audit for k in ('leases','owners','bindings','wrappers')):
            issues.append('RESOURCES_REMAIN_OR_UNOBSERVED')
        if c.get('client_status',{}).get('enabled') is not False:issues.append('CLIENT_LEFT_ENABLED')
    return dict(schema='dcat.reference.refusal.review.v1',passed=not issues,issues=sorted(set(issues)),
        checks=[dict(kind=c.get('kind'),exit_code=c.get('reply',{}).get('exit_code'),
            canary_exact=c.get('after')==before) for c in checks],qualification_granted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.input.read_bytes();result=review(json.loads(raw))
    result['receipt_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)
