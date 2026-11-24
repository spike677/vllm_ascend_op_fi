"""D-package product evidence under the fixed V4 noise rule, never a grant.

Full phase arrays and calibration stay local. The trusted deployment adapter
must collect them from real requests; API top5 cannot stand in for full logits.
Qwen keeps the existing exact tool. This module does not change product gates.
"""
import argparse,json
from pathlib import Path
from tools.product_effect_probe import canonical,sha,require,COUNTERS
from fault_injection.dsv4_noise import RULE,calibrate,review as review_noise


def run(adapter,prompt,combination,calibration,*,checkpoint=lambda value:None):
    """capture returns the existing noise.v1 full-array run plus provenance.

    The adapter owns authenticated control and request observation. No permits,
    Worker creation or new fault state machine are implemented here.
    """
    calibrate(combination,calibration)
    require(adapter.identity()==combination,'NOISE_PRODUCT_DEPLOYMENT_CHANGED')
    raw=dict(schema='dcat.effect.noise.raw.v1',identity=combination,rule=RULE,scopes=[],
             tool_sha256=sha(Path(__file__).read_bytes()),qualification_granted=False)
    for _ in range(2):
        session=adapter.new_session()
        s=dict(owner=session.owner,worker_identity=session.worker_identity,phases={},controls=[])
        raw['scopes'].append(s)
        def call(name):
            try:
                value=getattr(session,name)();s['controls'].append(dict(action=name,result=value));return value
            except Exception as error:
                s['controls'].append(dict(action=name,error=str(error)));raise
            finally:checkpoint(raw)
        def capture(phase):
            s['phases'][phase]=adapter.capture_noise(phase,prompt,dict(temperature=0,seed=42,max_tokens=4),session)
            checkpoint(raw)
        call('attach')
        try:
            capture('normal');call('enable');capture('fault');call('status')
        finally:call('disable')
        capture('off');capture('health');call('detach');s['evidence']=session.evidence();checkpoint(raw)
    return raw


def summarize(raw,raw_bytes,calibration,calibration_bytes):
    require(raw.get('schema')=='dcat.effect.noise.raw.v1' and raw.get('rule')==RULE,'NOISE_PRODUCT_SCHEMA')
    scopes=raw.get('scopes',[]);identity=raw['identity']
    require(len(scopes)==2 and len({s['owner'] for s in scopes})==2,'TWO_OWNERS_REQUIRED')
    require(scopes[0]['worker_identity']==scopes[1]['worker_identity'],'WORKERS_CHANGED')
    workers=scopes[0]['worker_identity']
    require(sorted(w.get('rank',-1) for w in workers)==list(range(8)),'NOISE_ALL_WORKERS_REQUIRED')
    worker_sha=sha(canonical(workers));packed=[];processes=set();provider_owners=set()
    from tools.extract_registration_record import DEPLOYMENT
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_identity import registration_identity_issues
    issues=registration_identity_issues(identity)
    issues.extend('IDENTITY_MISSING:'+k for k in DEPLOYMENT if k!='image_digest' and identity.get(k) is None)
    for s in scopes:
        phases=s['phases'];proof=review_noise(identity,calibration,phases)
        require(proof['off_health_passed'],'NOISE_OFF_HEALTH_OR_STORAGE_FAILED')
        requests=[]
        for phase,p in phases.items():
            require(p.get('owner')==s['owner'] and p.get('worker_identity_sha256')==worker_sha,
                    'NOISE_PHASE_OWNER_WORKER_MISMATCH')
            require(isinstance(p.get('request_id'),str) and p['request_id'] and p.get('evidence_paths'),
                    'NOISE_REAL_REQUEST_PROVENANCE_REQUIRED')
            requests.append(p['request_id']);processes.add(p['process_identity'])
        require(len(set(requests))==4,'NOISE_PHASE_REQUEST_REUSED')
        ranks=s['evidence']['ranks'];rows=[];inactive=[]
        require(sorted(r.get('rank',-1) for r in ranks)==list(range(8)),'NOISE_RANK_EVIDENCE_REQUIRED')
        for r in sorted(ranks,key=lambda r:r['rank']):
            c=r['counts'];po=r.get('provider_owner')
            require(r.get('owner')==s['owner'] and isinstance(po,str) and po and po not in provider_owners,
                    'NOISE_PROVIDER_OWNER_MISMATCH')
            provider_owners.add(po)
            require(phases['fault']['request_id'] in r.get('request_ids',[]) and r.get('evidence_paths'),
                    'NOISE_FAULT_CONSUMER_ASSOCIATION_REQUIRED')
            require(all(type(c.get(k)) is int and c[k]>=0 for k in COUNTERS),'EIGHT_COUNTERS_REQUIRED')
            require(c['observed']>=c['eligible']>=0 and c['eligible']==c['attempted']==c['applied']==c['consumed']==c['restored']
                    and c['failed']==0 and c['changed']<=c['applied'],'COUNTER_CONTRADICTION')
            require(type(r.get('changed_elements')) is int and r['changed_elements']>=0,'CHANGED_ELEMENTS_REQUIRED')
            require(all(r.get(k) is True for k in ('bytes_restored','resources_zero')),'STRICT_RECOVERY_OR_CONSUMER_FAILED')
            if c['eligible']:
                require(r.get('consumer') is True,'STRICT_RECOVERY_OR_CONSUMER_FAILED')
            else:
                reason=r.get('not_applicable_reason')
                require(isinstance(reason,str) and reason and r.get('consumer') in (None,False)
                        and r['changed_elements']==0,'NOISE_ZERO_CALL_RANK_PROOF_REQUIRED')
                inactive.append([r['rank'],reason])
            rows.append([r['rank'],*[c[k] for k in COUNTERS],r['changed_elements']])
        require(sum(r[2] for r in rows)>0,'NOISE_NO_ELIGIBLE_CALLS')
        packed.append(dict(owner=s['owner'],ranks=rows,model_effect_visible=proof['model_effect_visible'],
                           off_health_within_noise=True,strict_storage_recovery=True,
                           inactive_ranks=inactive,review_sha256=sha(canonical(proof))))
    require(len(processes)==1,'NOISE_TWO_OWNERS_PROCESSES_CHANGED')
    page=dict(schema='dcat.effect.noise.page.v1',rule=RULE,identity=identity,scopes=packed,
              original=[len(raw_bytes),sha(raw_bytes)],calibration=[len(calibration_bytes),sha(calibration_bytes)],
              tool_sha256=raw.get('tool_sha256'),reviewer_sha256=sha(Path(__file__).read_bytes()),
              worker_identity_sha256=worker_sha,identity_issues=issues,evidence_passed=not issues,
              registration_ready=False,qualification_granted=False)
    page['extract_sha256']=sha(canonical(page));require(len(canonical(page))+1<=4096,'PAGE_EXCEEDS_4096_NO_TRUNCATION')
    return page


def verify(page):
    d=dict(page);value=d.pop('extract_sha256',None)
    require(sha(canonical(d))==value,'EXTRACT_SHA_MISMATCH')
    require(len(canonical(page))+1<=4096 and page.get('schema')=='dcat.effect.noise.page.v1'
            and page.get('rule')==RULE,'NOISE_PAGE_SCHEMA_OR_SIZE')
    require(page.get('registration_ready') is False and page.get('qualification_granted') is False,'NOISE_PAGE_CANNOT_GRANT')
    require(len(page['scopes'])==2 and len({s['owner'] for s in page['scopes']})==2,'TWO_OWNERS_REQUIRED')
    from tools.extract_registration_record import DEPLOYMENT
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_identity import registration_identity_issues
    issues=registration_identity_issues(page['identity'])
    issues.extend('IDENTITY_MISSING:'+k for k in DEPLOYMENT if k!='image_digest' and page['identity'].get(k) is None)
    require(issues==page['identity_issues'] and page['evidence_passed']==(not issues),'IDENTITY_CONCLUSION_CONTRADICTORY')
    for s in page['scopes']:
        require(s['off_health_within_noise'] is True and s['strict_storage_recovery'] is True,'NOISE_RECOVERY_UNPROVEN')
        require([r[0] for r in s['ranks']]==list(range(8)),'NOISE_RANK_EVIDENCE_REQUIRED')
        for r in s['ranks']:
            require(len(r)==10 and all(type(x) is int and x>=0 for x in r),'COUNTER_ROW_INVALID')
            _,o,e,t,a,c,u,z,f,n=r
            require(o>=e>=0 and e==t==a==u==z and f==0 and c<=a and (a>0 or n==0),'COUNTER_CONTRADICTION')
        inactive=s.get('inactive_ranks',[])
        require([r[0] for r in inactive]==[r[0] for r in s['ranks'] if r[2]==0]
                and all(len(r)==2 and isinstance(r[1],str) and r[1] for r in inactive),'NOISE_ZERO_CALL_RANK_PROOF_REQUIRED')
        require(sum(r[2] for r in s['ranks'])>0,'NOISE_NO_ELIGIBLE_CALLS')
    return dict(integrity_valid=True,evidence_passed=page['evidence_passed'],registration_ready=False,qualification_granted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw');p.add_argument('--calibration');p.add_argument('--verify');a=p.parse_args()
    if a.verify:print(canonical(verify(json.loads(Path(a.verify).read_bytes()))).decode())
    else:
        require(a.raw and a.calibration,'RAW_AND_CALIBRATION_REQUIRED')
        raw=Path(a.raw).read_bytes();cal=Path(a.calibration).read_bytes();data=json.loads(cal)
        value=json.loads(raw);require(data['combination']==value['identity'],'CALIBRATION_COMBINATION_MISMATCH')
        print(canonical(summarize(value,raw,data['calibration'],cal)).decode())
