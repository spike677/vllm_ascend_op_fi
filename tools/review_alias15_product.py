"""Independent Fault15 product receipt review; no runtime injection logs needed."""
import argparse,hashlib,json
from pathlib import Path
from tools.review_product_mode import review as product_review

def review(data,*,expected_tp=2,require_a3=False):
    generic=product_review(data,expected_tp=expected_tp)
    if data['mechanism']!='15.alias_detach':raise ValueError('ALIAS15_MECHANISM_REQUIRED')
    scopes=[]
    for scope in data['scopes']:
        ranks={i:dict(calls=0,changed=0,changed_elements=0,initial_nonfinite_calls=0) for i in range(expected_tp)}
        for request in scope['requests'][:3]:
            for rank in request['result']['ranks']:
                proof=rank.get('alias_detach_effect',{});calls=proof.get('calls',[])
                def need(ok):
                    if not ok:raise ValueError('ALIAS15_INDEPENDENT_MECHANISM_PROOF_FAILED')
                need(proof.get('semantics')=='ND_VIEW_ALIAS_DETACH_NO_PRIVATE_FORMAT_TRIGGER' and proof.get('pending')==0)
                if require_a3:
                    identity=proof.get('runtime_identity') or {}
                    need(expected_tp==2 and identity.get('source')=='YELLOW_DEDICATED_INSTANCE' and identity.get('tp')==2)
                    need(identity.get('soc',{}).get('observed_name')=='Ascend910_9382' and identity.get('soc',{}).get('family')=='A3')
                    need(identity.get('toolkit_version')=='9.1.0' and identity.get('cann_observation_source')=='VERIFIED_MAPPED_ELF_LINKED_TO_INSTALLED_VERSION_METADATA')
                    libs=identity.get('loaded_runtime',[])
                    need(len(libs)==2 and {r.get('name') for r in libs}=={'libascendcl.so','libruntime.so'})
                    need(all(r.get('version')=='9.1.0' and all(isinstance(r.get(k),str) and len(r[k])==64 and all(c in '0123456789abcdef' for c in r[k]) for k in ('sha256','metadata_sha256')) for r in libs))
                need(len(calls)==rank['coverage']['eligible'] and len(calls)>0)
                need(rank['changed_elements']==sum(e.get('changed_elements',-1) for e in calls))
                need(rank['coverage']['changed']==sum(e.get('changed',-1) for e in calls))
                for e in calls:
                    need(e.get('request_ids')==[rank['request_id']] and e.get('dtype')=='torch.bfloat16' and e.get('format')==2)
                    need(e.get('initial_view_exact') is True and e.get('detached_storage') is True)
                    need(e.get('before_sha256') is not None and e['before_sha256']==e.get('after_sha256')==e.get('restored_sha256')==e.get('gate_input_sha256'))
                    need(e.get('gate_result_sha256') is not None and e['gate_result_sha256']==e.get('consumer_sha256'))
                    need(e.get('gate_device_read_completed') and e.get('projection_reads_exact_produced_value') and e.get('device_consumer_completed'))
                    need(e.get('restore_kind')=='UNCHANGED_BASE_BYTES_TEMPORARY_RELEASED')
                    need(type(e.get('changed_elements')) is int and e['changed_elements']>=0 and e.get('changed')==int(e['changed_elements']>0))
                    need((e['intended_sha256']!=e['after_sha256'])==(e['changed_elements']>0))
                    row=ranks[rank['rank']];row['calls']+=1;row['changed']+=e['changed'];row['changed_elements']+=e['changed_elements'];row['initial_nonfinite_calls']+=int(e.get('initial_all_finite') is not True)
        scopes.append(dict(scope=scope['scope'],ranks=ranks,all_rank_resources_zero=True,new_request_canary_exact=True))
    return dict(status='ALIAS15_PRODUCT_MECHANISM_REVIEW_PASSED',source_manifest_sha256=data['source_manifest_sha256'],semantics='ND_VIEW_ALIAS_DETACH_NO_PRIVATE_FORMAT_TRIGGER',scopes=scopes,qualification_granted=False,product_closed=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('--tp',type=int,choices=(1,2),default=2);p.add_argument('--output');p.add_argument('--a3',action='store_true');a=p.parse_args();raw=Path(a.receipt).read_bytes();result=review(json.loads(raw),expected_tp=a.tp,require_a3=a.a3)
    result.update(receipt_sha256=hashlib.sha256(raw).hexdigest(),reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    text=(json.dumps(result,indent=2)+'\n').encode()
    if a.output:Path(a.output).write_bytes(text)
    else:print(text.decode(),end='')
if __name__=='__main__':main()
