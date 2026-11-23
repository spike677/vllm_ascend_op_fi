"""Review local product receipt and print rank/scope text pages, not a grant."""
import argparse,json,sys
from pathlib import Path
from tools.review_product_mode import review
from tools.extract_registration_record import canonical,sha,DEPLOYMENT,file_id
from demoncat.runtime.vllm_ascend.worker_extension.deployment_identity import registration_identity_issues


def pages(receipt,identity):
    raw=Path(receipt).read_bytes();iraw=Path(identity).read_bytes();d=json.loads(raw);actual=json.loads(iraw)
    issues=registration_identity_issues(actual)
    missing=[k for k in DEPLOYMENT if k!='image_digest' and actual.get(k) is None]
    if missing:issues.append('IDENTITY_MISSING:'+','.join(missing))
    error=None;result=None
    try:result=review(d,expected_tp=actual.get('tp'),hardware_scope='DECLARED_RUNTIME_IDENTITY_REQUIRES_BLUE_REVIEW',
                      require_native_runtime=actual.get('hardware_family')=='A3')
    except Exception as e:error=type(e).__name__+':'+str(e)
    common=dict(schema='dcat.product.extract.v1',mechanism=d.get('mechanism'),source_manifest_sha256=d.get('source_manifest_sha256'),
        identity={k:actual.get(k) for k in DEPLOYMENT},identity_source='YELLOW_RUNTIME_IDENTITY_TEXT',identity_issues=issues,
        files=[file_id(receipt,raw),file_id(identity,iraw)],review_passed=result is not None,review_error=error,
        reviewer_sha256=sha(Path(__file__).with_name('review_product_mode.py').read_bytes()),qualification_granted=False,product_closed=False)
    output=[]
    if result:
        for scope in result['scopes']:
            for rank in range(actual['tp']):
                if scope.get('batch'):
                    b=scope['batch'];counts=b['unique_batch_calls_by_rank'][rank];details=dict(actual_batch_sizes=b['actual_eligible_batch_sizes'],counter_unit='UNIQUE_BATCH_CALLS')
                else:
                    rows=[e for e in scope['effects'] if e['rank']==rank];counts={k:sum(e['coverage'][k] for e in rows) for k in rows[0]['coverage']}
                    details=dict(consumers=sorted({c for e in rows for c in e['consumers']}),changed_elements=sum(e['changed_elements'] for e in rows),counter_unit='REAL_CALLS')
                    if d['mechanism'] in ('41.tail_column','26.fia_destination','20.buffer_substitute'):
                        native=[e for e in rows if e['coverage']['eligible']]
                        identities=[e.get('native_runtime') for e in native]
                        # Extraction preserves missing identity rather than claiming runtime proof.
                        details['native_library_sha256']=sorted({e.get('native_library_sha256') for e in native if e.get('native_library_sha256')})
                        details['native_runtime']=identities[0] if identities and all(x==identities[0] for x in identities) else None
                        details['native_runtime_complete']=bool(identities and all(x and x==identities[0] for x in identities))
                output.append(dict(common,scope=scope['scope'],rank=rank,counts=counts,details=details,canary_exact=scope['canary_exact'],all_resources_zero=scope['all_resources_zero']))
    else:output=[common]
    for page in output:
        page['extract_sha256']=sha(canonical(page))
        if len(canonical(page))+1>4096:raise ValueError('PRODUCT_EXTRACT_EXCEEDS_4096_NO_TRUNCATION')
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('--runtime-identity',required=True);a=p.parse_args()
    for row in pages(a.receipt,a.runtime_identity):sys.stdout.buffer.write(canonical(row)+b'\n')


if __name__=='__main__':main()
