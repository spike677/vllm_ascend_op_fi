"""Offline text review -> draft registration -> maintenance staging. Never grants live admission."""
import argparse,json,hashlib
from pathlib import Path
from demoncat.runtime.vllm_ascend.worker_extension.product_registration import digest,validate_record,validate_identity
from tools.extract_registration_record import canonical,sha

def prepare(pages,*,expected_ranks,owners,source,negative_controls,mechanism,source_manifest_sha256,approved_reviewer_sha256):
    tp=len(expected_ranks)
    for a in expected_ranks:validate_identity(a)
    if len(owners)!=2 or len(set(owners.values()))!=2:raise ValueError('REGISTRATION_OWNER_TABLE_REQUIRED')
    seen=set();coverage=[];receipt_ids=set();reviewers=set()
    for page in pages:
        body={k:v for k,v in page.items() if k!='extract_sha256'}
        if len(canonical(page))+1>4096 or sha(canonical(body))!=page.get('extract_sha256'):raise ValueError('EXTRACT_SHA_OR_SIZE_INVALID')
        if page.get('schema')!='dcat.product.extract.v1' or page.get('mechanism')!=mechanism or page.get('source_manifest_sha256')!=source_manifest_sha256:raise ValueError('EXTRACT_SOURCE_OR_MECHANISM_MISMATCH')
        if page.get('review_passed') is not True or page.get('review_error') or page.get('identity_issues'):raise ValueError('EXTRACT_REVIEW_NOT_PASSED')
        key=(str(page.get('scope')),page.get('rank'))
        if key in seen or key[0] not in owners or type(key[1]) is not int or not 0<=key[1]<tp:raise ValueError('EXTRACT_SCOPE_RANK_INVALID')
        seen.add(key);expected=expected_ranks[key[1]]
        for k,v in page['identity'].items():
            if k=='image_digest':continue
            if k in expected and v!=expected[k]:raise ValueError('EXTRACT_IDENTITY_MISMATCH:'+k)
        if page.get('canary_exact') is not True or page.get('all_resources_zero') is not True:raise ValueError('EXTRACT_RECOVERY_FAILED')
        details=page.get('details',{})
        if not details.get('consumers') and details.get('counter_unit')!='UNIQUE_BATCH_CALLS':raise ValueError('EXTRACT_CONSUMER_MISSING')
        files=page.get('files',[])
        if not files:raise ValueError('EXTRACT_RECEIPT_IDENTITY_MISSING')
        if type(files[0]) is not list or len(files[0]) != 3 or type(files[0][1]) is not int or files[0][1] <= 0 or not files[0][0]:
            raise ValueError('EXTRACT_RECEIPT_IDENTITY_MISSING')
        receipt_ids.add(files[0][2]);reviewers.add(page.get('reviewer_sha256'))
        coverage.append(dict(owner=owners[key[0]],rank=key[1],counts=page['counts']))
    if seen!={(s,r) for s in owners for r in range(tp)} or len(receipt_ids)!=1 or reviewers!={approved_reviewer_sha256}:raise ValueError('EXTRACT_COMPLETE_COHORT_REQUIRED')
    labels={'missing_permit','unknown_profile','unknown_mechanism'}
    for control in negative_controls.values():
        if (control.get('ranks') != list(range(tp)) or control.get('owners') != list(owners.values())
                or not isinstance(control.get('receipt_sha256'),str) or len(control['receipt_sha256']) != 64):
            raise ValueError('NEGATIVE_CONTROL_COHORT_REQUIRED')
    if set(negative_controls)!=labels or any(v.get('rejected') is not True or v.get('positive_control') is not True or v.get('resources_zero') is not True for v in negative_controls.values()):raise ValueError('NEGATIVE_AND_POSITIVE_CONTROLS_REQUIRED')
    evidence=dict(mode='product',level='YELLOW_TEXT_ATTESTED' if source=='YELLOW_DEDICATED_INSTANCE' else 'BLUE_ORIGINAL_REVIEWED',owners=list(owners.values()),review_passed=True,receipt_sha256=receipt_ids.pop(),reviewer_sha256=approved_reviewer_sha256,source_manifest_sha256=source_manifest_sha256,consumers=True,bytes_restored=True,resources_zero=True,canary_recovered=True,negative_controls=True,coverage=coverage,
        extracts_sha256=digest(pages),negative_controls_sha256=digest(negative_controls),originals_location='YELLOW_RETAINED' if source=='YELLOW_DEDICATED_INSTANCE' else 'BLUE_PRIVATE_RETAINED')
    record=dict(schema='dcat.product.registration.v1',source=source,mechanism=mechanism,expected_ranks=expected_ranks,evidence=evidence);record['record_sha256']=digest(record);validate_record(record);return record

def stage_maintenance(record,registry,*,approved_sha):
    validate_record(record)
    if approved_sha!=record['record_sha256']:raise ValueError('EXPLICIT_REVIEW_APPROVAL_SHA_REQUIRED')
    if registry.get('schema')!='dcat.product.registry.v1':raise ValueError('REGISTRY_SCHEMA_INVALID')
    rows=[validate_record(r) for r in registry['records']]
    if any(r['record_sha256']==approved_sha for r in rows):raise ValueError('REGISTRATION_ALREADY_PRESENT')
    return dict(schema=registry['schema'],records=rows+[record])

def main():
 p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',required=True);p.add_argument('--registry');p.add_argument('--approved-sha');a=p.parse_args()
 data=json.loads(Path(a.input).read_bytes())
 try:
  if a.registry:
   result=stage_maintenance(data,json.loads(Path(a.registry).read_bytes()),approved_sha=a.approved_sha)
  else:result=prepare(**data)
 except (ValueError,KeyError,TypeError) as e:
  print(json.dumps(dict(status='NEEDS_SUPPLEMENT_OR_REJECT',reason=str(e),registration_created=False)));raise SystemExit(2)
 Path(a.output).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(status='REVIEWED_DRAFT_ONLY',qualification_granted=False,record_sha256=result.get('record_sha256'))))
if __name__=='__main__':main()
