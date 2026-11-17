"""Publish summaries/hashes only, after a real controller-completed case."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
controller=json.loads((a.case.parent/'controller.json').read_bytes())
case=next(x for x in controller['cases'] if x['id']==a.case.name)
if case.get('exit_code')!=0:raise RuntimeError('CASE_NOT_COMPLETED_SUCCESSFULLY')
result=json.loads((a.case/'result.json').read_bytes())
runtime=json.loads((a.case/'serving/runtime/serving.json').read_bytes())
if result['status']!='PASS_COMMON_FIA26_BFOFF' or not runtime['recovered'] or runtime['error']:
    raise RuntimeError('MODEL_RECOVERY_NOT_PROVEN')
summary=dict(schema='demoncat.p4.fia26.summary.v1',status=result['status'],case=str(a.case),
    actual_model_class=result['setup']['model_class'],model_type=result['setup']['model_type'],
    baseline_control=result['baseline_control'],deployment=result.get('deployment',{}),
    common_worker=True,installed_source_sha256=result['setup']['installed_source_sha256'],
    image_id=case['image_id'],raw_data_published=False,common_worker_all_eligible=False,
    core_methods_replaced=False,workers_created_by_extension=0,recovered=True,phases=[],
    hashes={name:sha(a.case/name) for name in ('result.json','serving/runtime/serving.json')},
    collector_hashes={name:sha(a.case/'serving/capture'/name) for name in ('config.json','collector_identity.json')})
ref=runtime['rows'][0]
for row,host in zip(runtime['rows'],result['phases']):
    if row['request_id']!=host['mapping']['internal_request_id'] or host['output']['request_id']!=host['mapping']['external_request_id']:
        raise RuntimeError('ACTUAL_REQUEST_MAPPING_MISMATCH')
    item=dict(phase=row['phase'],hidden_finite=row['hidden_finite'],logits_finite=row['logits_finite'],
        hidden_changed=row['hidden_sha256']!=ref['hidden_sha256'],logits_changed=row['logits_sha256']!=ref['logits_sha256'],
        input_sha256=row['input_sha256'],hidden_sha256=row['hidden_sha256'],logits_sha256=row['logits_sha256'],
        top1=row['top1'],generated_tokens=host['output']['tokens'],generated_text=host['output']['text'],
        actual_scheduler_driver_mapping=True,actual_request_id_sha256=hashlib.sha256(row['request_id'].encode()).hexdigest())
    native=row.get('native')
    if native:
        completed=native['completed'];coverage=native['coverage']
        item.update(native_claims=native['claims'],completed=len(completed),all_native_pass=all(x.get('pass') is True for x in completed),
            coverage={k:coverage[k] for k in ('candidates','all_observed_supported_calls_routed','all_discovered_candidates_covered','cleanup_completed')},
            pending=native['pending'])
        if row['phase']=='comparison' and not result['baseline_control']:
            item['writer_consumer_recovery']=dict(all_real_kernel_writes=all(x['real_kernel_wrote_victim'] for x in completed),
                all_consumers_matched=all(x['consumer_matched'] and x['consumer_executed'] for x in completed),
                all_original_fia_outputs_copied=all(x['output_copy_equal'] for x in completed),
                all_residuals_changed=all(x['victim_changed'] for x in completed),
                all_residuals_restored=all(x['victim_hash_before']==x['victim_hash_restored'] for x in completed))
    summary['phases'].append(item)
summary['cleanup']={k:result[k]['status'] for k in ('unregister','detach','unset')}
summary['limits']=['TP1, eager, single fresh prefill request per phase, CANN8.5.1 deployment only',
    'Native runtime proof is separate from the independent raw statistics comparison',
    'No arbitrary simultaneous owners, continuous generation or public all_eligible qualification']
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(dict(model=summary['actual_model_class'],phase_count=len(summary['phases']),status=summary['status'])))
