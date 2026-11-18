"""Read-only consistency review; never grants qualification or changes raw data."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(path.read_bytes())


def valid_prefix(row):
    """Check the recorded complete logical range, without reading tensor values."""
    try:
        before,after=row['input_before'],row['input_after']
        proof=row['valid_kv_range_proof'];meta=row['boundary']['observational_metadata']
        cache=row['boundary']['cache']
        shape=cache['shape'];blocks,page,heads,columns=shape
        lengths,pages=meta['sequence_lengths'],meta['used_pages']
        if len(lengths)!=1 or len(pages)!=1:return False
        length=lengths[0];width=heads*columns
        if not all(type(x) is int and x>0 for x in (*shape,length)):return False
        selected=pages[0]
        if (len(selected)!=(length+page-1)//page or len(set(selected))!=len(selected)
                or any(type(x) is not int or not 0<=x<blocks for x in selected)):return False
        expected=[dict(logical_token_start=i*page,physical_page=p,token_count=min(page,length-i*page),
            element_offset=p*page*width,elements=min(page,length-i*page)*width) for i,p in enumerate(selected)]
        if (proof['ranges']!=expected or proof['bytes_per_cache']!=length*width*2
                or not 0<proof['bytes_per_cache']<=1024*1024 or proof['read_only'] is not True
                or proof['selected_slot']!=row['boundary']['slot'] or before!=after
                or before['valid_tokens']!=length or before['elements_per_token']!=width):return False
        for name in ('key','value'):
            actual=proof['storage'][name];observed=meta[name]
            if any(actual[k]!=observed[k] for k in ('shape','stride','dtype','storage_identity','storage_offset')):return False
            # Native FIA receives a legal flattening of heads x head_dim; the
            # real cache allocation remains four-dimensional. Preserve both.
            if actual['shape'] not in (shape,[blocks,page,width]) or actual['dtype']!='torch.bfloat16':return False
            stride=[];elements=1
            for dim in reversed(actual['shape']):stride.insert(0,elements);elements*=dim
            if actual['stride']!=stride:return False
        if any(proof['storage']['value'][k]!=cache[k] for k in ('storage_identity','storage_offset')):return False
        return (all(isinstance(before.get(k),str) and re.fullmatch('[0-9a-f]{64}',before[k]) for k in
                    ('slot_sha256','query_sha256','active_key_sha256','active_value_sha256'))
                and all(type(before.get(k)) is int and before[k]==0 for k in
                    ('slot_nonfinite','query_nonfinite','active_key_nonfinite','active_value_nonfinite')))
    except (KeyError,TypeError,ValueError,IndexError):return False


def review_arm(folder, group_name='c2', *, require_valid_prefix=False):
    path=folder/'result.json'
    return review_document(read(path),group_name,require_valid_prefix=require_valid_prefix,result_sha256=sha(path))


def review_document(document, group_name='c2', *, require_valid_prefix=False,
                    result_sha256=None, native_owner=None):
    """Same saved-evidence checks, without requiring a new/rewritten result file.

    Public Core has a separate RPC owner; its caller must join native_owner to
    the actual deployment receipt. Existing standalone callers keep their owner.
    """
    if group_name not in ('c2','c3'):raise ValueError('EXPLICIT_C2_OR_C3_REQUIRED')
    group=document['groups'][group_name];c=group['coverage'];records=c['completed_requests']
    planned=set(c['discovery']['observed_target_ids']);n=len(planned)
    inventories=[row['inventory'] for row in c['independent_discovery']]+[row['inventory'] for row in records]
    module_sets=[set(map(int,frame['module_calls'])) for frame in inventories]
    exact_counts=all(frame['complete'] and len(frame['calls'])==n and len(frame['writers'])==n
        and len(frame['module_calls'])==n
        and all(count==1 for count in frame['module_calls'].values())
        and all(row['original_calls']==1 and row['original_returned'] for row in frame['calls'])
        and {row['module_object_id'] for row in frame['calls']}==module_sets[0]
        and {row['module_object_id'] for row in frame['writers']}==module_sets[0] for frame in inventories)
    executions=c['executions'];native=[row['native'] for row in executions]
    off={row['target_id']:row for row in records[-1]['byte_evidence']}
    baseline=records[0]['model_output_sha256'];health=c['health'];recovery=c['recovery']
    controls=document['arm']=='baseline'
    repeats=2 if group_name=='c3' else 1
    checks=dict(two_independent_healthy_inventories=len(c['independent_discovery'])==2,
        actual_trigger=c['trigger']==('every_match' if repeats==2 else 'once') and not (controls and repeats==2),
        complete_actual_writer_consumer_denominators=exact_counts and all(value==module_sets[0] for value in module_sets),
        all_targets_lossless=len(c['per_target_lossless'])==n and {row['target_id'] for row in c['per_target_lossless']}==planned,
        complete_native_execution_distribution=len(executions)==repeats*n
            and Counter(row['target_id'] for row in executions)==Counter({target:repeats for target in planned}),
        actual_native_owner_generation=all(str(row['owner'])==(c['owner'] if native_owner is None else str(native_owner))
            and row['generation']==c['generation'] for row in native),
        actual_native_effect_restore=all(row['original_calls']==1 and row['consumer_returned'] and row['restored']
            and row['released'] and row['readback_exact']
            and row['changed_elements']==(0 if controls else row['selected_elements']) for row in native),
        per_consumer_input_restore=all(row['input_before']==row['input_after'] for record in records for row in record['byte_evidence']),
        all_targets_off=set(off)==planned and all(row['input_before']==row['input_after'] for row in off.values()),
        per_target_recovery_proof_hashes=len(recovery['targets'])==n and all(
            row['proof_sha256']==digest(off[digest(row['lease']['target'])]) and row['off_output_sha256']==baseline
            for row in recovery['targets']),
        same_model_normal_health=health['model_output_sha256']==baseline
            and health['actual_model_object_id']==group['phases'][0]['provider']['model_instance'],
        actual_health_request_mapping=health['request']['request_ids']==[group['health']['mapping']['internal_request_id']],
        health_proof_hashes=all(row['proof_sha256']==digest(health) for row in recovery['workers']),
        recovery_resources_zero=recovery['complete'] and not recovery['held_leases'] and not recovery['pending_leases']
            and not recovery['pending_drains'] and not recovery['errors'] and recovery['inflight']==0
            and c['owned_hooks']==c['owned_api_patches']==c['live_native_leases']==0
            and all(all(row[key]==0 for key in ('active_leases','device_bytes','host_bytes','inflight_calls','owned_hooks'))
                    for row in recovery['workers']))
    phase_rows=[dict(phase=row['phase'],forward_id=row['inventory']['forward_id'],
        actual_request_ids=row['inventory']['request_ids'],root_sha256=row['model_output_sha256']) for row in records]
    decisions=Counter(row['decision']['reason'] for row in c['calls'])
    independent_calls={row['call_id']:row for record in records[1:] for row in record['inventory']['calls']}
    hits=[row for row in c['calls'] if row['receipt']['outcome']=='HIT']
    checks['joined_call_outcomes']=len(c['calls'])==len(independent_calls)==3*n and not c['missing_outcomes'] and all(
        row['call_id'] in independent_calls and row['request_ids']==independent_calls[row['call_id']]['request_ids']
        and row['forward_id']==independent_calls[row['call_id']]['forward_id']
        and row['receipt']['original_calls']==independent_calls[row['call_id']]['original_calls']==1
        for row in c['calls'])
    checks['observed_hit_distribution']=(not hits if controls else len(hits)==repeats*n
        and Counter(row['target_id'] for row in hits)==Counter({target:repeats for target in planned}))
    if require_valid_prefix:
        checks['every_complete_valid_kv_range_restored']=all(valid_prefix(row)
            for record in records for row in record['byte_evidence'])
        controls_to_compare=records if controls else [records[0],records[-1]] + (records[2:3] if repeats==1 else [])
        reference={row['target_id']:row for row in records[0]['byte_evidence']}
        checks['lossless_off_and_exhausted_full_prefix_exact']=all(
            row['input_before']==reference[row['target_id']]['input_before']
            and row['consumer_output_sha256']==reference[row['target_id']]['consumer_output_sha256']
            for record in controls_to_compare for row in record['byte_evidence'])
        provider=document['cleanup']['provider']
        checks['actual_final_provider_resources_zero']=(provider['installed'] is False
            and provider['owned_hooks']==provider['owned_api_patches']==provider['native_live_leases']==0
            and not provider['pending'])
    return document,dict(result_sha256=result_sha256,source_sha256=document['source_sha256'],
        native_library_sha256=document['setup']['library_sha256'],checks=checks,
        recomputed_consistency=all(checks.values()),planned_targets=n,independent_inventories=len(inventories),
        actual_eligible_fault_calls=2*n,actual_off_calls=n,recomputed_hits=len(hits),decision_counts=dict(decisions),
        phases=phase_rows,worker_identity=c['discovery']['expected_workers'],
        limitation=('Complete logical K/V range hashes and actual physical mapping reviewed; unused whole-cache fields remain in raw statistics.'
            if require_valid_prefix else 'Prior receipt hashes current V slot/query and FIA output, not every effective K/V prefix byte.'),
        public_qualification_granted=False)


def raw_changes(baseline,fault):
    changes=[]
    def walk(left,right,path):
        if isinstance(left,dict):
            if left.keys()!=right.keys():raise ValueError('RAW_STRUCTURE_MISMATCH')
            for key in left:
                if key in ('Max','Min','Mean','Norm'):
                    a,b=left[key],right[key]
                    if a is None or b is None:continue # retained by the existing full audit, never equality evidence
                    a,b=Decimal(str(int(a) if isinstance(a,bool) else a)),Decimal(str(int(b) if isinstance(b,bool) else b))
                    if a.is_finite() and b.is_finite() and abs(b-a)>Decimal('1e-8')+Decimal('1e-5')*abs(a):
                        changes.append(dict(identity=path,field=key,baseline=str(a),fault=str(b)))
                else:walk(left[key],right[key],path+'/'+str(key))
        elif isinstance(left,list):
            if len(left)!=len(right):raise ValueError('RAW_STRUCTURE_MISMATCH')
            for index,(a,b) in enumerate(zip(left,right)):walk(a,b,path+'/'+str(index))
    for key in baseline:walk(baseline[key],fault[key],key)
    return changes


def review(root):
    base=root/'postwrite-groups-card2-a1';audit=root/'postwrite41-groups-a1-raw-review'
    documents,summaries={},{}
    for arm in ('baseline','fault'):
        documents[arm],summaries[arm]=review_arm(base/('postwrite41-c2-'+arm))
    raw={arm:read(base/('postwrite41-c2-'+arm+'/capture/observation/step1/rank0/dump.json'))['data']
         for arm in ('baseline','fault')}
    changes=raw_changes(raw['baseline'],raw['fault'])
    stack=read(base/'postwrite41-c2-fault/capture/observation/step1/rank0/stack.json')
    full_write=[];cache=[];other=[]
    for change in changes:
        op=change['identity'].split('/')[0];b,f=raw['baseline'][op],raw['fault'][op]
        full=(op.startswith('Tensor.__setitem__.') and change['identity'].endswith('/input_args/0')
            and b['input_args'][0]['shape']==f['input_args'][0]['shape']==[1,16,128]
            and b['input_args'][1]==f['input_args'][1]==dict(type='slice',value=[None,1,None])
            and b['input_args'][2]==f['input_args'][2]
            and any(op in row[0] and 'output[:num_tokens] = attn_output[:num_tokens]' in '\n'.join(row[1])
                    for row in stack.values()))
        if full:full_write.append(change)
        elif change['identity'] in ('NPU._npu_reshape_and_cache.22.forward/input_kwargs/value_cache',
                                  'NPU.npu_fused_infer_attention_score.22.forward/input_kwargs/value'):cache.append(change)
        else:other.append(change)
    step1=[];layer22={}
    for arm,document in documents.items():
        phase=[r for r in document['groups']['c2']['coverage']['completed_requests'] if r['phase']=='group_fault']
        step1.append({row['target_key'][0]:row for row in phase[1]['byte_evidence']})
        layer22[arm]=[]
        for step,record in enumerate(phase):
            row=next(x for x in record['byte_evidence'] if x['target_key'][0]=='model.layers.22.self_attn.attn')
            metadata=row['boundary']['observational_metadata']
            dump=read(base/('postwrite41-c2-'+arm+'/capture/observation/step'+str(step)+'/rank0/dump.json'))['data']
            writer=dump['NPU._npu_reshape_and_cache.22.forward']['input_kwargs']
            fia=dump['NPU.npu_fused_infer_attention_score.22.forward']['input_kwargs']
            layer22[arm].append(dict(step=step,pages=metadata['used_pages'],lengths=metadata['sequence_lengths'],
                slot=row['boundary']['slot'],writer_value_max=writer['value']['Max'],
                pre_writer_whole_cache_max=writer['value_cache']['Max'],pre_consumer_whole_cache_max=fia['value']['Max']))
    audit_summaries={}
    for name in ('bf_step0','bf_step1','baseline_self','baseline_repeat'):
        path=audit/name/'full.json';value=read(path)
        file_checks={arm:{name:sha(Path(info['path']))==info['sha256'] for name,info in files.items()}
                     for arm,files in value['files'].items()}
        audit_summaries[name]={key:value.get(key) for key in ('aligned','rtol','atol','first_exceedance',
            'exceedance_count','e5_statistics_comparable','whole_dump_comparable')}
        audit_summaries[name].update(audit_sha256=sha(path),raw_file_hashes_rechecked=file_checks,
            inventory_counts={arm:row['counts'] for arm,row in value['full_dump_inventory'].items()})
    return dict(schema='demoncat.postwrite41.group_independent_consistency_review.v1',arms=summaries,
        audits=audit_summaries,step1=dict(all_retained_exceedances=len(changes),
            full_overwrite_destination_fields=len(full_write),whole_cache_fields=len(cache),unclassified_fields=other,
            first_exceedance=changes[0],whole_cache_changes=cache,layer22=layer22,
            same_full_fia_outputs=sum(step1[0][key]['consumer_output_sha256']==step1[1][key]['consumer_output_sha256'] for key in step1[0]),
            same_current_slot_and_query=sum(step1[0][key]['input_before']==step1[1][key]['input_before'] for key in step1[0]),
            interpretation='36 pre-overwrite output-buffer fields are proven full-overwrite destinations. The two whole-cache extrema are consistent with prior page10 history; old records do not prove every page11 prefix byte equal.',
            raw_fields_removed=0),c3=dict(accepted=False,
                recovery_errors=documents['fault']['failure_snapshot']['coverage']['recovery']['errors']),
        public_qualification_granted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/home/17-fault-c2c3-20261117'))
    print(json.dumps(review(parser.parse_args().root),indent=2,allow_nan=False))
