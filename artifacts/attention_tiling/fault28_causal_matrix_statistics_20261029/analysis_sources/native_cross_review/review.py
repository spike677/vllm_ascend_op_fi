"""Read-only CPU cross-profile audit after controller completion. No framework imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_evidence as pe
import query_work_order as oracle
from causal_core import compare_orders, require

ROOT=Path('/home/fault28-causal-20261029')
MANIFEST_SHA='3d3ca8e0bc12003e19c6f3f4cfba21775e516385fe5807b10d59c6bb53cb1785'
SCHEMA_SHA='82457b4440b60061e3a32345fb3b33adacf1a0f1b527f078f7d2557cb1af743a'
UTIL_SHA='442648c354814d149ba7f8185e4b16cc7d528d1333f8437f07c0f3c5c93187d9'

def read(p):
    return json.loads(p.read_bytes())

def completed(attempt):
    c=read(attempt/'completion.json')
    require(c['status']=='CAPTURE_COMPLETED' and c['owned_container_removed'] is True and
            c['device_lock_released'] is True and c['cleanup_errors']==[], 'Incomplete capture/cleanup')
    for row in c['cases']:
        require(row['rc']==0 and row['status']=='CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE','Failed case')
    return c

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attempt',type=Path,required=True)
    ap.add_argument('--base-attempt',type=Path,required=True)
    ap.add_argument('--previous-analysis',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),'Output must be new')
    comp,basecomp=completed(a.attempt),completed(a.base_attempt)
    schema_path=ROOT/'build/evidence_a1/fa_pod_schema.json'
    vendor_path=ROOT/'vendor_stage_a1/vendor_manifest.json'
    source=ROOT/'source/ops-transformer-v9.1.0'
    require(pe.sha(schema_path)==SCHEMA_SHA and pe.sha(vendor_path)==MANIFEST_SHA,'Frozen schema/vendor changed')
    schema,vendor=read(schema_path),read(vendor_path)
    source_files={**oracle.SOURCE_FILES,'common/include/op_kernel/util.h':UTIL_SHA}
    for p,s in source_files.items():require(pe.sha(source/p)==s,'Kernel/source changed '+p)
    policy=read(ROOT/'build/evidence_a1/policy_source_manifest.json')
    for x in policy['files']:require(pe.sha(source/x['path'])==x['after_sha256'],'Policy source changed')
    for x in vendor['artifacts']:require(pe.sha(ROOT/vendor['vendor_root']/x['path'])==x['sha256'],'Vendor changed '+x['path'])
    parser=pe.module(a.previous_analysis/'extract_fa_log_profile.py','cross_native_parser','fae9ce5be05d002376680a808b006711931334b83d8efb289de44931118a3f44')
    util=pe.module(a.previous_analysis/'verify_stock_off.py','cross_pod_util','71c63c54f3f32f90d82e3177385197b5830ce07117b766818945b4bfe5e25fe1')
    profiles={};rows=[];expected=set()
    decoder=ROOT/'build/evidence_a1/fa_pod_decoder'
    with tempfile.TemporaryDirectory(prefix='cross_pod_',dir=a.output.parent) as temp:
        for attempt,c in [(a.base_attempt,basecomp),(a.attempt,comp)]:
            for item in c['cases']:
                name=item['case_id'];folder=attempt/name;result=read(folder/'result.json')
                require(pe.sha(folder/'result.json')==item['result_sha256'],'Completed result changed')
                require(result['status']=='CAPTURE_COMPLETED' and result['original_api_calls']==1,'Incomplete original native call')
                profile=pe.verified_profile(folder,attempt/(name+'.log'),result,schema,util,parser,oracle)
                event=profile['event'];p=event['typed'];pod=profile['pod']
                desired={'base':(128,8),'m_only':(64,8),'r_only':(128,4)}
                label=result['profile'];batch=result['batch_size'];repeat=result['repeat_index']
                require(label in desired and (p['m'],p['n_ratio'])==desired[label],'Actual factors differ')
                require(event['eligible'] is True and event['requested_profile']==event['resolved_profile']==label and
                        event['eligibility_checks']==len(event['guard_reports']) and event['eligibility_checks']>0 and
                        all(g['passed'] is True and g['failed_guards']==[] for g in event['guard_reports']),'Native guard failed')
                require(event['m_override_applied'] is (p['m']==64) and event['r_override_applied'] is (p['n_ratio']==4),'Control/raw inconsistency')
                require(p['b']==batch and p['n1']==16 and p['n2']==8 and p['g']==2 and p['s1']==p['s2']==2048 and
                        p['s2_base']==128 and p['d']==128 and p['sparse_mode']==2 and p['impl_mode']==0 and
                        p['calc_bytes']==4 and p['high_precision'] is True and p['enable_l1_reuse'] is True,'Native semantic drift')
                require(event['tiling_key']=='22619644976' and p['has_mask']==1 and p['has_pse']==p['has_dropout']==0,'Kernel or mask policy changed')
                raw=bytes.fromhex(event['raw_hex']);tmp=Path(temp)/(name+'.bin');tmp.write_bytes(raw)
                cpp=json.loads(subprocess.check_output([str(decoder),str(tmp)]))
                require(cpp['fields']==pod and len(pod)==205,'Public C++/Python POD discrepancy')
                key=(batch,label,repeat);require(key not in profiles,'Repeated identity');profiles[key]=profile
                if attempt==a.attempt:expected.add(key)
                rows.append({'case_id':name,'batch':batch,'factor_label':label,'repeat':repeat,'actual_m':p['m'],
                    'actual_ratio':p['n_ratio'],'actual_s2_base':p['s2_base'],'actual_s2_group':p['s2_group'],
                    'template':p['template'],'key':event['tiling_key'],'block_dim':event['block_dim'],
                    'schedule_mode':event['schedule_mode'],'workspace_sizes':event['workspace_sizes'],
                    'enclosing_api_workspace_bytes':profile['native']['api_workspace_bytes'],
                    'raw_sha256':profile['raw_sha256'],'callback_status':event['callback_status'],
                    'cpp_python_pod_fields_equal':205,'guard_checks':event['eligibility_checks'],
                    'core_num':p['core_num'],'query_tasks':p['total_query_tasks'],'bmm1':profile['bmm']['bmm1'],'bmm2':profile['bmm']['bmm2'],
                    'native_launch':profile['native'],'source_query_coverage':{k:profile['trace'][k] for k in
                        ('queries_checked','every_real_query_owned_once','each_query_legal_keys_covered_once','query_split_between_cores','full_task_trace_sha256')}})
    require(expected=={(b,p,r) for b in (1,8) for p in ('m_only','r_only') for r in (0,1)},'Cross matrix incomplete')
    contrasts={}
    for b in (1,8):
        for p in ('base','m_only','r_only'):
            require(profiles[b,p,0]['profile_fingerprint']==profiles[b,p,1]['profile_fingerprint'],'Repeated actual profile differs')
        for p in ('m_only','r_only'):
            x,z=profiles[b,'base',0],profiles[b,p,0]
            contrasts[f'B{b}.base_to_{p}']={**compare_orders(x['trace'],z['trace']),
                'bmm_changed_fields':{s:{k:[x['bmm'][s][k],z['bmm'][s][k]] for k in x['bmm'][s] if x['bmm'][s][k]!=z['bmm'][s][k]} for s in ('bmm1','bmm2')}}
    result={'schema':'fault28.causal.cross_native_review.v1','status':'PASS_ACTUAL_CROSS_NATIVE_TILING_AND_SOURCE_COVERAGE',
            'scope':'8 cross and4 completed base calls; no numerical-effect or micro-arithmetic conclusion.',
            'vendor_manifest_sha256':MANIFEST_SHA,'public_pod_schema_sha256':SCHEMA_SHA,'source_files':source_files,
            'kernel_source_unchanged':True,'same_binary_rehashed_artifacts':len(vendor['artifacts']),'calls':rows,
            'source_plan_comparisons':contrasts,'source_plan_is_hardware_trace':False,'numeric_effect_accepted':False,
            'msprobe_first_difference_accepted':False,'review_container_or_npu_started':False,
            'completion_sha256':{'cross':pe.sha(a.attempt/'completion.json'),'base':pe.sha(a.base_attempt/'completion.json')},
            'tool_sha256':pe.sha(Path(__file__))}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'calls':len(rows),'output':str(a.output),'sha256':pe.sha(a.output)}))

if __name__=='__main__':
    main()
