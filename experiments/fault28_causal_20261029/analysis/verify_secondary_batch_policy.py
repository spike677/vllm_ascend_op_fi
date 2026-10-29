"""Read-only 12-call, predeclared controlled Batch -> profile confirmation.

Requires a valid full primary analysis. It does not retroactively attribute
natural stock batch effects, infer a historical policy, or identify a unique
floating-point instruction. Every secondary output must reproduce the same
actual primary profile under the same batch and exact inputs.
"""
import argparse
import json
from pathlib import Path

import profile_evidence as pe
import query_work_order as oracle
import verify_causal_matrix as main_gate
from causal_core import PROFILES, digest, require

MODES={'batch_m':'m_only','batch_r':'r_only','batch_joint':'joint'}


def analyze(args):
    primary=main_gate.read(args.primary_report)
    require(primary['status']=='ANALYZED_FULL_SEQUENCE_FACTORIAL' and
            primary['stock_bridge_compatibility']['status']=='PASS_STOCK_REPLAY_AND_PRIVATE_BASE_COMPATIBILITY',
            'Primary matrix/stock bridge not complete and stable')
    require(primary['preregistration_sha256']==pe.sha(Path(__file__).parent/'preregistered_contract.json'),'Preregistration changed')
    manifest,inputs=main_gate.verify_input_source(args.input_manifest)
    require(primary['input_manifest_sha256']==pe.sha(args.input_manifest),'Secondary uses a different fixed input')
    for path,sha in primary['dependencies'].items():
        require(pe.sha(Path(path))==sha,'Primary audited dependency changed: '+path)
    schema=main_gate.read(args.pod_schema)
    require(pe.sha(args.pod_schema)=='82457b4440b60061e3a32345fb3b33adacf1a0f1b527f078f7d2557cb1af743a','POD schema differs')
    util=pe.module(args.previous_analysis/'verify_stock_off.py','secondary_util','71c63c54f3f32f90d82e3177385197b5830ce07117b766818945b4bfe5e25fe1')
    parser=pe.module(args.previous_analysis/'extract_fa_log_profile.py','secondary_log','fae9ce5be05d002376680a808b006711931334b83d8efb289de44931118a3f44')
    matrix=main_gate.read(args.cases_dir/'manifest.json')
    require(len(matrix['cases'])==12,'Secondary matrix must contain all 12 prespecified calls')
    approved={(x['batch_size'],x['profile'],x['repeat_index']):x for x in primary['calls']}
    frozen=primary['frozen_execution']
    expected_kwargs={**manifest['native_kwargs'],'pse':None,'padding_mask':None,'prefix':None,'actual_seq_qlen':None,'actual_seq_kvlen':None,'sink':None}
    mask_sha=digest(b''.join(b'\0'*(q+1)+b'\1'*(2047-q) for q in range(2048)))
    seen=set();identities=set();reports=[]
    for item in matrix['cases']:
        cp=main_gate.child(args.cases_dir,item['file'])
        require(pe.sha(cp)==item['sha256'],'Frozen secondary case changed')
        case=main_gate.read(cp);folder=args.results_dir/case['case_id'];rp=folder/'result.json';r=main_gate.read(rp)
        mode,b,repeat=case['profile'],case['batch_size'],case['repeat_index']
        key=(mode,b,repeat)
        require(mode in MODES and b in (1,8) and repeat in (0,1) and key not in seen,'Duplicate/unapproved secondary cell')
        seen.add(key)
        expected='base' if b==1 else MODES[mode]
        reference=approved[b,expected,repeat]
        require(r['schema']=='fault28.causal.replay.v1' and r['status']=='CAPTURE_COMPLETED' and r['original_api_calls']==1 and
                r['model_forward'] is False and r['source_case_sha256']==item['sha256'],'Secondary call did not complete correctly')
        for field in ('case_id','profile','batch_size','repeat_index'):
            require(r[field]==case[field]==item[field],'Secondary ledger differs: '+field)
        require(r['input_manifest_sha256']==primary['input_manifest_sha256'] and r['replay_source_sha256']==frozen['replay_source_sha256'] and
                r['versions']==frozen['versions'] and r['vendor']['manifest_sha256']==frozen['vendor_manifest_sha256'],'Secondary execution/input source differs')
        collector={k:r['statistics_collector'].get(k) for k in frozen['collector_build']}
        require(collector==frozen['collector_build'],'Secondary collector differs')
        for name in inputs:
            main_gate.check_snapshot(r['inputs_before'][name],r['inputs_after'][name],inputs[name],b)
        require(r['native_kwargs']==expected_kwargs and r['mask_before']==r['mask_after'] and r['mask_before']['full_sha256']==mask_sha,
                'Secondary native scalars/mask differ')
        call=r['native_call'];identity=(call['pid'],call['scope_token'])
        require(identity not in identities and call['phase']=='operator_replay' and call['status']=='returned','Repeated/invalid secondary call identity')
        identities.add(identity)
        out=main_gate.private_output(folder,r)
        profile=pe.verified_profile(folder,args.results_dir/(case['case_id']+'.log'),r,schema,util,parser,oracle)
        event=profile['event'];actual=(event['typed']['m'],event['typed']['n_ratio'])
        require(actual==PROFILES[expected] and event['typed']['b']==b,'Secondary actual profile is not the predeclared Batch policy')
        require(event['requested_profile']==mode and event['resolved_profile']==expected and event['eligible'] is True and
                event['m_override_applied'] is (actual[0]==64) and event['r_override_applied'] is (actual[1]==4),'Control witness differs from independently decoded profile')
        require(event['eligibility_checks']==len(event['guard_reports']) and event['eligibility_checks']>0 and
                all(x['passed'] is True and x['failed_guards']==[] for x in event['guard_reports']),'Secondary eligibility guard failed')
        require(profile['raw_sha256']==reference['raw_sha256'],'Secondary raw profile differs from same-batch primary profile')
        primary_result=args.primary_results_dir/reference['case_id']/'result.json'
        require(str(primary_result) in primary['evidence_sha256'] and pe.sha(primary_result)==primary['evidence_sha256'][str(primary_result)],
                'Primary comparison result source changed')
        primary_runtime=main_gate.read(primary_result)
        original=main_gate.private_output(primary_result.parent,primary_runtime)
        require(out['target_output_sha256']==original['target_output_sha256']==reference['target_output_sha256'] and
                Path(out['target_private_path']).read_bytes()==Path(original['target_private_path']).read_bytes(),
                'Secondary does not reproduce the same-batch primary output')
        require([x['name'] for x in r['auxiliary_outputs']]==['softmax_max','softmax_sum','softmax_out','seed','offset'],'Secondary auxiliary returns missing')
        auxiliary={a['name']:{'shape_equal':a['shape']==z['shape'],'dtype_equal':a['dtype']==z['dtype'],
            'full_return_bit_difference':a['full_sha256']!=z['full_sha256']} for a,z in zip(r['auxiliary_outputs'],primary_runtime['auxiliary_outputs'])}
        reports.append({'case_id':case['case_id'],'batch_size':b,'policy':mode,'repeat_index':repeat,
            'actual_profile':expected,'actual_m':actual[0],'actual_n_ratio':actual[1],
            'complete_target_sha256':out['target_output_sha256'],'same_batch_primary_byte_equal':True,
            'same_batch_primary_raw_equal':True,'result_sha256':pe.sha(rp),'host_events_sha256':profile['host_events_sha256'],
            'native_log_sha256':profile['native']['source_sha256'],'auxiliary_returns_vs_same_profile_primary':auxiliary,
            'auxiliary_change_does_not_imply_O_change':True})
    require(seen=={(m,b,r) for m in MODES for b in (1,8) for r in (0,1)},'Incomplete secondary matrix')
    return {'schema':'fault28.causal.secondary_analysis.v1','status':'PASS_CONTROLLED_BATCH_PROFILE_CHAIN',
            'primary_report_sha256':pe.sha(args.primary_report),'preregistration_sha256':primary['preregistration_sha256'],
            'matrix_manifest_sha256':pe.sha(args.cases_dir/'manifest.json'),'calls':reports,
            'effect_sizes':'Use the independently computed primary full-sequence factorial contrasts; no output-dependent profile selection.',
            'observed_main_output_difference_required':False,'historical_default_batch_policy_identified':False,
            'floating_point_instruction_cause':'NOT_IDENTIFIED','private_tensor_values_in_report':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('primary-report','primary-results-dir','cases-dir','results-dir','input-manifest','pod-schema','previous-analysis','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Output must be new')
    try:report=analyze(a)
    except Exception as exc:report={'status':'FAIL_SECONDARY_CONTRACT','error_type':type(exc).__name__,'error':str(exc),'causal_attribution':'NOT_IDENTIFIED'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status']}))
    return 0 if report['status']=='PASS_CONTROLLED_BATCH_PROFILE_CHAIN' else 1


if __name__=='__main__':raise SystemExit(main())
