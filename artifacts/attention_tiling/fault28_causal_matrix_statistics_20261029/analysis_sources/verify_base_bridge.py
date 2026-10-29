"""Mid-run gate using only four stock and four private BASE calls. CPU/read-only.

This gate never fabricates missing factorial cells. Its PASS permits the later
cross-profile experiment but is not a causal-effect, msprobe, or model verdict.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import profile_evidence as pe
import query_work_order as oracle
import verify_causal_matrix as full
from causal_core import digest, require


def select_base_cells(matrix):
    require(matrix['schema']=='fault28.causal.matrix.v1' and matrix['phase']=='primary','Expected frozen primary case matrix')
    selected=[x for x in matrix['cases'] if x['profile']=='base']
    require(len(selected)==4 and {(x['batch_size'],x['repeat_index']) for x in selected}=={(b,r) for b in (1,8) for r in (0,1)},'Need four distinct BASE cells')
    return selected


def analyze(args):
    require(pe.sha(args.vendor_manifest)=='3d3ca8e0bc12003e19c6f3f4cfba21775e516385fe5807b10d59c6bb53cb1785','Unapproved causal vendor')
    require(pe.sha(args.pod_schema)=='82457b4440b60061e3a32345fb3b33adacf1a0f1b527f078f7d2557cb1af743a','Unapproved public POD schema')
    dependencies={str(args.vendor_manifest):pe.sha(args.vendor_manifest),str(args.pod_schema):pe.sha(args.pod_schema)}
    for relative,expected in {**oracle.SOURCE_FILES,'common/include/op_kernel/util.h':full.UTIL_SHA}.items():
        file=full.child(args.source_root,relative)
        require(pe.sha(file)==expected,'Matched kernel/source helper changed')
        dependencies[str(file)]=expected
    util_path=args.previous_analysis/'verify_stock_off.py';parser_path=args.previous_analysis/'extract_fa_log_profile.py'
    util=pe.module(util_path,'bridge_pod_util','71c63c54f3f32f90d82e3177385197b5830ce07117b766818945b4bfe5e25fe1')
    parser=pe.module(parser_path,'bridge_native_log','fae9ce5be05d002376680a808b006711931334b83d8efb289de44931118a3f44')
    dependencies.update({str(util_path):pe.sha(util_path),str(parser_path):pe.sha(parser_path)})
    schema=full.read(args.pod_schema);manifest,inputs=full.verify_input_source(args.input_manifest)
    matrix=full.read(args.cases_dir/'manifest.json')
    # The frozen primary manifest is allowed to contain the unexecuted cross
    # cells, but this gate reads only the four BASE case configurations/results.
    selected=select_base_cells(matrix)
    rows=[];profiles={};common=None;identities=set();evidence={};summaries=[]
    expected_kwargs={**manifest['native_kwargs'],'pse':None,'padding_mask':None,'prefix':None,'actual_seq_qlen':None,'actual_seq_kvlen':None,'sink':None}
    mask_sha=digest(b''.join(b'\0'*(q+1)+b'\1'*(2047-q) for q in range(2048)))
    for item in selected:
        cp=full.child(args.cases_dir,item['file']);case=full.read(cp)
        require(pe.sha(cp)==item['sha256'] and case['operator_variant']=='private_vendor' and case['phase']=='primary','BASE frozen config differs')
        folder=args.results_dir/case['case_id'];rp=folder/'result.json';result=full.read(rp)
        require(result['schema']=='fault28.causal.replay.v1' and result['status']=='CAPTURE_COMPLETED' and
                result['original_api_calls']==1 and result['model_forward'] is False and result['operator_variant']=='private_vendor','BASE original replay incomplete')
        require(result['source_case_sha256']==item['sha256'] and result['input_manifest_sha256']==pe.sha(args.input_manifest),'BASE source/input SHA differs')
        for field in ('case_id','profile','batch_size','repeat_index'):
            require(result[field]==case[field]==item[field],'BASE case identity differs')
        call=result['native_call'];identity=(call['pid'],call['scope_token'])
        require(identity not in identities and call['scope_token'] and call['phase']=='operator_replay' and call['status']=='returned' and
                call['api']=='torch_npu.npu_fusion_attention_v3' and call['enter_monotonic_ns']<call['exit_monotonic_ns'],'BASE native identity/phase invalid')
        identities.add(identity)
        collector={k:result['statistics_collector'].get(k) for k in ('variant','precision','manifest_sha256','patched_processor_sha256','original_processor_sha256')}
        current=(result['replay_source_sha256'],result['versions'],result['vendor']['manifest_sha256'],collector)
        require(common is None or current==common,'BASE executable/vendor/collector differs')
        common=current
        require(current[2]==case['approved_vendor_manifest_sha256']==pe.sha(args.vendor_manifest),'BASE vendor differs')
        for name in inputs:
            full.check_snapshot(result['inputs_before'][name],result['inputs_after'][name],inputs[name],case['batch_size'])
        require(result['native_kwargs']==expected_kwargs and result['mask_before']==result['mask_after'] and
                result['mask_before']['full_sha256']==mask_sha and result['mask_before']['shape']==[2048,2048] and
                result['mask_before']['dtype']=='torch.bool','BASE precision/scalar/mask changed')
        output=full.private_output(folder,result)
        values=np.memmap(output['target_private_path'],dtype='<f2',mode='r')
        require(bool(np.isfinite(values).all()),'Nonfinite BASE output')
        del values
        profile=pe.verified_profile(folder,args.results_dir/(case['case_id']+'.log'),result,schema,util,parser,oracle)
        event=profile['event'];typed=event['typed']
        require((typed['b'],typed['n1'],typed['n2'],typed['g'],typed['m'],typed['n_ratio'])==(case['batch_size'],16,8,2,128,8),'Actual BASE geometry differs')
        require(event['requested_profile']==event['resolved_profile']=='base' and event['eligible'] is True and
                event['m_override_applied'] is False and event['r_override_applied'] is False,'BASE control was not inert')
        require(event['eligibility_checks']==len(event['guard_reports']) and event['eligibility_checks']>0 and
                all(x['passed'] is True and x['failed_guards']==[] for x in event['guard_reports']),'BASE context guard failure')
        profiles[case['case_id']]=profile
        rows.append({'case_id':case['case_id'],'batch_size':case['batch_size'],'profile':'base','repeat_index':case['repeat_index'],**output})
        evidence[str(rp)]=pe.sha(rp)
        summaries.append({'case_id':case['case_id'],'batch_size':case['batch_size'],'repeat_index':case['repeat_index'],
            'complete_target_sha256':output['target_output_sha256'],'actual_raw_sha256':profile['raw_sha256'],
            'host_events_sha256':profile['host_events_sha256'],'native_log_sha256':profile['native']['source_sha256'],
            'legal_query_key_coverage':profile['trace']['each_query_legal_keys_covered_once'],
            'query_split_between_cores':profile['trace']['query_split_between_cores']})
    for batch in (1,8):
        group=[x for x in rows if x['batch_size']==batch]
        require(len({x['target_output_sha256'] for x in group})==1 and
                len({profiles[x['case_id']]['profile_fingerprint'] for x in group})==1,'BASE repeat output/raw unstable')
    bridge=full.stock_bridge_gate(args,inputs,manifest,rows,profiles,common,parser)
    return {'schema':'fault28.causal.base_bridge_gate.v1','status':'PASS_STOCK_AND_PRIVATE_BASE_BRIDGE',
            'scope':'Only stock4 and privateBASE4; cross-profile effects are not yet accepted.',
            'preregistration_sha256':pe.sha(Path(__file__).parent/'preregistered_contract.json'),
            'input_manifest_sha256':pe.sha(args.input_manifest),'base_cases':summaries,
            'stock_bridge':bridge,'dependencies':dependencies,'evidence_sha256':evidence,
            'cross_profiles_analyzed':False,'missing_factorial_cases_fabricated':False,'historical_defect_reproduced':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cases-dir','results-dir','stock-cases-dir','stock-results-dir','input-manifest','source-root','pod-schema','vendor-manifest','previous-analysis','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Output must be new')
    try:report=analyze(args)
    except Exception as exc:report={'status':'FAIL_STOCK_BASE_BRIDGE','error_type':type(exc).__name__,'error':str(exc),'cross_profiles_analyzed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status']}))
    return 0 if report['status']=='PASS_STOCK_AND_PRIVATE_BASE_BRIDGE' else 1


if __name__=='__main__':raise SystemExit(main())
