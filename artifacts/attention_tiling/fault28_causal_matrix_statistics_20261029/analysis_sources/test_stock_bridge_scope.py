"""Targeted CPU-only regression for experiment-stage/native-phase separation."""
import copy
import json
from pathlib import Path
import tempfile

from verify_causal_matrix import validate_stock_scope


def main():
    root=Path(__file__).resolve().parents[3]
    case={'phase':'stock_bridge','stock_bridge':True,'operator_variant':'stock'}
    runtime={'operator_variant':'stock','vendor':{'custom_opp_path':None},'native_call':{
        'phase':'operator_replay','scope_token':None,'status':'returned','api':'torch_npu.npu_fusion_attention_v3'}}
    checks=[]
    with tempfile.TemporaryDirectory(dir=root,prefix='_f28scope_') as name:
        folder=Path(name).resolve()
        assert folder.parent==root and folder.is_relative_to(root)
        validate_stock_scope(case,runtime,folder)
        checks.append({'name':'case_stock_bridge_native_operator_replay_accepted','status':'PASS'})
        def reject(label,c,r):
            try:validate_stock_scope(c,r,folder)
            except ValueError:checks.append({'name':label,'status':'PASS_REJECTED'});return
            raise AssertionError('Unexpected acceptance: '+label)
        wrong=copy.deepcopy(runtime);wrong['native_call']['scope_token']='injected_token'
        reject('non_null_scope_token',case,wrong)
        wrong=copy.deepcopy(runtime);wrong['vendor']['custom_opp_path']='/audit/private'
        reject('custom_vendor',case,wrong)
        wrong=copy.deepcopy(runtime);wrong['native_call']['phase']='stock_bridge'
        reject('case_phase_misused_as_api_phase',case,wrong)
        wrong=copy.deepcopy(case);wrong['phase']='primary'
        reject('non_stock_case_stage',wrong,runtime)
        (folder/'host_events.jsonl').write_text('{}\n',encoding='utf-8')
        reject('host_callback_evidence_forbidden',case,runtime)
    report={'status':'PASS_TARGETED_CPU_SCOPE_REGRESSION','npu_api_calls':0,'checks':checks,
            'does_not_confirm_native_execution':True}
    (Path(__file__).parent/'stock_scope_cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':len(checks)}))


if __name__=='__main__':main()
