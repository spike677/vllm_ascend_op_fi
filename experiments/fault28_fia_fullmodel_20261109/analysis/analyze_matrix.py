"""Independent statistical first, then exact numerical/coverage acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import graph_evidence
from compare_model import compare
from coverage import audit

ROOT=Path('/home/fault28-fia-fullmodel-20261109')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def folder(name):return ROOT/'outputs'/(name+'_a5')/name


def statistical_pair(left,right,out,label):
    # Only raw statistics, config, collector identity and actual FX/GE graphs
    # enter this function. No case/result/host/coverage/numerics are read.
    observed=graph_evidence.compare(folder(left),folder(right))
    path=out/(label+'_msprobe.json');save(path,observed)
    return observed,sha(path)


def exact_pair(left,right,out,label,logits_only=False):
    report=compare(folder(left),folder(right),logits_only=logits_only)
    path=out/(label+'_complete.json');save(path,report)
    return report,sha(path)


def primary(out,fault):
    baseline='model_b8_baseline'
    label=fault.removeprefix('model_b8_')
    observed,stats_sha=statistical_pair(baseline,fault,out,label)
    exact,exact_sha=exact_pair(baseline,fault,out,label)
    nodes,_=graph_evidence.graph(folder(baseline))
    fia_names=[n['name'] for n in nodes if n['type']=='FusedInferAttentionScore']
    inferred=[]
    for step in observed['independent_first_producer']:
        assert step['status']=='ALIGNED_OBSERVED_FIRST_DIFFERENCE',step
        first=step['first']
        assert first['type']=='FusedInferAttentionScore' and first['boundary']=='output' and first['index']==0,first
        inferred.append(fia_names.index(first['node']))
    assert exact['first_changed_attention_inputs_equal'], 'Earliest changed Attention has unequal Q/K/V'
    assert all(layer==exact['first_changed_attention_layer'] for layer in inferred), 'Statistics misses the true earliest changed Attention'
    logits=next(r for r in exact['comparisons'] if r['kind']=='logits')
    first_attention=next(r for r in exact['comparisons'] if r['kind']=='attention' and r['layer']==inferred[0])
    return {'baseline':baseline,'fault':fault,'independent_first_layers':inferred,
        'first_changed_attention_layer':exact['first_changed_attention_layer'],
        'first_changed_last_query_attention_layer':exact['first_changed_last_query_attention_layer'],
        'first_attention_requests':first_attention['requests'],'logits_requests':logits['requests'],
        'logits_changed_requests':exact['logits_changed_requests'],'top1_changed_requests':exact['top1_changed_requests'],
        'first_attention_QKV_equal_all_requests':True,'statistics_report':label+'_msprobe.json',
        'statistics_report_sha256':stats_sha,'complete_report':label+'_complete.json','complete_report_sha256':exact_sha}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--primary-only',action='store_true');a=p.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=False)
    results=[primary(out,'model_b8_fault_r1')]
    if a.primary_only:
        save(out/'primary.json',{'status':'PRIMARY_ANALYZED_NOT_FINAL_ACCEPTANCE','result':results[0]})
        print(json.dumps({'status':'PRIMARY_ANALYZED_NOT_FINAL_ACCEPTANCE',**results[0]}));return
    results.append(primary(out,'model_b8_fault_r2'))
    controls=[]
    for left,right,label in [('model_b8_fault_r1','model_b8_fault_r2','fault_repeat'),
        ('model_b8_baseline','model_b8_restore','off_restore_b8'),
        ('model_b1_baseline','model_b1_fault_r1','b1_guard_r1'),
        ('model_b1_baseline','model_b1_fault_r2','b1_guard_r2'),
        ('model_b1_baseline','model_b1_restore','off_restore_b1')]:
        observed,stats_sha=statistical_pair(left,right,out,label)
        exact,exact_sha=exact_pair(left,right,out,label)
        assert exact['all_observed_elements_equal'],label
        assert all(row['status']=='NO_OBSERVED_FINITE_DIFFERENCE' for row in observed['independent_first_producer']),label
        controls.append({'name':label,'all_elements_equal':True,'statistics_sha256':stats_sha,'exact_sha256':exact_sha})
    neutrality=[]
    for left,right,label in [('model_b8_baseline','model_b8_noalias_baseline','baseline_alias_neutrality'),
                            ('model_b8_fault_r1','model_b8_noalias_fault','fault_alias_neutrality')]:
        exact,digest=exact_pair(left,right,out,label,logits_only=True)
        assert exact['all_observed_elements_equal'],label
        neutrality.append({'name':label,'all_request_logits_byte_equal':True,'report_sha256':digest})
    matrix=json.loads((ROOT/'setup/matrix_v5.json').read_bytes());coverage=[]
    for name in matrix['case_ids']:
        report=audit(name+'_a5');save(out/'coverage'/(name+'.json'),report)
        expected=128 if name in ('model_b8_fault_r1','model_b8_fault_r2','model_b8_noalias_fault') else 512
        assert report['actual_m']==expected and report['no_splitkv'],name
        coverage.append({k:report[k] for k in ('case_id','batch','logical_fia_nodes','actual_fia_node_executions',
            'layer_request_executions','native_selector_evaluations','shared_compiled_kernels','actual_m','result_sha256')})
    assert sum(r['actual_fia_node_executions'] for r in coverage)==840
    result={'schema':'fault28.fia_fullmodel.acceptance.v1',
        'status':'PASS_FULLMODEL_FIA_COVERAGE_FIRST_DIFFERENCE_AND_RESTORE',
        'model':'original Qwen3-0.6B checkpoint uniformly loaded FP16; registered real FIA attention',
        'scope':'Single TorchAir fullgraph of the complete 28-layer model, fixed prefill inputs, logits_to_keep=1, no decode.',
        'cases':10,'formal_alias_cases':8,'model_graph_executions':30,'fia_node_executions':840,
        'logical_compiled_fia_nodes':280,'layer_request_executions':sum(r['layer_request_executions'] for r in coverage),
        'formal_b8_fault_node_executions':168,'formal_b8_fault_layer_request_executions':1344,
        'all_supported_layers_requests_covered':True,'coverage':coverage,'primary_pairs':results,
        'controls':controls,'alias_neutrality':neutrality,'all_parameters_inputs_unchanged':True,
        'first_token_logits_propagation':all(r['logits_changed_requests']>0 for r in results),
        'all_eight_request_logits_changed':all(r['logits_changed_requests']==8 for r in results),
        'matrix_sha256':sha(ROOT/'setup/matrix_v5.json'),'analysis_source_sha256':sha(Path(__file__)),
        'historical_Pangu_version_reproduced':False,
        'limits':['HF default SDPA/FA route is explicitly adapted through its AttentionInterface to real FIA.',
                  'Fixed B1/B8, FP16/BNSD/2048/Q16/KV8/D128, CANN9.1/910B4/TP1; not arbitrary shapes or decode coverage.',
                  'Controlled current-version causal kernel strategy, not historical Pangu default strategy reproduction.',
                  'High precision statistics use a disclosed uniform CPU FP64 collector and actual graph output aliases.',
                  'Measured column granularity mechanism also affects BMM1/BMM2; not unique Reduce instruction attribution.',
                  'Full all-request tensors/raw DEBUG logs in task tmpfs are private and volatile; key target snapshots/logits retained under /home.']}
    save(out/'acceptance.json',result)
    print(json.dumps({k:result[k] for k in ('status','cases','model_graph_executions','fia_node_executions',
        'layer_request_executions','first_token_logits_propagation','all_eight_request_logits_changed')}))


if __name__=='__main__':main()
