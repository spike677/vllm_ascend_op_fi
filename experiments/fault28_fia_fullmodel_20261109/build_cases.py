"""Preregister the complete model matrix; no numerical outcome is assumed."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/fault28-fia-fullmodel-20261109')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    requests=json.loads((ROOT/'private/requests.json').read_bytes())
    inventory=ROOT/'outputs/cpu_export_a2/inventory.json'
    assert json.loads(inventory.read_bytes())['status']=='PASS_CPU_FULLGRAPH_EXPORT_NOT_NPU_ACCEPTED'
    cases=[]
    for batch in (1,8):
        for arm,mode in [('baseline','causal_base512'),('fault_r1','batch_causal_m128'),
                         ('fault_r2','batch_causal_m128'),('restore','off')]:
            cases.append((f'model_b{batch}_{arm}',batch,mode,True))
    cases += [('model_b8_noalias_baseline',8,'causal_base512',False),
              ('model_b8_noalias_fault',8,'batch_causal_m128',False)]
    hashes={}
    for name,batch,mode,aliases in cases:
        case={'schema':'fault28.fia_fullmodel.case.v1','case_id':name,
            'operator_variant':'private_vendor','model_family':'qwen3_fullmodel','mode':mode,
            'batch':batch,'sequence_length':2048,'layout':'BNSD','q_heads':16,'kv_heads':8,
            'head_dim':128,'aicore_num':16,'vectorcore_num':32,'sparse_mode':2,'inner_precise':0,
            'repeats':2,'capture_aliases':aliases,'capture_msprobe':True,'device_scalar_observer':True,
            'ge_statistics':True,'deterministic_algorithms':True,'use_cache':False,
            'request_ids':requests['request_sets'][f'b{batch}'],
            'input_manifest':'/audit/private/requests.json',
            'input_manifest_sha256':sha(ROOT/'private/requests.json'),
            'model_manifest':'/audit/model_manifest.json','model_manifest_sha256':sha(ROOT/'model_manifest.json'),
            'cpu_inventory':'/audit/outputs/cpu_export_a2/inventory.json','cpu_inventory_sha256':sha(inventory),
            'statistics_variant':'task_local_cpu_fp64_high_v1',
            'approved_collector_manifest_file':'/audit/collector_fp64_a1/stage_manifest.json',
            'approved_collector_manifest_sha256':sha(ROOT/'collector_fp64_a1/stage_manifest.json'),
            'approved_vendor_manifest_file':'/audit/vendor_stage_a5/vendor_manifest.json',
            'approved_vendor_manifest_sha256':sha(ROOT/'vendor_stage_a5/vendor_manifest.json'),
            'approved_custom_opp_path':'/audit/vendor_stage_a5/vendors/fault28_fia_graph_transformer',
            'scope':'Full Qwen3 prefill through registered FIA attention, not default HF SDPA route or decode.'}
        path=ROOT/'setup'/(name+'.json');assert not path.exists()
        path.write_text(json.dumps(case,indent=2)+'\n');hashes[name]=sha(path)
    runtime={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.sh')}
    matrix={'schema':'fault28.fia_fullmodel.matrix.v1','case_ids':list(hashes),'case_sha256':hashes,
        'formal_alias_cases':8,'neutrality_cases':2,'model_forwards_per_case':3,
        'expected_fia_nodes_per_graph':28,'expected_formal_node_executions':672,
        'total_model_forwards':30,'total_node_executions':840,
        'request_ids':requests['request_sets'],'runtime_sha256':runtime,
        'acceptance':'Coverage, complete per-request Attention/logits, repeat/OFF and independently inferred first producer; no expected difference is fabricated.',
        'preregistered_before_npu_execution':True}
    (ROOT/'setup/matrix_v1.json').write_text(json.dumps(matrix,indent=2)+'\n')
    print(json.dumps({'status':'FROZEN_NOT_NPU_ACCEPTED','cases':len(cases),'matrix_sha256':sha(ROOT/'setup/matrix_v1.json')}))


if __name__=='__main__':main()
