"""Public bytes/ABI/census and physical statistics-only blind inference."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

FORBIDDEN={'.pt','.pth','.bin','.npy','.npz','.safetensors','.o','.so','.run','.log','.gz','.pyc'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def extended(path):
    path=path.absolute()
    return Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--blind-work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--primary-bf-only',action='store_true',help='Require exactly the two selected B8 FIA groups.')
    a=p.parse_args();root=extended(a.bundle).resolve(strict=True);work=extended(a.blind_work)
    assert not work.exists() and not work.resolve().is_relative_to(root);work.mkdir(parents=True)
    sums={}
    for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);assert name not in sums and '..' not in Path(name).parts;sums[name]=digest
    files={f.relative_to(root).as_posix():f for f in root.rglob('*') if f.is_file()}
    assert set(files)==set(sums)|{'SHA256SUMS'}
    for name,file in files.items():
        assert not file.is_symlink() and file.resolve().is_relative_to(root)
        assert file.suffix.lower() not in FORBIDDEN
        assert not any(n in ('private_output','private_device_debug','inputs','__pycache__') for n in file.relative_to(root).parts)
        if name!='SHA256SUMS':assert sha(file)==sums[name],name
    mapping=json.loads((root/'source_mapping.json').read_bytes())
    by_source={}
    for r in mapping['files']:
        file=files[r['published']]
        assert file.stat().st_size==r['bytes'] and sha(file)==r['sha256'] and r['original_bytes_preserved']
        assert r['source'] not in by_source;by_source[r['source']]=r
    acceptance=json.loads((root/'reports/acceptance.json').read_bytes())
    if a.primary_bf_only:
        assert acceptance['schema']=='fault28.fia_fullmodel.primary_bf_acceptance.v1'
        assert acceptance['status']=='PASS_PRIMARY_BF_FIA_COVERAGE_FIRST_DIFFERENCE_AND_LOGITS'
        expected_cases,expected_executions,expected_participations=2,168,1344
        scope=json.loads((root/'publication_scope.json').read_bytes())
        assert scope['case_ids']==['model_b8_baseline','model_b8_fault_r1'] and scope['native_calls']==0
        assert scope['all_retained_capture_bytes_unchanged']
        pair=acceptance['primary_pairs'];assert len(pair)==1
        assert pair[0]['baseline']=='model_b8_baseline' and pair[0]['fault']=='model_b8_fault_r1'
        for field in ('statistics','complete'):
            assert sha(root/'reports'/pair[0][field+'_report'])==pair[0][field+'_report_sha256']
    else:
        assert acceptance['status']=='PASS_FULLMODEL_FIA_COVERAGE_FIRST_DIFFERENCE_AND_RESTORE'
        expected_cases,expected_executions,expected_participations=10,840,4368
    assert acceptance['cases']==expected_cases and acceptance['fia_node_executions']==expected_executions
    assert acceptance['first_token_logits_propagation'] and acceptance['all_eight_request_logits_changed']
    sys.path.insert(0,str(root/'analysis'))
    import graph_evidence,public_plan,coverage
    schema=json.loads((root/'build/fia_abi_schema.json').read_bytes())
    cases=sorted((root/'cases').iterdir());assert len(cases)==expected_cases
    if a.primary_bf_only:
        assert {c.name for c in cases}=={'model_b8_baseline','model_b8_fault_r1'}
        expected_reports={'acceptance.json','fault_r1_msprobe.json','fault_r1_complete.json',
                          'coverage/model_b8_baseline.json','coverage/model_b8_fault_r1.json'}
        assert {f.relative_to(root/'reports').as_posix() for f in (root/'reports').rglob('*') if f.is_file()}==expected_reports
    executions=0;participations=0;csv_count=0
    for case_folder in cases:
        name=case_folder.name
        result=json.loads((case_folder/'result.json').read_bytes())
        case=json.loads((case_folder/'case.json').read_bytes())
        if a.primary_bf_only:
            assert case['batch']==8 and case['capture_aliases'] and case['capture_msprobe']
            assert case['mode']==('causal_base512' if name=='model_b8_baseline' else 'batch_causal_m128')
            selection=json.loads((case_folder/'public_ge_selection.json').read_bytes())
            assert selection['selection']=='all graph nodes'
            assert selection['available_actual_graph_csv']==selection['published_csv']
        captured=json.loads((case_folder/'controller/completion.json').read_bytes())
        assert captured['status']=='CAPTURE_COMPLETED' and captured['owned_container_removed']
        assert captured['device_lock_acquired'] and captured['device_lock_released'] and not captured['cleanup_errors']
        assert not captured['external_container_mutations'] and captured['cases'][0]['result_sha256']==sha(case_folder/'result.json')
        assert result['status']=='CAPTURE_COMPLETED' and result['actual_backend_calls']==3 and result['single_graph_backend_replay']
        assert len(result['backend_graphs'])==1 and result['repeated_all_outputs_equal'] and result['parameters_unchanged']
        for step in (0,1):
            # Validates original stack/construct, synchronous unfiltered high
            # statistics, null preservation and no tensor payload/checksums.
            graph_evidence.load_step(case_folder/f'observation/step{step}/rank0/dump.json')
        fx=json.loads((case_folder/'fx_graph_0.json').read_bytes())
        fian=[n for n in fx['nodes'] if n['target']=='npu.npu_fused_infer_attention_score.default']
        assert len(fian)==28 and [n['layer'] for n in fian]==list(range(28))
        nodes,_=graph_evidence.graph(case_folder);ge_fia=[n for n in nodes if n['type']=='FusedInferAttentionScore']
        names={n['name'] for n in ge_fia};assert len(names)==28
        provenance=json.loads((case_folder/'native_provenance.json').read_bytes())
        raw=(case_folder/'native_lines.txt').read_bytes();parts=raw.splitlines(keepends=True)
        assert sha(case_folder/'native_lines.txt')==provenance['extract_sha256']
        assert len(parts)==len(provenance['original_lines'])
        for part,ref in zip(parts,provenance['original_lines']):
            assert len(part)==ref['bytes'] and hashlib.sha256(part).hexdigest()==ref['sha256']
        native=coverage.parse_native(case_folder/'native_lines.txt')
        coverage.verify_census(native,names,result['owner_pid'])
        public=public_plan.verify(case_folder,schema)
        actual={r['actual_host_key'] for r in public};assert len(actual)==1
        assert all(r['decoded']['baseParams.bSize']==case['batch'] for r in public)
        expected=128 if name in ('model_b8_fault_r1','model_b8_fault_r2','model_b8_noalias_fault') else 512
        assert all(r['decoded']['innerSplitParams.mBaseSize']==expected for r in public)
        compiler=json.loads((case_folder/'compiler_tiling_excerpt.json').read_bytes())
        events=[json.loads(line) for line in (case_folder/'host_events.jsonl').read_text(encoding='utf-8').splitlines()]
        assert {r['raw_hex'] for r in events}=={compiler['run_info']['tiling_data']}
        assert str(compiler['run_info']['tiling_key'])==next(iter(actual))
        meta=json.loads((case_folder/'compiler_binary_metadata.json').read_bytes())
        inventory=json.loads((case_folder/'jit_compile_inventory.json').read_bytes())
        binary={Path(r['file']).stem:r['sha256'] for r in inventory['files'] if r['file'].endswith('.o')}
        assert meta['sha256']==binary[meta['kernelName']]
        assert {n['kernel'] for n in native['binaries']}=={meta['kernelName']}
        report=json.loads((root/'reports/coverage'/(name+'.json')).read_bytes())
        assert report['status']=='PASS_ALL_MATCHED_LAYERS_REQUESTS_ACTUAL_GRAPH'
        assert report['raw_log_sha256']==provenance['source_log_sha256']
        for row in report['all_layer_request_iterations']:
            source='/home/fault28-fia-fullmodel-20261109/outputs/'+name+'_a5/'+name+'/'+row['csv']
            entry=by_source[source];assert entry['sha256']==row['sha256']
            csv_file=root/entry['published'];table=graph_evidence.csv_rows(csv_file)
            assert table[('Output',0)]['Shape']==f'{case["batch"]}x16x2048x128'
            assert row['request_slots']==list(range(case['batch']));csv_count+=1
        assert report['actual_fia_node_executions']==84
        executions+=report['actual_fia_node_executions'];participations+=report['layer_request_executions']
    assert executions==csv_count==expected_executions and participations==expected_participations
    roles=json.loads((root/'primary_BF.json').read_bytes());blind={}
    if a.primary_bf_only:
        assert roles['delivery']=='primary_bf_only' and 'fault_repeat' not in roles and 'off_restore' not in roles
    selected_roles=('baseline','fault') if a.primary_bf_only else ('baseline','fault','fault_repeat')
    for role in selected_roles:
        source=root/roles[role];target=work/source.name;target.mkdir()
        for name in ('config.json','collector_provenance.json','fx_graph_0.json'):shutil.copyfile(source/name,target/name)
        shutil.copytree(source/'observation',target/'observation')
        shutil.copytree(source/'msprobe_graph',target/'msprobe_graph')
        blind[role]=target
    reviews=[]
    for role in selected_roles[1:]:
        report=graph_evidence.compare(blind['baseline'],blind[role])
        first=report['independent_first_producer']
        assert all(r['status']=='ALIGNED_OBSERVED_FIRST_DIFFERENCE' and r['first']['type']=='FusedInferAttentionScore'
            and r['first']['boundary']=='output' and r['first']['index']==0 for r in first)
        target=work/(role+'_blind_first.json')
        with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
        reviews.append({'role':role,'report_file':str(target),'sha256':sha(target),'independent_first_producer':first})
    result={'schema':'fault28.fia_fullmodel.public_evidence_review.v1',
        'status':'PASS_PUBLIC_BYTES_ALL_NODE_REQUEST_CENSUS_ABI_AND_BLIND_FIRST',
        'files':len(files),'bytes':sum(f.stat().st_size for f in files.values()),
        'mapped_originals':len(mapping['files']),'source_mapping_sha256':sha(root/'source_mapping.json'),
        'cases':expected_cases,'statistics_triplets':2*expected_cases,'actual_fia_node_executions':executions,
        'layer_request_executions':participations,'bound_actual_fia_csv_files':csv_count,
        'physical_blind_reviews':reviews,'blind_inputs_exclude_control_and_numeric_evidence':True,
        'private_tensor_files_opened':0,'native_calls':0,'analysis_source_sha256':sha(Path(__file__)),
        'limits':['Public data cannot reread private complete tensors, raw logs or binary machine code.',
            ('Exactly two FIA groups include all native graph CSV; controls remain in the parent commit.' if a.primary_bf_only else
             'Primary B/F/F2 include all native graph CSV. Other controls retain every FIA CSV plus original statistics/graphs; full private comparisons are reports.'),
            'Four-statistic equality does not establish element equality, and observed first producer is not a unique physical root cause.']}
    output=extended(a.output);assert not output.exists() and not output.resolve().is_relative_to(root)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','files','bytes','mapped_originals','actual_fia_node_executions','layer_request_executions')}))


if __name__=='__main__':main()
