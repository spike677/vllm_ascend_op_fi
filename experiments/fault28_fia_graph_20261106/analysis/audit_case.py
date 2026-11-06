"""CPU audit of an already finished graph case, independent of localization."""
import argparse
import hashlib
import json
from pathlib import Path
import native_plan
import public_plan

ROOT=Path('/home/fault28-fia-graph-20261106')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok,text):
    if not ok:raise ValueError(text)


def compiler_binding(folder,native,host):
    inventory=json.loads((folder/'jit_compile_inventory.json').read_bytes())
    docs={};objects=[]
    for row in inventory['files']:
        if row['file'].endswith('.o'):objects.append(row)
        if 'private_copy' in row:
            path=folder/row['private_copy']
            need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Saved compiler metadata differs')
            docs[path.name]=(json.loads(path.read_bytes()),row['sha256'])
    isolated=[v for k,v in docs.items() if '_isolate_tiling_' in k]
    binaries=[v for k,v in docs.items() if 'te_fusedinferattentionscore_' in k]
    need(len(isolated)==len(binaries)==len(objects)==1,'Incomplete/ambiguous actual compiler artifacts')
    tiling,ts=isolated[0];binary,bs=binaries[0]
    h=json.loads((folder/'host_events.jsonl').read_text().splitlines()[0])
    actual=tiling['run_info'];final=tiling['tiling_info']
    need(str(actual['tiling_key'])==str(final['tiling_key'])==h['tiling_key']==native['selected']['jit_template_key'],
         'Compiler actual template key differs from host/native selection')
    need(actual['tiling_data']==final['tiling_data']==final['raw_run_info']['tiling_data']==h['raw_hex'],
         'Compiler did not receive the identical final public raw tiling')
    need(final['tiling_data_size']==1576 and final['block_num']==h['block_dim']==native['static_task_launch']['block_dim']
         and final['schedule_mode']==h['schedule_mode']==native['static_task_launch']['schedule_mode'],
         'Compiler/raw/actual launch parameters differ')
    need(tiling['is_static_shape'] is True and final['tiling_info_completed'] is True,'Incomplete static tiling')
    need(binary['binFileName']==binary['kernelName']==native['binary']['kernel_name'] and
         binary['sha256']==objects[0]['sha256'],'Actual GE task not bound to compiled binary SHA')
    return {'actual_template_key':str(actual['tiling_key']),'compiler_public_raw_sha256':host[0]['raw_sha256'],
            'compiler_raw_identical_to_final_public_host':True,'isolated_tiling_metadata_sha256':ts,
            'kernel_metadata_sha256':bs,'binary_sha256':objects[0]['sha256'],
            'generated_device_tiling_header_sha256':hashlib.sha256(final['file_content'].encode()).hexdigest(),
            'kernel_name':binary['kernelName'],'block_dim':final['block_num'],'schedule_mode':final['schedule_mode'],
            'runtime_static_key_distinct_from_jit_key':0,'kernel_argument_raw_observed':False}


def audit(attempt):
    root=ROOT/'outputs'/attempt
    completion=json.loads((root/'completion.json').read_bytes())
    need(completion['status']=='CAPTURE_COMPLETED' and completion['card']==5 and
         completion['device_lock_acquired'] and completion['device_lock_released'] and
         completion['owned_container_removed'] and not completion['cleanup_errors'] and
         not completion['external_container_mutations'],'Unclean/incomplete controlled execution')
    identity=json.loads((root/'container_identity.json').read_bytes())
    need(identity['image']=='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9',
         'Formal FIA graph was not executed in the pinned CANN9.1 image')
    need(len(completion['cases'])==1,'Audit expects one registered case')
    case_id=completion['cases'][0]['case_id'];folder=root/case_id
    result=json.loads((folder/'result.json').read_bytes());case=json.loads((folder/'case.json').read_bytes())
    need(result.get('torch_version')=='2.10.0+cpu' and result.get('torch_npu_version')=='2.10.0.post4',
         'Formal framework version differs from the validated ABI/runtime')
    need(sha(folder/'result.json')==completion['cases'][0]['result_sha256'],'Completion/result binding mismatch')
    is64=case.get('statistics_variant')=='task_local_cpu_fp64_high_v1'
    matrix=json.loads((ROOT/('setup/fp64_matrix_v1.json' if is64 else 'setup/formal_matrix_v3.json')).read_bytes())
    need(sha(folder/'case.json')==matrix['case_sha256'][case_id],'Not the final preregistered case')
    sources=json.loads((root/'source_manifest.json').read_bytes())
    runtime_path=ROOT/('setup/fp64_runtime_v1.json' if is64 else 'setup/formal_runtime_v3.json')
    runtime=json.loads(runtime_path.read_bytes())
    for name,expected in runtime['files'].items():need(sources[name]==expected,'Different frozen runtime: '+name)
    for name,expected in sources.items():need(sha(root/'source'/name)==expected,'Changed frozen source: '+name)
    need(result['status']=='CAPTURE_COMPLETED' and len(result['backend_graphs'])==1 and
         len(result['backend_graphs'][0]['fia_nodes'])==1 and
         [c['call_id'] for c in result['calls']]==[0,1,2] and all(c['inputs_unchanged'] for c in result['calls']),
         'Missing full graph/replay/input checks')
    need(result['repeated_all_outputs_equal'] and result['repeated_target_bytes_equal'],'Unstable repeated FIA output')
    need(len({c['consumer']['full_sha256'] for c in result['calls']})==1,'Unstable consumer output')
    native=native_plan.parse(root/(case_id+'.log'),folder)
    host=None;compiler=None
    if case['operator_variant']=='private_vendor':
        launch=json.loads((root/(case_id+'.vendor_launch.json')).read_bytes())
        need(launch['case_sha256']==sha(folder/'case.json') and launch['probe_sha256']==sources['graph_probe.py'],
             'Wrong fresh-process launcher binding')
        manifest_path=ROOT/Path(case['approved_vendor_manifest_file']).relative_to('/audit')
        need(sha(manifest_path)==case['approved_vendor_manifest_sha256'],'Wrong approved vendor')
        vendor=json.loads(manifest_path.read_bytes());abi=vendor['public_abi']
        schema_file=ROOT/abi['schema_file']
        need(sha(schema_file)==abi['schema_sha256'] and abi['generation']=='a4','Unbound ABI generation')
        host=public_plan.verify(folder,json.loads(schema_file.read_bytes()))
        compiler=compiler_binding(folder,native,host)
    return {'schema':'fault28.fia_graph.completed_case_audit.v1','status':'PASS_RECORDED_EXECUTION',
            'case_id':case_id,'attempt':attempt,'batch':case['batch'],'mode':case['mode'],
            'completion_sha256':sha(root/'completion.json'),'result_sha256':sha(folder/'result.json'),
            'preregistered_case_sha256':sha(folder/'case.json'),'runtime_freeze_sha256':sha(runtime_path),
            'native':native,'public_host':host,'compiler_binding':compiler,
            'numeric_mechanism_accepted':False,'scope':'Execution, full graph, repeated output, actual compiler key/raw and public serializer checks only; statistics localization and cross-case numerical causality are separate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:report=audit(a.attempt);rc=0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as e:
        report={'schema':'fault28.fia_graph.completed_case_audit.v1','status':'FAIL_AUDIT','error':str(e),'attempt':a.attempt};rc=1
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'error':report.get('error'),'attempt':a.attempt}))
    return rc


if __name__=='__main__':raise SystemExit(main())
