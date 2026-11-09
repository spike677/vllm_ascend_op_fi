"""All logical FIA nodes, actual compiled binary/tasks and request coverage.

One shape-identical JIT result can serve 28 GE nodes. A host callback is not a
runtime call counter. Every logical node needs its own binding and 3 CSVs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from graph_evidence import graph,csv_rows,alias_mapping,csv_inventory
from public_plan import verify
from extract_native import extract

ROOT=Path('/home/fault28-fia-fullmodel-20261109')
SELECT=re.compile(r'OP\((\d+),.*?\[GenTilingKey\]\[(\d+)\] OpName:\[([^\]]+)\] FIA tilingKey_: (\d+)\.')
BINARY=re.compile(r'FillTaskDefAfterGenTask:Set kernel_name\[([^\]]+)\] for the kernel_def of node\[([^,]+), FusedInferAttentionScore\]')
LAUNCH=re.compile(r'StreamLaunchKernelV2: kernel info : .*?stream_id=(\d+), task_id=(\d+),.*?kernel_name=((?:te_fusedinferattentionscore_[0-9a-f]+)|(?:FusedInferAttentionScore_[0-9a-f]+_[0-9]+)),.*?coreDim=(\d+),.*?schemMode=(\d+)')
TASK=re.compile(r'Distribute:KernelTaskInfo Distribute Success, op: ([^,]+), taskid: (\d+), blockdim: (\d+), streamid: (\d+)')


def need(ok,text):
    if not ok:raise ValueError(text)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_native(path):
    selected=[];binaries=[];launches=[];tasks=[];windows=[];active=None
    for number,line in enumerate(path.read_text().splitlines(),1):
        if line.startswith('{'):
            e=json.loads(line)
            if e.get('schema')=='fault28.fia_graph.call_marker.v1':
                if e['event']=='enter':
                    need(active is None,'Nested graph call');active=e
                else:
                    need(active is not None and all(e[k]==active[k] for k in ('call_id','pid','batch','layout')),'Mismatched graph call')
                    windows.append(active);active=None
        s=SELECT.search(line);b=BINARY.search(line);l=LAUNCH.search(line);t=TASK.search(line)
        if s:selected.append({'pid':int(s[1]),'tid':int(s[2]),'node':s[3],'key':s[4]})
        if b:binaries.append({'kernel':b[1],'node':b[2]})
        if l:
            pid=re.search(r'RUNTIME\((\d+),',line)
            need(pid is not None,'Runtime owner absent')
            launches.append({'pid':int(pid[1]),'stream':int(l[1]),'task':int(l[2]),'kernel':l[3],
                'block_dim':int(l[4]),'schedule':int(l[5]),'call_id':active['call_id'] if active else None})
        if t and 'FusedInferAttentionScore' in t[1]:
            tasks.append({'node':t[1],'task':int(t[2]),'block_dim':int(t[3]),'stream':int(t[4])})
    need(active is None and [w['call_id'] for w in windows]==[0,1,2],'Not 3 completed graph executions')
    return {'selected':selected,'binaries':binaries,'launches':launches,'tasks':tasks,'windows':windows}


def verify_census(native,names,owner):
    need(len({w['pid'] for w in native['windows']})==1 and native['windows'][0]['pid']==owner,
        'Foreign graph owner')
    bound,tasks,launch=native['binaries'],native['tasks'],native['launches']
    need(Counter(n['node'] for n in bound)==Counter(names),'Missing/duplicate logical kernel binding')
    need(Counter(n['node'] for n in tasks)==Counter(names),'Missing/duplicate logical static task')
    need(Counter(n['kernel'] for n in launch)==Counter(n['kernel'] for n in bound),'Runtime FIA kernel census differs')
    need(all(n['call_id']==0 and n['pid']==owner and n['block_dim']==16 for n in launch),
        'FIA task outside real warmup owner/core budget')
    need(all(n['block_dim']==16 for n in tasks),'Unexpected per-node core budget')


def audit(attempt,matrix_name='matrix_v5.json'):
    directory=ROOT/'outputs'/attempt
    completion=json.loads((directory/'completion.json').read_bytes())
    need(completion['status']=='CAPTURE_COMPLETED' and completion['card']==5 and
        completion['device_lock_acquired'] and completion['device_lock_released'] and
        completion['owned_container_removed'] and not completion['cleanup_errors'] and
        not completion['external_container_mutations'],'Unclean or incomplete controlled capture')
    need(len(completion['cases'])==1,'Need one exact case')
    cid=completion['cases'][0]['case_id'];folder=directory/cid
    result=json.loads((folder/'result.json').read_bytes());case=json.loads((folder/'case.json').read_bytes())
    matrix=json.loads((ROOT/'setup'/matrix_name).read_bytes())
    need(sha(folder/'case.json')==matrix['case_sha256'][cid],'Unregistered case')
    need(sha(folder/'result.json')==completion['cases'][0]['result_sha256'],'Result/receipt mismatch')
    source=json.loads((directory/'source_manifest.json').read_bytes())
    for name,digest in matrix['runtime_sha256'].items():need(source[name]==digest,'Wrong frozen runtime: '+name)
    for name,digest in source.items():need(sha(directory/'source'/name)==digest,'Changed actual source: '+name)
    need(result['status']=='CAPTURE_COMPLETED' and len(result['backend_graphs'])==1 and
        result['actual_backend_calls']==3 and result['single_graph_backend_replay'] and
        result['repeated_all_outputs_equal'] and result['parameters_unchanged'],'Graph or replay contract failed')
    request_map=json.loads((folder/'inputs.json').read_bytes())['request_rows']
    need([r['slot'] for r in request_map]==list(range(case['batch'])) and
        [r['request_id'] for r in request_map]==case['request_ids'] and
        len(set(case['request_ids']))==case['batch'] and all(r['prompt_length']==2048 for r in request_map),
        'Actual request mapping incomplete')
    need(all(c['inputs_unchanged'] and all(len(v['request_sha256'])==case['batch'] and
        v['shape'][0]==case['batch'] for v in c['values']) for c in result['calls']),
        'Missing actual request outputs or changed inputs')
    bindings=json.loads((folder/'compiled_engine_bindings.json').read_bytes())
    need(bindings['all_user_inputs_bound'] and bindings['same_live_objects_replayed'],
        'Actual input binding to reused fullgraph missing')
    fx=json.loads((folder/'fx_graph_0.json').read_bytes())
    fx_fia=[n for n in fx['nodes'] if n['target']=='npu.npu_fused_infer_attention_score.default']
    need(len(fx_fia)==28 and [n['layer'] for n in fx_fia]==list(range(28)) and
        fx['backend_output_matches_observed_tuple'],'Missing actual FX layer/output mapping')
    nodes,_=graph(folder);ge_fia=[n for n in nodes if n['type']=='FusedInferAttentionScore']
    names={n['name'] for n in ge_fia};need(len(names)==28,'Not all 28 GE FIA nodes present')
    aliases=alias_mapping(folder)
    if case['capture_aliases']:
        edges=aliases['output_edges'];need(len(edges)==113,'Wrong observed output arity')
        for layer,node in enumerate(ge_fia):
            edge=edges[4+4*layer]
            need(edge['producer']==node['name'] and edge['producer_type']=='FusedInferAttentionScore' and
                edge['producer_output_index']==0,'Actual GE output alias does not match layer order')
            need(all(edges[1+4*layer+i]['edge']==node['inputs'][i] for i in range(3)),
                'Observed QKV are not the actual corresponding FIA input edges')
    trace=directory/'native_extract'
    if not trace.exists():extract(directory/(cid+'.log'),trace)
    native=parse_native(trace/'native_lines.txt')
    verify_census(native,names,result['owner_pid'])
    bound=native['binaries'];tasks=native['tasks'];launch=native['launches']
    events=[json.loads(line) for line in (folder/'host_events.jsonl').read_text().splitlines()]
    vendor=json.loads((ROOT/'vendor_stage_a5/vendor_manifest.json').read_bytes())
    abi=vendor['public_abi'];schema_path=ROOT/abi['schema_file']
    need(sha(schema_path)==abi['schema_sha256'],'Public ABI fingerprint changed')
    public=verify(folder,json.loads(schema_path.read_bytes()))
    need(len(public)==len(events)>=1,'Missing final public tiling')
    for event in events:
        need(event['node'] in names,'Selector belongs to another graph node')
        need(any(s['node']==event['node'] and s['pid']==event['pid'] and s['tid']==event['tid'] and
                 s['key']==event['tiling_key'] for s in native['selected']),'Host/JIT actual selection differs')
    keys={p['actual_host_key'] for p in public};raws={e['raw_hex'] for e in events}
    need(len(keys)==len(raws)==1,'This fixed-shape matrix requires one unambiguous shared kernel plan')
    inventory=json.loads((folder/'jit_compile_inventory.json').read_bytes());docs=[];objects={}
    for row in inventory['files']:
        if row['file'].endswith('.o'):objects[Path(row['file']).stem]=row
        if row.get('private_copy'):
            p=folder/row['private_copy'];need(sha(p)==row['sha256'],'Compiler metadata changed')
            docs.append((Path(row['file']).name,json.loads(p.read_bytes()),row['sha256']))
    isolated=[d for n,d,h in docs if '_isolate_tiling_' in n]
    compiled={d['kernelName']:d for n,d,h in docs if n.startswith('te_fusedinferattentionscore_')}
    need(set(compiled)==set(objects)=={n['kernel'] for n in bound},'Actual binary inventory does not cover every node')
    need(bool(isolated),'Missing actual compiler tiling')
    for d in isolated:
        actual,final=d['run_info'],d['tiling_info']
        need(str(actual['tiling_key'])==str(final['tiling_key'])==next(iter(keys)) and
            actual['tiling_data']==final['tiling_data']==final['raw_run_info']['tiling_data']==next(iter(raws)),
            'Compiler did not consume the exact actual public plan')
        need(final['tiling_data_size']==1576 and final['block_num']==16 and final['tiling_info_completed'],
            'Incomplete compiler plan')
    for name,d in compiled.items():need(d['sha256']==objects[name]['sha256'],'Compiled binary SHA differs')
    iterations=[]
    for layer,node in enumerate(ge_fia):
        for call in range(3):
            prefix='FusedInferAttentionScore.'+node['name']+'.'
            files=[p for p in csv_inventory(folder) if p.name.startswith(prefix) and p.parent.name==str(call)
                   and p.parent.parent.parent.name=='graph_1']
            need(len(files)==1,'Missing/ambiguous actual per-layer FIA CSV: '+node['name']+' '+str(call))
            data=csv_rows(files[0]);batch=case['batch']
            for boundary,index,shape in [('Input',0,[batch,16,2048,128]),('Input',1,[batch,8,2048,128]),
                ('Input',2,[batch,8,2048,128]),('Output',0,[batch,16,2048,128])]:
                row=data[(boundary,index)]
                need(row['DataType']=='DT_FLOAT16' and list(map(int,row['Shape'].split('x')))==shape,
                    'Actual node tensor geometry differs')
            iterations.append({'layer':layer,'node':node['name'],'call_id':call,'request_slots':list(range(batch)),
                'csv':str(files[0].relative_to(folder)),'sha256':sha(files[0])})
    return {'schema':'fault28.fia_fullmodel.coverage.v1','status':'PASS_ALL_MATCHED_LAYERS_REQUESTS_ACTUAL_GRAPH',
        'case_id':cid,'attempt':attempt,'batch':case['batch'],'logical_fia_nodes':28,
        'actual_fia_node_executions':84,'layer_request_executions':84*case['batch'],
        'native_selector_evaluations':len(events),'shared_compiled_kernels':len(compiled),
        'actual_request_mapping':request_map,
        'all_node_kernel_bindings':bound,'actual_static_tasks':tasks,'runtime_instantiations':launch,
        'all_layer_request_iterations':iterations,'actual_host_key':next(iter(keys)),
        'actual_m':public[0]['decoded']['innerSplitParams.mBaseSize'],
        'no_splitkv':all(p['decoded']['fdParams.numOfFdHead']==0 for p in public),
        'public_plan':public,'compiled_binary_sha256':{n:r['sha256'] for n,r in objects.items()},
        'raw_log_sha256':json.loads((trace/'provenance.json').read_bytes())['source_log_sha256'],
        'matrix_sha256':sha(ROOT/'setup'/matrix_name),'result_sha256':sha(folder/'result.json'),
        'analysis_source_sha256':sha(Path(__file__)),
        'limits':['Selector count is JIT compilation count, not runtime layer coverage.',
                  'All requests are rows in each real full-batch node; a request is not a separate FIA launch.',
                  'Device scalar observations may aggregate shared kernels; node coverage uses actual task/binary/iteration evidence.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:r=audit(a.attempt);rc=0
    except Exception as e:r={'status':'FAIL_COVERAGE','attempt':a.attempt,'error':str(e)};rc=1
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k in ('status','error','case_id','logical_fia_nodes','actual_fia_node_executions','actual_m','shared_compiled_kernels')}))
    return rc


if __name__=='__main__':raise SystemExit(main())
