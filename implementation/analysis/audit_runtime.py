"""Actual eager call/host raw/native launch census, separate from first difference."""
import argparse,collections,hashlib,json,re
from pathlib import Path
from public_plan import verify
from statistics_values import load_step,is_tensor

ROOT=Path('/home/fault28-reduce-replay-20261113')
API=re.compile(r'NPU\.npu_fused_infer_attention_score\.\d+\.forward\Z')
LAUNCH=re.compile(r'RUNTIME\((\d+),.*?StreamLaunchKernelV2: kernel info : .*?stream_id=(\d+), task_id=(\d+),.*?kernel_name=(FusedInferAttentionScore_[^,]+),.*?coreDim=(\d+), taskRation=(\d+)')
SELECT=re.compile(r'OP\((\d+),.*?\[GenTilingKey\]\[(\d+)\] OpName:\[([^\]]+)\] FIA tilingKey_: (\d+)\.')
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def native_log(path):
    rows=[];active=None;forward=None;excerpts=[];errors=[]
    for number,line in enumerate(path.open(errors='strict'),1):
        if line.startswith('{'):
            try:e=json.loads(line)
            except json.JSONDecodeError:e={}
            if e.get('schema')=='fault28.eager.forward_marker.v1':
                if e['event']=='enter':need(forward is None,'Nested forward');forward=e['call_id']
                else:need(forward==e['call_id'] and active is None,'Incomplete forward');forward=None
            if e.get('schema')=='fault28.eager.api_marker.v1':
                excerpts.append({'line':number,'text':line.rstrip()})
                if e['event']=='enter':
                    need(active is None and forward==e['forward'],'Nested/outside FIA');active={'marker':e,'selected':[],'launches':[]}
                else:
                    need(active is not None and all(e[k]==active['marker'][k] for k in ('api_index','forward','layer')),'Mismatched FIA exit')
                    active['exit']=e;rows.append(active);active=None
        s=SELECT.search(line);l=LAUNCH.search(line)
        if s:
            need(active is not None,'Selector outside synchronized API window')
            active['selected'].append({'pid':int(s[1]),'tid':int(s[2]),'node':s[3],'key':s[4]})
        if l:
            need(active is not None,'FIA launch outside synchronized API window')
            active['launches'].append({'pid':int(l[1]),'stream':int(l[2]),'task':int(l[3]),'kernel':l[4],'block_dim':int(l[5]),'task_ratio':int(l[6])})
        if s or l:excerpts.append({'line':number,'text':line.rstrip()})
        if '[ERROR]' in line:errors.append({'line':number,'text':line.rstrip()})
    need(active is None and forward is None,'Incomplete native call/forward log')
    need(not errors,'Native runtime reported an error')
    return rows,excerpts

def audit(folder):
    folder=Path(folder);attempt=folder.parent;result=json.loads((folder/'result.json').read_text());case=json.loads((folder/'case.json').read_text())
    completion=json.loads((attempt/'completion.json').read_text())
    need(completion['status']=='CAPTURE_COMPLETED' and completion['card'] in (2,5) and completion['device_lock_acquired'] and completion['device_lock_released'] and completion['owned_container_removed'] and not completion['cleanup_errors'],'Capture/cleanup not complete')
    record=[c for c in completion['cases'] if c['case_id']==case['case_id']];need(len(record)==1 and record[0]['result_sha256']==sha(folder/'result.json'),'Completion/result mismatch')
    frozen=json.loads((attempt/'source_manifest.json').read_text())
    for name,digest in frozen.items():need(sha(attempt/'source'/name)==digest,'Frozen source mismatch: '+name)
    need(result['status']=='CAPTURE_COMPLETED' and result['execution_mode']=='eager' and result['torch_compile_calls']==0 and result['actual_eager_forwards']==len(result['calls']) and result['repeated_all_outputs_equal'],'Eager/repeat contract failed')
    config=json.loads((folder/'execution_config.json').read_text());need(config['mode']=='eager' and config['torch_compile_forbidden'] ,'Compiled backend was requested')
    layers=1 if case['capture_scope']=='calibration' else 28
    batch=case['batch'];forwards=len(result['calls'])
    expected=[(c,l) for c in range(forwards) for l in range(layers)]
    markers=[m for c in result['calls'] for m in c['api_markers']]
    need([(m['forward'],m['layer']) for m in markers]==expected and result['actual_eager_FIA_API_calls']==forwards*layers,'Not every real FIA call covered')
    need(all(c['inputs_unchanged'] and len(c['values'])==(1 if layers==1 else 3 if case['capture_scope']=='harvest' else 113) and all(len(v['request_sha256'])==batch and v['shape'][0]==batch for v in c['values']) for c in result['calls']),'Missing all-request observation')
    if layers==28:
        need(result['parameters_unchanged'] and json.loads((folder/'parameters_receipt.json').read_text())['unchanged'],'Weights changed')
        requests=json.loads((folder/'inputs.json').read_text())['request_rows']
        need([r['slot'] for r in requests]==list(range(batch)) and [r['request_id'] for r in requests]==case['request_ids'] and len(set(case['request_ids']))==batch,'Missing distinct request mapping')
    else:requests=None
    raw=verify(folder,json.loads(Path('/home/fault28-fia-fullmodel-20261109/build/evidence_a4/fia_abi_schema.json').read_text()))
    calls,lines=native_log(attempt/(case['case_id']+'.log'))
    need([(x['marker']['forward'],x['marker']['layer']) for x in calls]==expected and len(raw)==len(calls),'Raw/actual API count mismatch')
    events=[json.loads(l) for l in (folder/'host_events.jsonl').read_text().splitlines()]
    census=[]
    for call,event,plan in zip(calls,events,raw):
        need(len(call['selected'])==len(call['launches'])==1,'No unique native selection/launch')
        s=call['selected'][0];launch=call['launches'][0];m=call['marker']
        need(s['pid']==event['pid']==launch['pid']==result['owner_pid'] and s['tid']==event['tid'] and s['key']==('109000000000400000' if plan['denominator_replay'] else plan['actual_host_key']),'Foreign or mismatched selection')
        need(launch['kernel'].endswith('_'+plan['actual_host_key']) and launch['block_dim']==16 and launch['task_ratio']==2,'Wrong actual native kernel')
        need(m['input_shapes']==[[batch,16,2048,128],[batch,8,2048,128],[batch,8,2048,128]] and call['exit']['output_shape']==[batch,16,2048,128],'Wrong actual API geometry')
        census.append({'forward':m['forward'],'layer':m['layer'],'request_slots':list(range(batch)),
            'api_index':m['api_index'],'actual_m':event['selected_m'],'actual_key':plan['actual_host_key'],'pre_replay_selector_key':s['key'],'denominator_replay':plan['denominator_replay'],'launch':launch,
            'raw_sha256':plan['raw_sha256']})
    vendor=json.loads((attempt/(case['case_id']+'.vendor_launch.json')).read_text())['verified_vendor']
    loaded=json.loads((folder/'loaded_native_libraries.json').read_bytes())['custom_library_file_sha256']
    expected={str(Path(vendor['vendor_root'])/row['path']):row['sha256'] for row in vendor['artifacts']}
    for suffix in ('op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so','op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so'):
        matches=[path for path in loaded if path.endswith(suffix)]
        need(len(matches)==1 and loaded[matches[0]]==expected[matches[0]],'Actual loaded host/proto library differs from the pinned package')
    binaries=[x for x in vendor['artifacts'] if x['path'].endswith('.o') and '/kernel/ascend910b/' in x['path']]
    need(len(binaries)==1,'Wrong private eager binary inventory')
    api_rows=[]
    if case['capture_msprobe']:
        for step in result['statistics_steps']:
            data=load_step(folder/f'observation/step{step}/rank0/dump.json')
            actual=[e for e in data['entries'] if API.fullmatch(e['name'])];need(len(actual)==layers,'Missing original native FIA API records')
            for layer,e in enumerate(actual):
                args=e['node']['input_args'];out=e['node']['output']
                need(len(args)==3 and all(is_tensor(t) for t in args) and [t['shape'] for t in args]==[[batch,16,2048,128],[batch,8,2048,128],[batch,8,2048,128]],'Wrong captured QKV geometry')
                need(len(out)==2 and out[0]['shape']==[batch,16,2048,128] and out[1]['shape']==[0],'Wrong captured original FIA return')
                need(any(p.startswith('Module.') for p in e['parents']),'Missing original containing module')
                api_rows.append({'step':step,'layer':layer,'node':e['name'],'parents':e['parents'],'stack':e['frames'],'request_slots':list(range(batch))})
    return {'schema':'fault28.eager.runtime_coverage.v1','status':'PASS_ACTUAL_EAGER_NATIVE_COVERAGE','case_id':case['case_id'],
        'physical_card':completion['card'],'layers':layers,'requests':batch,'eager_forwards':forwards,'actual_FIA_executions':len(census),
        'layer_request_executions':batch*len(census),'all_outputs_repeat_exact':True,'torch_compile_calls':0,
        'actual_request_mapping':requests,'native_API_statistics':api_rows,'census':census,
        'public_plan_unique':[p for i,p in enumerate(raw) if p['raw_sha256'] not in {v['raw_sha256'] for v in raw[:i]}],
        'private_binary_sha256':binaries[0]['sha256'],'native_log_sha256':sha(attempt/(case['case_id']+'.log')),
        'native_log_excerpt':lines,'result_sha256':sha(folder/'result.json'),'source_manifest_sha256':sha(attempt/'source_manifest.json'),
        'limits':['A request is a row in a batch kernel, not a separate launch.',
            'Public host raw is decoded independently; device kernel argument bytes are not directly observed.',
            'Inner reduction width is derived from actual raw plus executed source variant, not a measured device instruction trace.',
            'Complete tensors remain private. Native statistics alone cannot prove alias/value identity or unique physical root cause.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:r=audit(a.case);rc=0
    except Exception as e:r={'status':'FAIL_RUNTIME_AUDIT','error':str(e)};rc=1
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k in ('status','error','case_id','actual_FIA_executions','layer_request_executions')}));return rc

if __name__=='__main__':raise SystemExit(main())
