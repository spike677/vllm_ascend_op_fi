"""Actual GE task instantiation + JIT selection + completed graph iterations.

GE instantiates one static FIA task and replays the containing model. It does
not dispatch NNOP tiling once per graph replay. The runtime static key 0 must
not be confused with the CANN JIT template key used by the host compiler.
"""
import hashlib
import json
from pathlib import Path
import re

SELECT=re.compile(r'OP\((\d+),.*?\[GenTilingKey\]\[(\d+)\] OpName:\[([^\]]+)\] FIA tilingKey_: (\d+)\.')
BINARY=re.compile(r'FillTaskDefAfterGenTask:Set kernel_name\[([^\]]+)\] for the kernel_def of node\[([^,]+), FusedInferAttentionScore\]')
LAUNCH=re.compile(r'StreamLaunchKernelV2: kernel info : .*?stream_id=(\d+), task_id=(\d+),.*?kernel_name=((?:te_fusedinferattentionscore_[0-9a-f]+)|(?:FusedInferAttentionScore_[0-9a-f]+_[0-9]+)),.*?coreDim=(\d+),.*?schemMode=(\d+)')
TASK=re.compile(r'Distribute:KernelTaskInfo Distribute Success, op: ([^,]+), taskid: (\d+), blockdim: (\d+), streamid: (\d+)')


def require(ok,text):
    if not ok:raise ValueError(text)


def parse(path,folder):
    windows=[];active=None;selected=[];binaries=[];launches=[];tasks=[];excerpts=[]
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for number,raw in enumerate(stream,1):
            digest.update(raw);line=raw.decode('utf-8','strict').rstrip()
            if line.startswith('{'):
                try:event=json.loads(line)
                except json.JSONDecodeError:event={}
                if event.get('schema')=='fault28.fia_graph.call_marker.v1':
                    if event['event']=='enter':
                        require(active is None,'Nested graph call marker');active=event
                        active['_line']=number
                    elif event['event']=='exit':
                        require(active is not None and all(event[k]==active[k] for k in ('call_id','pid','batch','layout')),'Unmatched graph exit')
                        windows.append({'call_id':active['call_id'],'pid':active['pid'],'enter_line':active['_line'],
                                        'exit_line':number,'phase':active['phase']});active=None
            selected_match=SELECT.search(line);binary_match=BINARY.search(line)
            launch_match=LAUNCH.search(line);task_match=TASK.search(line)
            if selected_match:
                x=selected_match
                selected.append({'pid':int(x[1]),'tid':int(x[2]),'node':x[3],'jit_template_key':x[4],'line':number})
            if binary_match:
                binaries.append({'kernel_name':binary_match[1],'node':binary_match[2],'line':number})
            if launch_match:
                x=launch_match
                launches.append({'pid':int(re.search(r'RUNTIME\((\d+),',line)[1]),'stream_id':int(x[1]),
                    'runtime_enqueue_task_id':int(x[2]),'kernel_name':x[3], 'block_dim':int(x[4]),
                    'schedule_mode':int(x[5]),'call_id':active['call_id'] if active else None,'line':number})
            actual_task=task_match and 'FusedInferAttentionScore' in task_match[1]
            if actual_task:
                x=task_match
                tasks.append({'node':x[1],'ge_task_id':int(x[2]),'block_dim':int(x[3]),'stream_id':int(x[4]),'line':number})
            if selected_match or binary_match or launch_match or actual_task:
                excerpts.append({'line':number,'text':line})

    require(active is None and [w['call_id'] for w in windows]==[0,1,2],'Incomplete graph call census')
    if not binaries and len(launches)==len(selected)==len(tasks)==1 and launches[0]['kernel_name'].startswith('FusedInferAttentionScore_'):
        entry=launches[0]['kernel_name'];base,key=entry.rsplit('_',1)
        require(key==selected[0]['jit_template_key'],'Runtime built-in entry selects a different key')
        binaries=[{'kernel_name':base,'node':tasks[0]['node'],'line':launches[0]['line'],
                   'source':'Actual runtime-selected builtin entry; no custom JIT binary metadata observed'}]
    require(len(selected)==len(binaries)==len(launches)==len(tasks)==1,'Need one actual selection/static FIA graph task: '+str((len(selected),len(binaries),len(launches),len(tasks))))
    s,b,l,t=selected[0],binaries[0],launches[0],tasks[0]
    require(s['node']==b['node']==t['node'] and (b['kernel_name']==l['kernel_name'] or b['kernel_name']+'_'+s['jit_template_key']==l['kernel_name']),'GE task/kernel identity mismatch')
    require(l['call_id']==0 and l['pid']==windows[0]['pid'] and len({w['pid'] for w in windows})==1 and l['block_dim']==t['block_dim']==16 and l['stream_id']==t['stream_id'],
            'Static FIA task not instantiated in warmup with verified core budget')
    csvs=[]
    for call in (0,1,2):
        files=[p for p in (folder/'msprobe_graph').rglob('*.csv')
               if p.name.startswith('FusedInferAttentionScore.'+s['node']+'.') and p.parent.name==str(call)
               and p.parent.parent.parent.name=='graph_1']
        require(len(files)==1,'Missing/ambiguous actual FIA iteration CSV')
        csvs.append({'call_id':call,'path':files[0].relative_to(folder).as_posix(),
                     'sha256':hashlib.sha256(files[0].read_bytes()).hexdigest()})
    host=folder/'host_events.jsonl'
    if host.is_file():
        events=[json.loads(line) for line in host.read_text().splitlines()]
        require(len(events)==1,'Unaccounted host callback count')
        h=events[0]
        require(h['pid']==s['pid'] and h['tid']==s['tid'] and h['node']==s['node']
                and h['tiling_key']==s['jit_template_key'],'Host public data not bound to actual JIT selection')
        require(h['block_dim']==l['block_dim'] and h['schedule_mode']==l['schedule_mode'],
                'Host versus actual task launch metadata mismatch')
    return {'schema':'fault28.actual_graph_native_plan.v1','log_sha256':digest.hexdigest(),
            'selected':s,'binary':b,'static_task_launch':l,'ge_static_task':t,
            'graph_calls':windows,'actual_fia_output_iterations':csvs,'native_excerpts':excerpts,
            'kernel_argument_raw_observed':False,
            'scope':'One static GE FIA task instantiated, three completed graph iterations. JIT template key is separate from runtime static key0. Raw kernel args not read.'}
