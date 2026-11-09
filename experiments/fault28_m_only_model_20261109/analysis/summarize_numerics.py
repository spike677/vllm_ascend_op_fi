"""Independently join common observation records, actual calls, cache and logits.

No host mode or expected first layer determines numerical differences. B/F
comparisons were performed on every element by the same CPU observer. Baseline
references and complete logits are rehashed; full F tensors are not retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

MODES=('calibration','native','controlled','controlled','off')


def require(ok,text):
    if not ok:raise ValueError(text)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def reference_path(folder,original):
    prefix='/evidence/'+folder.name+'/'
    require(original.startswith(prefix),'Unexpected private-reference mount')
    relative=Path(original[len(prefix):])
    require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='private_reference',
            'Unsafe private-reference path')
    path=folder/relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()),
            'Missing private native reference')
    return path


def observation(snapshot,phase,folder,kind,layer):
    row=snapshot['exact_numeric_observation']
    require(row['mode']==phase['mode'] and row['stage']==phase['stage'] and row['shape']==snapshot['shape']
            and row['full_sha256']==snapshot['full_sha256'] and row['kind']==kind
            and row['layer_index']==layer and row['dtype']==snapshot['dtype']=='torch.float16',
            'Numeric snapshot identity/dtype mismatch')
    if phase['mode']=='native':
        path=reference_path(folder,row['private_reference_file'])
        require(path.is_file() and sha(path)==row['reference_sha256']==row['full_sha256'],'Changed private baseline reference')
    if phase['mode'] in ('controlled','off'):
        c=row['comparison'];elements=int(np.prod(row['shape']))
        require(c['elements']==elements and c['changed_elements']==sum(c['per_request_changed_elements'])
                ==sum(c['per_query_changed_elements']),'Inconsistent elementwise comparison totals')
        require(len(c['per_request_changed_elements'])==snapshot['shape'][0] and
                c['last_query_changed_elements']==c['per_query_changed_elements'][-1], 'Incomplete request/last-query observation')
        require(c['bitwise_equal']==(row['reference_sha256']==row['full_sha256']),'Byte equality contradicts independent digest')
        if phase['mode']=='off':require(c['bitwise_equal'] and c['changed_elements']==0,'OFF did not restore complete tensor')
    return row


def reference_matches(row,native):
    require(row['reference_sha256']==native['full_sha256'],'Comparison refers to another native baseline')
    require(row['comparison']['last_query_B_sha256']==native['last_query_sha256'] and
            row['comparison']['last_query_F_sha256']==row['last_query_sha256'],
            'Last-query comparison refers to another native/fault tensor')


def compact(row):
    if 'comparison' not in row:return {'full_sha256':row['full_sha256'],'last_query_sha256':row['last_query_sha256']}
    return {'full_sha256':row['full_sha256'],'last_query_sha256':row['last_query_sha256'],
            **{k:row['comparison'][k] for k in ('changed_elements','per_request_changed_elements','last_query_changed_elements',
                                               'queries_with_changes','max_abs','bitwise_equal')}}


def summarize(folder):
    result=json.loads((folder/'result.json').read_bytes())
    require(result['status']=='CAPTURE_COMPLETED','Incomplete capture')
    phases=json.loads((folder/'phase_manifest.json').read_bytes())['phases']
    require(len(phases)==10 and [p['mode'] for p in phases]==[m for m in MODES for _ in (0,1)],'Wrong registered run order')
    calls={};allrows={};rows=[]
    for item in result['calls']:
        file=folder/item['file'];require(sha(file)==item['sha256'],'Call sidecar changed')
        call=json.loads(file.read_bytes());calls[call['step'],call['layer_index']]=call
    for phase in phases:
        step=phase['step'];layer_rows=[]
        cache_path=folder/phase['cache_observation_file']
        require(sha(cache_path)==phase['cache_observation_sha256'],'Cache evidence changed')
        cache=json.loads(cache_path.read_bytes())
        require(len(cache['layers'])==28 and cache['step']==step and cache['cache_length']==phase['cache_length_after'],
                'Actual full cache missing/inconsistent')
        for layer in range(28):
            call=calls[step,layer]
            attention=observation(call['output'],phase,folder,'attention',layer)
            item=cache['layers'][layer];require(item['layer_index']==layer,'Reordered cache layer')
            parts={kind:observation(item[kind],phase,folder,'cache_'+kind,layer) for kind in ('keys','values')}
            if phase['mode'] in ('controlled','off'):
                native_step=2+(phase['stage']=='decode')
                reference_matches(attention,allrows[native_step,layer,'attention'])
                for kind,value in parts.items():reference_matches(value,allrows[native_step,layer,kind])
            allrows[step,layer,'attention']=attention
            for kind in parts:allrows[step,layer,kind]=parts[kind]
            layer_rows.append({'layer_index':layer,'attention':compact(attention),
                               'cache_keys':compact(parts['keys']),'cache_values':compact(parts['values'])})
        logits=observation(phase['logits'],phase,folder,'logits',None)
        if phase['mode'] in ('controlled','off'):
            native=phases[2+(phase['stage']=='decode')]['logits']['exact_numeric_observation']
            reference_matches(logits,native)
        file=folder/phase['private_logits_file'];require(sha(file)==logits['full_sha256'],'Complete private logits changed')
        rows.append({'step':step,'mode':phase['mode'],'stage':phase['stage'],'layers':layer_rows,'logits':compact(logits)})
    # Validate repeat and recovery with original full digests, including the
    # independently observed final query and both cache parts on all layers.
    for left,right in ((0,1),(2,3),(1,4)):
        for stage in (0,1):
            a,b=left*2+stage,right*2+stage
            for layer in range(28):
                for kind in ('attention','keys','values'):
                    x,y=allrows[a,layer,kind],allrows[b,layer,kind]
                    require(x['full_sha256']==y['full_sha256'] and x['last_query_sha256']==y['last_query_sha256'],
                            'Calibration/repeat/recovery not exact: '+str((a,b,layer,kind)))
            require(rows[a]['logits']['full_sha256']==rows[b]['logits']['full_sha256'],'Logits repeat/recovery not exact')
    faults=[]
    for step in (4,5,6,7):
        row=rows[step]
        faults.append({'step':step,'stage':row['stage'],'run_index':step//2,
                       'attention_changed_layers':[x['layer_index'] for x in row['layers'] if x['attention']['changed_elements']],
                       'attention_last_query_changed_layers':[x['layer_index'] for x in row['layers'] if x['attention']['last_query_changed_elements']],
                       'cache_changed_layers':[x['layer_index'] for x in row['layers'] if x['cache_keys']['changed_elements'] or x['cache_values']['changed_elements']],
                       'logits':row['logits']})
    return {'schema':'fault28.m_only.model_numeric_summary.v1','status':'PASS_EXACT_OBSERVATION_INTEGRITY',
            'case_id':result['case_id'],'batch_size':phases[0]['logits']['shape'][0],
            'case_result_sha256':sha(folder/'result.json'),'phase_manifest_sha256':sha(folder/'phase_manifest.json'),
            'fault_phases':faults,'phases':rows,'repeat_and_off_exact':True,
            'logits_change_observed':any(r['logits']['changed_elements'] for r in faults),
            'limits':['All B/F elements compared by common runtime CPU observer. Baseline reference and full logits rehashed here; complete F Attention/KV tensors not retained.',
                      'Separate host/launch evidence establishes M-only, separate msprobe analysis establishes observed first difference.',
                      'No measured change is not a failure of injection if actual strategy switched correctly.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:report=summarize(a.case_dir);rc=0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as e:
        report={'schema':'fault28.m_only.model_numeric_summary.v1','status':'FAIL_NUMERIC_INTEGRITY','error':str(e)};rc=1
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report['analysis_source_sha256']=sha(Path(__file__))
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','case_id','logits_change_observed','error')}))
    raise SystemExit(rc)
