"""Measured Softmax entry scalars, independently decoded from device printf.

Never synthesize observed rows from a selected/expected profile. Missing data
or disagreement between official decoding and per-call runtime logs fails.
This records column granularity, not internal reduction instruction order.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path('/home/fault28-fia-grain-20261109')
QUERIES=(63,255,511,767,1023,1535,2047)
NAMES=('block','m','tile','row','rows','s2','base','columns','actual','prune','brc')
PATTERN=re.compile(r'F28_GRAIN_V1 '+r' '.join(name+r'=(\d+)' for name in NAMES)+r'(?:\s|$)')
RUNTIME=re.compile(r'RUNTIME\((\d+),.*?ParsePrintToLog: PrintInfo: ')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()


def parse(line):
    match=PATTERN.search(line)
    if not match:return None
    row=dict(zip(NAMES,map(int,match.groups())))
    require(row['m'] in (128,256,512) and row['base']==512 and 0<=row['block']<32,'Unsupported device geometry')
    require(row['tile']%row['m']==0 and row['tile']<=row['row']<row['row']+row['rows']<=row['tile']+row['m']<=2048,
            'Invalid actual row interval')
    require(0<row['actual']<=row['columns']<=512 and row['columns']%32==0 and
            row['prune'] in (0,1) and row['brc'] in (0,1) and 0<=row['s2']<4,
            'Invalid actual Softmax shape')
    require(any(row['row']<=q<row['row']+row['rows'] for q in QUERIES),'Unregistered query observation')
    return row


def key(row):return tuple(row[k] for k in NAMES)


def expand(rows,m,prune):
    """Validate measured widths against actual public raw, without generating them."""
    points={}
    for row in rows:
        require(row['m']==m and row['prune']==prune and row['brc']==int(m<=128),
                'Device scalar does not match independently observed actual key/M')
        for q in QUERIES:
            if not row['row']<=q<row['row']+row['rows']:continue
            identity=(q,row['s2']);require(identity not in points,'Duplicate sampled query/inner tile')
            expected=min(512,((row['tile']+m-row['s2']*512+31)//32)*32) if prune else 512
            require(expected>0 and row['columns']==row['actual']==expected,
                    'Measured columnCount differs from the selected causal kernel rule')
            points[identity]={'query':q,'s2_index':row['s2'],'key_begin':row['s2']*512,
                'measured_columns':row['columns'],'measured_actual_columns':row['actual'],
                'query_tile_begin':row['tile'],'query_tile_rows':m,'vector_row_begin':row['row'],
                'vector_rows':row['rows'],'device_aiv_block':row['block']}
    expected_points={(q,index) for q in QUERIES for index in range((q//512)+1)}
    require(set(points)==expected_points,'Missing device observation for a registered query/inner tile')
    for q in QUERIES:
        legal=[]
        for (query,index),p in sorted(points.items()):
            if query==q:legal.extend(range(p['key_begin'],min(q+1,p['key_begin']+p['measured_columns'])))
        require(legal==list(range(q+1)),'Observed partition loses/duplicates an unmasked key')
    return list(points.values())


def audit(folder,execution,receipt_file):
    folder=folder.resolve(strict=True)
    require(execution['status']=='PASS_RECORDED_EXECUTION' and execution['case_id']==folder.name,
            'Device observations need an independent accepted actual execution')
    require(execution['public_host'],'No measured host/ABI evidence')
    hosts=execution['public_host'];require(len({r['actual_host_key'] for r in hosts})==1,'Ambiguous actual kernel')
    m=hosts[0]['decoded']['innerSplitParams.mBaseSize'];prune=int(hosts[0]['actual_host_key'].startswith('109'))
    receipt=json.loads(receipt_file.read_bytes())
    require(receipt['returncode']==0 and receipt['npu_devices_mapped'] is False and
            receipt['container_image']=='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9',
            'No matching official SDK CPU decode receipt')
    require(receipt['helper_sha256']==sha(Path(__file__).with_name('sdk_debug_tool_cpu.py')),
            'Unreviewed SDK decoder wrapper')
    actual_buffers=sorted((folder/'private_device_debug').rglob('asc_kernel_data_*.bin'))
    require({str(p) for p in actual_buffers}=={r['file'] for r in receipt['source_files']},'Not all actual device scalar buffers decoded')
    for row in receipt['source_files']:
        path=Path(row['file']);require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Changed raw scalar buffer')
    official=[];parsed_bindings=[]
    for row in receipt['parsed_outputs']:
        file=Path(row['file']);require(file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],'Changed SDK parser result')
        parsed_bindings.append(row)
        if str(file)!=receipt['saved_original_sdk_stdout']:continue
        require(sha(file)==receipt['decoder_stdout_sha256'] and file.read_text()==receipt['stdout'],
                'Official decoder stdout changed')
        for line in file.read_text().splitlines():
            observation=parse(line)
            if observation:
                header=re.search(r'\[AIV Block (\d+)/32\]',line)
                require(header is not None and int(header[1])==observation['block'],'SDK block header mismatch')
                official.append(observation)
    require(official,'No device scalar records in actual official parser output')
    windows=execution['native']['graph_calls'];observed_lines=[];runtime_rows=[]
    require([w['call_id'] for w in windows]==[0,1,2] and len({w['pid'] for w in windows})==1,
            'Incomplete actual graph execution census')
    log=folder.parent/(folder.name+'.log')
    with log.open(errors='strict') as stream:
        for number,line in enumerate(stream,1):
            origin=RUNTIME.search(line)
            if not origin:continue
            observation=parse(line)
            if observation is None:continue
            require(int(origin[1])==windows[0]['pid'] and number>windows[0]['enter_line'],
                    'Actual printf does not belong to the observed graph process')
            runtime_rows.append(observation)
            observed_lines.append({'line':number,'sha256':hashlib.sha256(line.encode()).hexdigest(),'values':observation})
    counts=Counter(map(key,official))
    require(counts==Counter(map(key,runtime_rows)),
            'Official scalar buffers and independently framed runtime records disagree')
    require(set(counts.values())=={3},'Device scalar multiplicities do not match the three actual graph executions')
    unique={key(row):row for row in official}
    points=expand(list(unique.values()),m,prune)
    return {'schema':'fault28.fia.measured_softmax_columns.v1','status':'PASS_MEASURED_DEVICE_SOFTMAX_COLUMNS',
        'case_id':folder.name,'actual_m':m,'actual_key':hosts[0]['actual_host_key'],
        'actual_causal_variant':bool(prune),'sampled_queries':list(QUERIES),'query_tiles':points,
        'actual_graph_calls':windows,'scalar_shape_multiplicities':sorted(counts.values()),
        'individual_scalar_to_graph_call_pairing_proven':False,
        'canonical_scalar_records':len(official),'runtime_log_sha256':sha(log),'runtime_records':observed_lines,
        'sdk_receipt_sha256':sha(receipt_file),'sdk_parsed_outputs':parsed_bindings,
        'raw_scalar_buffers':receipt['source_files'],'actual_binary_sha256':execution['compiler_binding']['binary_sha256'],
        'numeric_output_neutrality_accepted_here':False,'source_derived_observations':False,
        'limits':['Measured Softmax entry columnCount/actualColumnCount, not a device instruction or Reduce tree trace.',
                  'Static-graph printf is flushed asynchronously. Device buffers and own-process runtime text match in aggregate; individual records are not assigned to a replay.',
                  'Only seven preregistered query positions of request0/head0 sampled; complete numeric outputs checked separately.',
                  'Same N also participates in BMM1/Softmax/BMM2; no isolated contribution of a single Reduce instruction.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--execution',type=Path,required=True);p.add_argument('--sdk-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:report=audit(a.case_dir,json.loads(a.execution.read_bytes()),a.sdk_receipt);code=0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as error:
        report={'status':'FAIL_DEVICE_SCALAR_EVIDENCE','error':str(error)};code=1
    report['analysis_source_sha256']=sha(Path(__file__))
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','case_id','canonical_scalar_records','error')}))
    raise SystemExit(code)
