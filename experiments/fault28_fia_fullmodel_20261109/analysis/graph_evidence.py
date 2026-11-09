"""Independent statistics localization. Never reads host events, policy or result.

Reads original FX/GE graphs, native GE CSV and native statistics/mix triplets.
The reported first difference is an observed statistic, not tensor equality or
a unique physical root cause. Missing/empty/null fields stay unobservable.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from functools import lru_cache
from statistics_values import load_step, compare_values, inputs, signature, is_tensor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ops(text):
    # GE protobuf text: nested braces and escaped string braces are distinct.
    result=[]
    for match in re.finditer(r'^op \{',text,re.M):
        start=match.end(); depth=1; quoted=False; escaped=False; pos=start
        while depth and pos<len(text):
            c=text[pos]
            if quoted:
                if escaped: escaped=False
                elif c=='\\': escaped=True
                elif c=='"': quoted=False
            elif c=='"': quoted=True
            elif c=='{': depth+=1
            elif c=='}': depth-=1
            pos+=1
        if depth: raise ValueError('Unterminated GE op')
        block=text[start:pos-1]
        name=re.search(r'^  name: "([^"\n]+)"$',block,re.M)
        kind=re.search(r'^  type: "([^"\n]+)"$',block,re.M)
        if not name or not kind: raise ValueError('Missing GE op identity')
        edges=re.findall(r'^  input: "([^"\n]*)"$',block,re.M)
        result.append({'name':name[1],'type':kind[1],'inputs':edges})
    if not result or len({x['name'] for x in result})!=len(result):
        raise ValueError('Empty/duplicate graph op identity')
    known={x['name'] for x in result}
    done=set(); ordered=[]
    while len(done)<len(result):
        ready=[n for n in result if n['name'] not in done and
               all(e.rsplit(':',1)[0] in done for e in n['inputs'] if e and not e.endswith(':-1'))]
        if not ready: raise ValueError('Missing or cyclic graph dependencies')
        for n in ready:
            if any(e.rsplit(':',1)[0] not in known for e in n['inputs'] if e):
                raise ValueError('Unmapped GE edge')
            ordered.append(n);done.add(n['name'])
    return ordered


@lru_cache(maxsize=32)
def graph(folder):
    matches=list((folder/'msprobe_graph').rglob('dynamo_optimized_graph_*.txt'))
    if len(matches)!=1: raise ValueError('Exactly one actual optimized GE graph required')
    path=matches[0]
    return ops(path.read_text()), {'path':str(path.relative_to(folder)),'sha256':sha(path)}


def csv_rows(path):
    result={}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            row={k.replace(' ',''):v for k,v in row.items()}
            identity=(row['Input/Output'],int(row['Index']))
            if identity in result: raise ValueError('Duplicate GE statistic row')
            result[identity]=row
    return result


@lru_cache(maxsize=32)
def csv_inventory(folder):
    # Final evidence is immutable. Inventory once, not once per model node.
    return tuple((folder/'msprobe_graph').rglob('*.csv'))


def observed_rows(folder, node, call):
    prefix=node['type']+'.'+node['name']+'.'
    files=[p for p in csv_inventory(folder)
           if p.name.startswith(prefix) and p.parent.name==str(call)
           and p.parent.parent.parent.name=='graph_1']
    if len(files)>1:
        # GE records a shared Data tensor at multiple consumers (e.g. tied
        # embedding/lm_head). Preserve every original. Collapse observations
        # only within the SAME execution group and with IDENTICAL full rows.
        if len({p.parent for p in files})!=1:
            raise ValueError('Ambiguous graph CSV execution groups')
        observations=[csv_rows(p) for p in files]
        if any(row!=observations[0] for row in observations[1:]):
            raise ValueError('Repeated graph CSV observations disagree; temporal alignment required')
        return observations[0],{'observations':[{'path':str(p.relative_to(folder)),'sha256':sha(p)} for p in files],
            'equivalent_repeated_observations':True,
            'scope':'Identical complete CSV rows in one graph/model/iteration; not tensor equality or dropped originals.'}
    return (csv_rows(files[0]), {'path':str(files[0].relative_to(folder)),'sha256':sha(files[0])}) if files else ({},None)


def compare_row(a,b):
    meta=('DataSize','DataType','Format','Shape','Count')
    difference=[];unobservable=[]
    metadata=[{'field':k,'B':a.get(k),'F':b.get(k)} for k in meta if a.get(k)!=b.get(k)]
    # CANN can leave stale summary values on size-zero optional outputs.
    if int(a['DataSize'])==0 or int(b['DataSize'])==0:
        return {'metadata':metadata,'differences':[], 'unobservable':[{'reason':'empty tensor; all statistics ignored'}]}
    for field in ('MaxValue','MinValue','AvgValue','NanCount','NegativeInfCount','PositiveInfCount'):
        pair=[]
        for val in (a.get(field),b.get(field)):
            try: n=float(val)
            except (ValueError,TypeError): n=None
            pair.append(n if n is not None and math.isfinite(n) else None)
        item={'field':field,'B':a.get(field),'F':b.get(field)}
        if None in pair:unobservable.append(item)
        elif pair[0]!=pair[1]:difference.append(item)
    return {'metadata':metadata,'differences':difference,'unobservable':unobservable}


def ge_compare(b,f,call):
    bn,bhash=graph(b);fn,fhash=graph(f)
    if bn!=fn: raise ValueError('Graph topology differs; no aligned first difference')
    rows=[];first=None;missing_before=[];sources=[]
    for node in bn:
        if node['type'] in ('Const','NetOutput'):continue
        left,lp=observed_rows(b,node,call);right,rp=observed_rows(f,node,call)
        sources.append({'node':node['name'],'B':lp,'F':rp})
        if not left or not right:
            if first is None:missing_before.append(node['name'])
            rows.append({'node':node,'unobservable':'No unique native CSV; possible GE fusion needs mapping'})
            continue
        if left.keys()!=right.keys():raise ValueError('GE row boundaries differ')
        # Data nodes only have output. Other op inputs precede its outputs.
        for boundary in sorted(left,key=lambda k:(k[0]!='Input',k[1])):
            item={'node':node['name'],'type':node['type'],'boundary':boundary[0],'index':boundary[1],
                  **compare_row(left[boundary],right[boundary])}
            rows.append(item)
            if item['metadata']: raise ValueError('GE tensor metadata differs')
            if first is None and item['differences']:first=item
    return {'graph':{'B':bhash,'F':fhash},'sources':sources,'call':call,'first_observed_difference':first,
            'missing_nodes_before_first':missing_before,'comparisons':rows,
            'status':'FIRST_OBSERVED_WITH_GAPS' if missing_before else 'ALIGNED_FIRST_OBSERVED' if first else 'NO_OBSERVED_DIFFERENCE'}


def mix_compare(b,f,step):
    left=load_step(b/'observation'/f'step{step}'/'rank0/dump.json')
    right=load_step(f/'observation'/f'step{step}'/'rank0/dump.json')
    if left['config']!=right['config'] or left['collector']!=right['collector']:
        raise ValueError('Statistics config or collector differs')
    if [signature(e) for e in left['entries']] != [signature(e) for e in right['entries']]:
        raise ValueError('Source stack/construct trace differs')
    comparisons=[];first=None
    for a,c in zip(left['entries'],right['entries']):
        row={'B':a['name'],'F':c['name'],'input':compare_values(inputs(a['node']),inputs(c['node'])),
             'output':compare_values(a['node'].get('output'),c['node'].get('output'))}
        for side in ('input','output'):
            if row[side]['metadata_differences']:raise ValueError('Mix metadata differs')
            if first is None and row[side]['statistical_differences']:
                first={'node':a['name'],'boundary':side,'differences':row[side]['statistical_differences']}
        comparisons.append(row)
    return {'step':step,'evidence_sha256':{'B':left['hashes'],'F':right['hashes']},
            'first_observed_difference':first,'comparisons':comparisons,
            'status':'ALIGNED_MODULE_OBSERVATION' if first else 'NO_OBSERVED_DIFFERENCE'}


def output_tensor_paths(value,path='$'):
    """Actual serialized tensor leaves, preserving list/tuple wrapper levels."""
    if is_tensor(value):return [path]
    if isinstance(value,list):
        return [leaf for i,item in enumerate(value) for leaf in output_tensor_paths(item,path+f'[{i}]')]
    raise ValueError('Unmapped non-tensor graph output container')


def alias_mapping(folder):
    # A module output statistic can be mapped to an op only through graph edges.
    nodes,meta=graph(folder)
    output=[n for n in nodes if n['type']=='NetOutput']
    if len(output)!=1:raise ValueError('Ambiguous graph output')
    edges=[e for e in output[0]['inputs'] if e and not e.endswith(':-1')]
    table={n['name']:n for n in nodes}
    mapped=[]
    for slot,edge in enumerate(edges):
        producer,index=edge.rsplit(':',1)
        mapped.append({'module_output_slot':slot,'producer':producer,'producer_type':table[producer]['type'],
                       'producer_output_index':int(index),'edge':edge})
    fx_path=folder/'fx_graph_0.json'
    fx=json.loads(fx_path.read_bytes())
    raw=load_step(folder/'observation/step0/rank0/dump.json')
    boundaries=[e for e in raw['entries'] if e['name'].startswith('Module.') and 'GraphBoundary' in e['name']]
    if len(boundaries)!=1:raise ValueError('Need the actual single compiled GraphBoundary observation')
    paths=output_tensor_paths(boundaries[0]['node']['output'])
    if len(paths)!=len(mapped):raise ValueError('Serialized output leaves and actual GE output arity differ')
    for item,path in zip(mapped,paths):item['serialized_output_path']=path
    if 'output_nodes' in fx and len(fx['output_nodes'])!=len(mapped):
        raise ValueError('Actual FX/GE output arity differs')
    return {'ge':meta,'fx_sha256':sha(fx_path),'fx_code':fx['code'],'output_edges':mapped,
            'scope':'GraphBoundary returns compiled result unchanged; GE NetOutput edges establish producer aliases.'}


def infer_first(nodes,ge,mix,aliases):
    """Join observed fields through real graph edges, without a target name.

    Module tuple slot order is not execution order. Producer topology orders
    candidates. An earlier observed Data/input difference takes precedence.
    This is an observation bound, never a unique physical-fault assertion.
    """
    order={n['name']:i for i,n in enumerate(nodes)}
    table={n['name']:n for n in nodes}
    candidates=[]
    first=ge['first_observed_difference']
    if first:
        candidates.append({'node':first['node'],'type':first['type'],'boundary':first['boundary'].lower(),
                           'index':first['index'],'fields':first['differences'],'source':'native_msprobe_GE_CSV'})
    first=mix['first_observed_difference']
    if first and first['boundary']=='input':
        return {'status':'OBSERVED_GRAPH_BOUNDARY_INPUT_DIFFERENCE','first':first}
    if first and first['boundary']=='output':
        edges={e['module_output_slot']:e for e in aliases['output_edges']}
        paths={e.get('serialized_output_path',f'$[{slot}]'):slot for slot,e in edges.items()}
        byslot={}
        for difference in first['differences']:
            path,field=difference['path'].rsplit('.',1)
            if path not in paths or field not in ('Max','Min','Mean','Norm'):
                raise ValueError('Unmapped module output statistic')
            slot=paths[path]
            if slot not in edges:raise ValueError('Missing graph output alias')
            byslot.setdefault(slot,[]).append(difference)
        for slot,fields in byslot.items():
            edge=edges[slot];node=edge['producer']
            if node not in table:raise ValueError('Unknown direct output producer')
            candidates.append({'node':node,'type':edge['producer_type'],'boundary':'output',
                               'index':edge['producer_output_index'],'fields':fields,
                               'source':'native_msprobe_statistics_mix_via_direct_GE_NetOutput_alias',
                               'raw_module_node':first['node'],'raw_module_output_slot':slot,'graph_edge':edge['edge']})
    candidates.sort(key=lambda c:(order[c['node']],c['boundary']!='input',c['index']))
    return {'status':'ALIGNED_OBSERVED_FIRST_DIFFERENCE' if candidates else 'NO_OBSERVED_FINITE_DIFFERENCE',
            'first':candidates[0] if candidates else None,'ordered_candidates':candidates,
            'scope':'First observed producer in this actual graph, joining native GE rows and high precision module output aliases. Summary equality is not tensor equality.'}


def compare(b,f):
    ge=[ge_compare(b,f,i) for i in (0,1,2)]
    mix=[mix_compare(b,f,i) for i in (0,1)]
    aliases={'B':alias_mapping(b),'F':alias_mapping(f)}
    if aliases['B']['output_edges']!=aliases['F']['output_edges']:
        raise ValueError('Output producer aliases differ')
    nodes,_=graph(b)
    return {'schema':'fault28.independent_graph_statistics.v1','status':'ANALYZED','ge':ge,
            'statistics_mix':mix,'output_aliases':aliases,
            'independent_first_producer':[infer_first(nodes,ge[i+1],mix[i],aliases['B']) for i in (0,1)],
            'tensor_equality_proven':False,'physical_fault_location_unique':False,
            'limits':['Finite summaries can coincide for different tensors.',
                      'Null, missing, nonfinite and empty outputs remain unobservable.',
                      'Cross-batch aggregate statistics are not a same-query comparison.',
                      'High precision module outputs are mapped through actual GE NetOutput edges; they are not relabeled eager NPU API records.',
                      'No host events, expected profiles, mode or injector hit records were read.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--fault',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:
        report=compare(a.baseline,a.fault)
        rc=0
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:
        report={'schema':'fault28.independent_graph_statistics.v1','status':'INCONCLUSIVE_INVALID_OR_UNALIGNED',
                'error':str(e)};rc=1
    with a.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'error':report.get('error'),'output':str(a.output)}))
    return rc


if __name__=='__main__':raise SystemExit(main())
