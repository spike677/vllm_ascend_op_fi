"""Independent native API first difference; reads no injector/runtime result."""
import argparse
import json
from pathlib import Path
from statistics_values import load_step,signature,compare_values,inputs


def compare(baseline,fault,rtol=1e-5,atol=1e-8):
    arms={'baseline':load_step(baseline),'fault':load_step(fault)}
    b,f=arms.values()
    trace=[signature(x) for x in b['entries']]==[signature(x) for x in f['entries']]
    config=b['config']==f['config'] and b['collector']==f['collector']
    rows=[]
    if trace and config:
        for left,right in zip(b['entries'],f['entries']):
            rows.append({'B':left['name'],'F':right['name'],'parents':left['parents'],
                'input':compare_values(inputs(left['node']),inputs(right['node']),rtol,atol),
                'output':compare_values(left['node'].get('output'),right['node'].get('output'),rtol,atol)})
    first=None;first_exact=None
    metadata=sum(len(r[s]['metadata_differences']) for r in rows for s in ('input','output'))
    if not trace or not config:status='INCONCLUSIVE_TRACE_OR_CONFIG'
    elif metadata:status='INCONCLUSIVE_NODE_METADATA'
    else:
        # Module entries describe the containing function's completed return,
        # not its first executed leaf API. Preserve their differences separately.
        for row in rows:
            if row['B'].startswith('Module.'):continue
            for side in ('input','output'):
                if first_exact is None and row[side]['statistical_differences']:
                    first_exact={'B_node':row['B'],'F_node':row['F'],'boundary':side,
                                 'field_differences':row[side]['statistical_differences']}
                if first is None and row[side]['tolerance_differences']:
                    first={'B_node':row['B'],'F_node':row['F'],'boundary':side,
                        'field_differences':row[side]['tolerance_differences']}
        status='ALIGNED_ABOVE_TOLERANCE_API_DIFFERENCE' if first else 'ALIGNED_NO_ABOVE_TOLERANCE_API_DIFFERENCE'
    return {'schema':'fault28.visible.independent_msprobe.v1','status':status,
        'comparison':{'rtol':rtol,'atol':atol,'reference':'baseline',
                      'formula':'abs(F-B) > atol + rtol*abs(B)',
                      'relative_zero_baseline':None,'epsilon_substitution':False},
        'alignment':{'ordered_source_stack_and_construct_match':trace,'config_collector_match':config},
        'evidence':{k:{'step':v['step'],'sha256':v['hashes'],'collector':v['collector']} for k,v in arms.items()},
        'first_observed_api_difference':first,'first_exact_api_difference':first_exact,
        'all_aligned_records':len(rows),
        'native_API_records':sum(not r['B'].startswith('Module.') for r in rows),
        'unobservable_fields':sum(len(r[s]['unobservable']) for r in rows for s in ('input','output')),
        'comparisons':rows,'tensor_equality_proven':False,'physical_fault_location_proven':False,
        'limits':['First is the first finite above-tolerance difference in aligned raw leaf API order, inputs before outputs.',
            'No expected operator, host policy/hit, model result or output checksum selects the first difference.',
            'Module returns are retained but do not establish leaf execution order.',
            'Null/missing/nonfinite statistics stay unobservable. Equal summaries do not prove tensor equality.',
            'All-request statistics alone cannot prove a difference for each individual request.',
            'Native strategy/launch, complete QKV equality and model propagation need separate runtime evidence.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--fault',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rtol',type=float,default=1e-5);p.add_argument('--atol',type=float,default=1e-8)
    a=p.parse_args()
    try:r=compare(a.baseline,a.fault,a.rtol,a.atol);rc=2 if r['status'].startswith('INCONCLUSIVE') else 0
    except (ValueError,OSError,TypeError,KeyError,IndexError) as e:
        r={'schema':'fault28.eager.independent_msprobe.v1','status':'INVALID_CAPTURE_EVIDENCE','error':str(e)};rc=1
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as s:json.dump(r,s,indent=2,allow_nan=False);s.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('comparisons','evidence','limits')}))
    return rc


if __name__=='__main__':raise SystemExit(main())
