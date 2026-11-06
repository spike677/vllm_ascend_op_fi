"""Final fixed-matrix acceptance; numerical and localization evidence stay separate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_case import audit,ROOT
from compare_targets import load,difference,reference,ref_errors
import graph_evidence


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def require(ok,text):
    if not ok:raise ValueError(text)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def first_fia(report):
    rows=report['independent_first_producer']
    return len(rows)==2 and all(r['status']=='ALIGNED_OBSERVED_FIRST_DIFFERENCE' and
        r['first']['type']=='FusedInferAttentionScore' and r['first']['boundary']=='output' and
        r['first']['index']==0 for r in rows)


def accept(suffix,out,collection='fp32'):
    matrix_file=ROOT/('setup/fp64_matrix_v1.json' if collection=='fp64' else 'setup/formal_matrix_v3.json');matrix=json.loads(matrix_file.read_bytes())
    out.mkdir(parents=True,exist_ok=False)
    folders={n:ROOT/'outputs'/(n+'_'+suffix)/n for n in matrix['case_ids']}
    audits={};values={};errors={};ref=reference(Path('/home/fault28-causal-20261029/inputs'))
    for name,folder in folders.items():
        record=audit(name+'_'+suffix);audits[name]=record
        save(out/(name+'_audit.json'),record)
        value,meta=load(folder);values[name]=value
        selected_errors=ref_errors(value,ref);errors[name]=selected_errors
        require(all(r['max_abs']<=matrix['reference_sanity']['max_abs_limit'] and
                    r['relative_l2']<=matrix['reference_sanity']['relative_l2_limit'] for r in selected_errors),
                'Selected reference sanity failed: '+name)
        # Device full/per-request hashes were taken after synchronization; all
        # complete private target elements are independently re-read here.
        result=json.loads((folder/'result.json').read_bytes())
        for call in (0,1,2):
            v,_=load(folder,call);p,_=load(folder,call,True)
            require(np.array_equal(value,v),'Unstable complete attention: '+name)
            if call:
                p0,_=load(folder,0,True)
                require(np.array_equal(p,p0),'Unstable complete projection: '+name)
        require(all(len(c['request_sha256'])==record['batch'] for c in result['calls']), 'Incomplete request census')
    pairs={};localizations={}
    def pair(label,left,right,localize=False):
        b,bm=load(folders[left]);f,fm=load(folders[right])
        bp,bpm=load(folders[left],consumer=True);fp,fpm=load(folders[right],consumer=True)
        report={'schema':'fault28.complete_pair.v1','left':left,'right':right,
                'attention':{'B':bm,'F':fm,**difference(b,f)},
                'projection':{'B':bpm,'F':fpm,**difference(bp,fp)}}
        save(out/(label+'_complete.json'),report)
        if localize:
            report_stats=graph_evidence.compare(folders[left],folders[right])
            save(out/(label+'_msprobe.json'),report_stats);localizations[label]=report_stats
        pairs[label]={'left':left,'right':right,'attention_changed':report['attention']['changed_elements'],
                      'attention_max_abs':report['attention']['max_abs'],
                      'last_query_changed':report['attention']['last_query_changed_elements'],
                      'projection_changed':report['projection']['changed_elements'],
                      'first_fia_both_steps':first_fia(localizations[label]) if localize else None,
                      'complete_report':label+'_complete.json','statistics_report':label+'_msprobe.json' if localize else None}
        return report
    # Original/current stock OFF compatibility, all independent process repeats.
    for repeat in (1,2):
        for batch in (1,8):
            r=pair(f'bridge_b{batch}_r{repeat}',f'g_stock_b{batch}_r{repeat}',f'g_off_b{batch}_r{repeat}',True)
            require(r['attention']['B']['sha256']==r['attention']['F']['sha256'] and
                    r['projection']['B']['sha256']==r['projection']['F']['sha256'],'Stock/OFF bridge not exact')
    for profile in ('off','m128','m256','causal_base512','causal_m128','causal_m256'):
        r=pair('fixed_'+profile+'_cross_batch',f'g_{profile}_b1_r1',f'g_{profile}_b8_r1')
        require(r['attention']['B']['sha256']==r['attention']['F']['sha256'], 'Fixed profile has cross-batch attention confound: '+profile)
    for batch in (1,8):
        r=pair(f'base512_b{batch}',f'g_off_b{batch}_r1',f'g_causal_base512_b{batch}_r1',True)
        require(r['attention']['B']['sha256']==r['attention']['F']['sha256'] and
                r['projection']['B']['sha256']==r['projection']['F']['sha256'],'M512 variant not exact')
        for m in (128,256):
            pair(f'm_only{m}_b{batch}',f'g_off_b{batch}_r1',f'g_m{m}_b{batch}_r1',True)
            r=pair(f'causal_width{m}_b{batch}',f'g_m{m}_b{batch}_r1',f'g_causal_m{m}_b{batch}_r1',True)
            # Same public tiling, same M and Brcb flag; only the independently
            # compiled causal kernel variant differs.
            a=audits[f'g_m{m}_b{batch}_r1']['public_host'][0]
            b=audits[f'g_causal_m{m}_b{batch}_r1']['public_host'][0]
            require(a['raw_sha256']==b['raw_sha256'],'Same-M factorial has other raw tiling changes')
            require(r['attention']['changed_elements']>0,'No isolated causal-width numerical effect')
            require(first_fia(localizations[f'causal_width{m}_b{batch}']),'Independent same-M first producer is not FIA output')
    # Automatic policy: actual B1 selects512, B8 selects128 in the same binary.
    for repeat in (1,2):
        for batch in (1,8):
            n=f'g_batch_b{batch}_r{repeat}';row=audits[n]['public_host'][0]
            require(row['plan']['m']==(512 if batch==1 else 128) and row['plan']['pruned'], 'Batch selector did not use actual requested B')
            require(np.array_equal(values[n],values[f'g_causal_{"base512" if batch==1 else "m128"}_b{batch}_r1']),
                    'Automatic selector differs from corresponding fixed strategy')
        pair('automatic_cross_batch_r'+str(repeat),f'g_batch_b1_r{repeat}',f'g_batch_b8_r{repeat}')
        pair('automatic_same_batch_r'+str(repeat),'g_off_b8_r1',f'g_batch_b8_r{repeat}',True)
        require(pairs['automatic_same_batch_r'+str(repeat)]['attention_changed']>0 and
                first_fia(localizations['automatic_same_batch_r'+str(repeat)]),'Automatic policy lacks independent FIA first difference')
    r=pair('automatic_repeat','g_batch_b8_r1','g_batch_b8_r2',True)
    require(r['attention']['B']['sha256']==r['attention']['F']['sha256'] and
            r['projection']['B']['sha256']==r['projection']['F']['sha256'],'Independent F/F repeat differs')
    r=pair('restore','g_off_b8_r1','g_restore_b8_r1',True)
    require(r['attention']['B']['sha256']==r['attention']['F']['sha256'] and
            r['projection']['B']['sha256']==r['projection']['F']['sha256'],'Fresh-compiled OFF did not restore')
    for label in ('restore','automatic_repeat'):
        require(all(x['first_observed_difference'] is None for x in
                    localizations[label]['ge']+localizations[label]['statistics_mix']), 'Restored/repeated statistics differ')
    rows=[]
    for name,a in audits.items():
        host=a['public_host'][0] if a['public_host'] else None
        rows.append({'case_id':name,'batch':a['batch'],'mode':a['mode'],
                     'actual_key':a['native']['selected']['jit_template_key'],
                     'M':host['plan']['m'] if host else 512,'raw_and_compiler_verified':bool(host),
                     'dynamic_inner_widths':list(host['plan']['per_query_inner_width_histogram']) if host else ['512'],
                     'complete_target_sha256':load(folders[name])[1]['sha256'],
                     'graph_calls':len(a['native']['graph_calls'])})
    result={'schema':'fault28.fia_graph.final_acceptance.v1','status':'PASS_CONTROLLED_FIA_GRAPH_CAUSAL_MECHANISM',
            'matrix_sha256':sha(matrix_file),'cases':len(rows),'actual_graph_calls':sum(r['graph_calls'] for r in rows),
            'static_fia_tasks_instantiated':len(rows),'private_public_raw_and_compiler_verified_cases':sum(bool(r['raw_and_compiler_verified']) for r in rows),
            'complete_target_elements_per_call':16*2048*128,'original_model_query_layer':1,
            'pairs':pairs,'case_summary':rows,'selected_fp64_reference_errors':errors,
            'evidence_files':{p.name:sha(p) for p in sorted(out.glob('*.json'))},
            'limits':['Current CANN9.1 controlled host+kernel causal variant on real Qwen3 QKV/o_proj subgraph; not the historical Pangu defect.',
                      'Not complete-model graph/logits or all-layer FIA coverage acceptance.',
                      'Native GE CSV can show its first difference at downstream MatMul due displayed precision; original reports preserve this.',
                      'Task-local high precision statistics/mix graph outputs are mapped through direct GE NetOutput aliases, not relabeled as eager NPU API dumps.',
                      'Dynamic N widths are source-derived from verified actual key/raw and kernel implementation, not measured device trace.',
                      'N width affects BMM1 N, softmax Reduce and BMM2 K; no unique instruction-level contribution attribution.',
                      'Public host raw is identical to compiler-consumed raw; kernel argument bytes were not directly observed.',
                      'Fixed query reference checks cover selected positions, full output comparisons cover all query elements.']}
    save(out/'acceptance.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suffix',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--collection',choices=('fp32','fp64'),default='fp32');a=p.parse_args()
    try:
        report=accept(a.suffix,a.output,a.collection);print(json.dumps({k:report[k] for k in ('status','cases','actual_graph_calls')}));rc=0
    except (ValueError,AssertionError,OSError,KeyError,TypeError,IndexError) as e:
        report={'status':'FAIL_ACCEPTANCE','error':str(e)}
        if a.output.is_dir() and not (a.output/'failure.json').exists():save(a.output/'failure.json',report)
        print(json.dumps(report));rc=1
    raise SystemExit(rc)
