"""Join independently inferred FIA first difference with measured device scalars."""
import argparse
import hashlib
import json
from pathlib import Path
from audit_case import ROOT
from compare_targets import load,difference
import graph_evidence
from accept_matrix import first_fia


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,value):
    with p.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--device',type=Path,required=True);p.add_argument('--neutrality',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--suffix',default='grain_a2');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    acceptance=json.loads((a.reports/'acceptance.json').read_bytes())
    assert acceptance['status']=='PASS_CONTROLLED_FIA_GRAPH_CAUSAL_MECHANISM'
    device=json.loads((a.device/'summary.json').read_bytes())
    assert device['status']=='PASS_MEASURED_FIA_SOFTMAX_COLUMN_MATRIX' and device['measured_private_cases']==19
    neutrality=json.loads(a.neutrality.read_bytes())
    assert neutrality['status']=='PASS_ALL_NUMERICAL_OUTPUTS_UNCHANGED' and neutrality['paired_graph_calls']==69
    indexed={r['case_id']:r for r in device['rows']};matrix={r['case_id']:r for r in acceptance['case_summary']}
    def folder(name):return ROOT/'outputs'/(name+'_'+a.suffix)/name
    def record(name):
        row=indexed[name];file=a.device/row['report'];assert sha(file)==row['report_sha256']
        return json.loads(file.read_bytes())
    def actual_batch(name):
        audit=json.loads((a.reports/(name+'_audit.json')).read_bytes())
        return audit['public_host'][0]['decoded']['baseParams.bSize']
    baseline='g_causal_base512_b8_r1';faults=('g_batch_b8_r1','g_batch_b8_r2');primary=[]
    for fault in faults:
        # Freeze the source-only inference before opening host or scalar reports.
        observed=graph_evidence.compare(folder(baseline),folder(fault))
        name=f'primary_{fault}_msprobe.json';save(a.output/name,observed)
        assert first_fia(observed),'Independent observed first producer is not FIA output0'
        b=record(baseline);f=record(fault)
        assert b['actual_causal_variant'] and f['actual_causal_variant']
        assert (b['actual_m'],f['actual_m'])==(512,128)
        assert actual_batch(baseline)==actual_batch(fault)==8
        inputs=[json.loads((folder(n)/'inputs.json').read_bytes()) for n in (baseline,fault)]
        assert inputs[0]==inputs[1],'Within-B8 actual QKV/mask inputs differ'
        results=[json.loads((folder(n)/'result.json').read_bytes()) for n in (baseline,fault)]
        request_differences=[]
        for call in (0,1,2):
            b_value,b_meta=load(folder(baseline),call);f_value,f_meta=load(folder(fault),call)
            assert b_meta['sha256']!=f_meta['sha256']
            before,after=[r['calls'][call] for r in results]
            changed=[i for i,(x,y) in enumerate(zip(before['request_sha256'],after['request_sha256'])) if x!=y]
            assert len(before['request_sha256'])==len(after['request_sha256'])==8
            request_differences.append({'call_id':call,'changed_request_slots':changed})
        b_value,b_meta=load(folder(baseline));f_value,f_meta=load(folder(fault))
        bp,bpm=load(folder(baseline),consumer=True);fp,fpm=load(folder(fault),consumer=True)
        full={'baseline':baseline,'fault':fault,'attention':difference(b_value,f_value),
              'projection':difference(bp,fp),'B_attention':b_meta,'F_attention':f_meta}
        assert full['attention']['changed_elements']>0
        complete_file=f'primary_{fault}_complete.json';save(a.output/complete_file,full)
        primary.append({'baseline':baseline,'fault':fault,'input_file_sha256':sha(folder(baseline)/'inputs.json'),
            'same_causal_kernel_rule':True,'B_actual_m':512,'F_actual_m':128,
            'B_query63_columns':next(x['measured_columns'] for x in b['query_tiles'] if x['query']==63),
            'F_query63_columns':next(x['measured_columns'] for x in f['query_tiles'] if x['query']==63),
            'attention_changed_elements':full['attention']['changed_elements'],
            'attention_max_abs':full['attention']['max_abs'],'last_query_changed':full['attention']['last_query_changed_elements'],
            'projection_changed_elements':full['projection']['changed_elements'],
            'all_request_hash_differences':request_differences,
            'independent_first_report':name,'independent_first_report_sha256':sha(a.output/name),
            'complete_report':complete_file,'complete_report_sha256':sha(a.output/complete_file)})
    auto=[]
    for repeat in (1,2):
        one=f'g_batch_b1_r{repeat}';eight=f'g_batch_b8_r{repeat}'
        b=record(one);f=record(eight)
        x=json.loads((folder(one)/'inputs.json').read_bytes());y=json.loads((folder(eight)/'inputs.json').read_bytes())
        assert all(x['original_bnsd'][kind]['target_sha256']==y['original_bnsd'][kind]['target_sha256'] for kind in ('query','key','value'))
        assert (actual_batch(one),actual_batch(eight),b['actual_m'],f['actual_m'])==(1,8,512,128)
        auto.append({'repeat':repeat,'B1_actual_batch':actual_batch(one),'B8_actual_batch':actual_batch(eight),
                     'B1_actual_M':b['actual_m'],'B8_actual_M':f['actual_m'],
                     'same_complete_target_QKV':True,'B1_query63_columns':indexed[one]['query63_columns'],
                     'B8_query63_columns':indexed[eight]['query63_columns']})
    # Full logical hashes of all requests complement the private complete target
    # readback. They are equality evidence, not per-request error magnitudes.
    for left,right,label in (('g_batch_b8_r1','g_batch_b8_r2','repeat'),
                             (baseline,'g_restore_b8_r1','restore')):
        x,y=[json.loads((folder(n)/'result.json').read_bytes()) for n in (left,right)]
        for call in (0,1,2):
            b,f=[r['calls'][call] for r in (x,y)]
            for field in ('full_sha256','request_sha256'):assert b[field]==f[field],label
            for field in ('full_sha256','request_sha256'):assert b['consumer'][field]==f['consumer'][field],label
    result={'schema':'fault28.fia.measured_causal_chain.v1','status':'PASS_CONTROLLED_FIA_GRAPH_MEASURED_CAUSAL_CHAIN',
        'preregistered_cases':23,'actual_graph_calls':69,'measured_private_cases':19,
        'device_scalar_records':device['scalar_records'],'primary_same_rule_BF':primary,'automatic_batch':auto,
        'all_request_attention_and_projection_repeat_off_hashes_equal':True,
        'execution_acceptance_sha256':sha(a.reports/'acceptance.json'),'device_matrix_sha256':sha(a.device/'summary.json'),
        'observer_neutrality_sha256':sha(a.neutrality),'analysis_source_sha256':sha(Path(__file__)),
        'individual_scalar_to_replay_pairing_proven':False,'historical_Pangu_version_reproduced':False,
        'scope':'Fixed actual Qwen3 layer1 QKV and real o_proj in a real TorchAir fullgraph FIA subgraph.',
        'limits':['Controlled current-version batch selector and legitimate causal variant; not an unmodified native default Batch switch.',
                  'Measured Softmax column granularity; not a Reduce instruction tree or unique instruction-level error contribution.',
                  'Seven query positions of request0/head0 observed; full numerical output checked separately.',
                  'Asynchronous scalar flush is verified in aggregate across each process; replay pairing not established.',
                  'statistics/mix GraphBoundary output0 maps through actual GE NetOutput to FIA0; not a relabeled eager API dump.',
                  'Not a complete Pangu/Qwen3 model graph, natural all-layer FIA path or first-generated-token logits reproduction.']}
    save(a.output/'acceptance.json',result)
    print(json.dumps({k:result[k] for k in ('status','preregistered_cases','actual_graph_calls','device_scalar_records')}))


if __name__=='__main__':main()
