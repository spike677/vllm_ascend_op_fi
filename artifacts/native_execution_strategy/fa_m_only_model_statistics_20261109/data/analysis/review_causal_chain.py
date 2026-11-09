"""Keep strategy coverage, numeric propagation and source-only msprobe separate.

Summarize already accepted original evidence; do not select a desired first
layer, require a numeric effect, or replace null/negative outcomes with success.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();review=a.review
    execution=read(review/'execution_acceptance.json')
    assert execution['status']=='PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE'
    suite=read(review/'suite.json');cases=[];bindings={}
    for spec in suite['cases']:
        folder=Path(spec['case_dir']);suffix='b1' if folder.name.endswith('b1') else 'b8'
        # Read statistical inference before joining numerical or host outcomes.
        stats_file=Path(spec['statistics_report']);statistics=read(stats_file)
        firsts=[{'comparison_id':r['comparison_id'],'first':r['first_observed_model_api_difference'],
                 'alignment_complete':r['alignment_complete'],
                 'unobservable_fields':{s['stage']:s['unobservable_fields'] for s in r['phases']}}
                for r in statistics['runs']]
        numeric=read(review/f'numeric_{suffix}.json');logits=read(review/f'logits_{suffix}.json')
        assert numeric['status']=='PASS_EXACT_OBSERVATION_INTEGRITY' and numeric['repeat_and_off_exact']
        assert logits['status']=='PASS_COMPLETE_LOGITS_BYTES_REVIEW'
        for report in (numeric,logits):
            assert report['case_result_sha256']==sha(folder/'result.json')
            assert report['phase_manifest_sha256']==sha(folder/'phase_manifest.json')
        actual=next(c for c in execution['cases'] if c['case_id']==folder.name)
        runtime=[r for r in actual['runtime_numerics']['comparisons'] if r['comparison_id']=='native_vs_fault_a'][0]
        layers=numeric['phases'][4]['layers']
        first_attention=next((x['layer_index'] for x in layers if x['attention']['changed_elements']),None)
        first_last=next((x['layer_index'] for x in layers if x['attention']['last_query_changed_elements']),None)
        first_cache=next((x['layer_index'] for x in layers if x['cache_keys']['changed_elements'] or x['cache_values']['changed_elements']),None)
        changed=any(row['logits']['changed_elements'] for row in numeric['fault_phases'][:2])
        category=('M_ONLY_FULL_MODEL_LOGITS_PROPAGATION_OBSERVED' if changed else
                  'INTERMEDIATE_CHANGE_WITHOUT_LOGITS_CHANGE' if any(x['attention_changed_layers'] for x in numeric['fault_phases']) else
                  'STRATEGY_SWITCH_WITH_NO_OBSERVED_NUMERIC_CHANGE')
        rows=[r for r in logits['comparisons'] if r['comparison'].startswith('native_fault_a')]
        cases.append({'case_id':folder.name,'classification':category,'actual_strategy_switches':actual['actual_strategy_switches'],
                      'first_prefill_attention_changed_layer':first_attention,
                      'first_prefill_last_query_changed_layer':first_last,'first_prefill_cache_changed_layer':first_cache,
                      'prefill_layer0':layers[0], 'prefill_layer1':layers[1],
                      'fault_phases':numeric['fault_phases'],'offline_complete_logits':rows,
                      'runtime_first_differences':[{k:s[k] for k in ('stage','first_runtime_hash_difference')} for s in runtime['stages']],
                      'independent_msprobe_runs':firsts,'repeat_and_off_exact':True})
        for path in (stats_file,review/f'numeric_{suffix}.json',review/f'logits_{suffix}.json'):
            bindings[str(path)]=sha(path)
    report={'schema':'fault28.m_only.full_model_causal_review.v1','status':'REVIEW_COMPLETED',
            'strategy_coverage_accepted':True,'cases':cases,'evidence_sha256':bindings,
            'execution_acceptance_sha256':sha(review/'execution_acceptance.json'),
            'analysis_source_sha256':sha(Path(__file__)),
            'historical_fia_graph_defect_reproduced':False,'model_accuracy_loss_proven':False,
            'limits':['This is the unchanged native FA adapter in a full HF Qwen3 model, not historical Pangu FIA graph execution.',
                      'Every eligible prefill layer is switched; downstream changes include propagation and further legal M-only changes.',
                      'This experiment does not isolate layer 0 as the sole injected layer, or identify a unique kernel Reduce instruction.',
                      'msprobe statistics cannot establish tensor equality or identify request-level changes; separate exact observations do that.',
                      'No claims beyond the fixed version, shape, requests and one teacher-forced decode token.']}
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'cases':[{k:r[k] for k in ('case_id','classification')} for r in cases]}))


if __name__=='__main__':main()
