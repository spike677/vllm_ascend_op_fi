"""CPU-only fixtures for the additional local formal review; no device imports."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

import review_formal_summary as review
import test_model_analysis as fixture_source
from audit_model_coverage import audit
from compare_model_statistics import analyze, compare_phase
from model_statistics_io import load_capture, sha

HERE = Path(__file__).resolve().parent


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return sha(path)


def snap(shape, label):
    return {'shape': shape, 'dtype': 'torch.float16',
        'full_sha256': fixture_source.h(label),
        'per_batch_sha256': [fixture_source.h(label + ':' + str(i)) for i in range(shape[0])]}


def enriched_fixture(root):
    case, completion, config = fixture_source.fixture(root, 'formal_b1', 1)
    (case / 'model_config.json').write_bytes(config.read_bytes())
    phases = json.loads((case / 'phase_manifest.json').read_bytes())
    refs = []
    for phase in phases['phases']:
        step, stage, run = phase['step'], phase['stage'], phase['run_index']
        fault = run in (2, 3)
        qlen, kvlen = (2048, 2048) if stage == 'prefill' else (1, 2049)
        phase['logits'] = snap([1, 1, 151936], stage + ':logits:' + str(fault))
        for layer in range(28):
            inputs = {key: snap([1, heads, length, 128], f'{stage}:{key}:{layer}:{fault and layer > 0}')
                for key, heads, length in [('query',16,qlen),('key',8,kvlen),('value',8,kvlen)]}
            record = {'status': 'CAPTURE_COMPLETED', 'step': step, 'layer_index': layer,
                'call_id': phase['sdpa_call_ids'][layer], 'module_call_id': f's{step}_layer{layer}',
                'request_ids': ['P0'], 'request_mapping_file': phase['request_mapping_file'],
                'request_mapping_sha256': phase['request_mapping_sha256'],
                'inputs_before': inputs, 'inputs_after': copy.deepcopy(inputs),
                'output': snap([1,16,qlen,128], f'{stage}:output:{layer}:{fault and (stage == "prefill" or layer > 0)}')}
            filename = f'native_calls/call_{step * 28 + layer:04d}/call.json'
            digest = write(case / filename, record)
            refs.append({'file':filename,'sha256':digest,'step':step,'layer_index':layer,'call_id':record['call_id']})
    # Uniform input data can have distinct tensors with the same four summaries.
    # Layer 10 retains a sealed output hash difference but not an output summary difference.
    receipt = json.loads((case / 'receipt.json').read_bytes())
    for step in (4,6):
        path = case / f'observation/step{step}/rank0/dump.json'
        dump = json.loads(path.read_bytes())
        dump['data']['Functional.scaled_dot_product_attention.10.forward']['output'][0]['Mean'] = 0.25
        digest = write(path,dump)
        for ref in receipt['native_files']:
            if ref['path'] == path.relative_to(case).as_posix():
                ref.update(sha256=digest,bytes=path.stat().st_size)
    result = json.loads((case / 'result.json').read_bytes())
    result.update(calls=refs,original_api_calls=280,
        phase_manifest_sha256=write(case/'phase_manifest.json',phases),
        receipt_sha256=write(case/'receipt.json',receipt))
    done = json.loads(completion.read_bytes())
    done['cases'][0]['result_sha256'] = write(case/'result.json',result)
    write(completion,done)
    loaded=load_capture(case,completion)
    framework=audit(loaded,case/'model_config.json')
    framework['analysis_source_sha256']=review.source_hashes(('audit_model_coverage.py','model_statistics_io.py','statistics_values.py'))
    fp=root/'framework.json';write(fp,framework)
    stats=analyze(loaded,root/'statistics')
    stats['analysis_source_sha256']=review.source_hashes(('compare_model_statistics.py','model_statistics_io.py','statistics_values.py'))
    sp=root/'statistics/summary.json';write(sp,stats)
    return {'case_dir':case,'completion':completion,'framework_report':fp,'statistics_report':sp}


def reject(function):
    try:
        function()
    except (ValueError,KeyError,TypeError):
        return
    raise AssertionError('Expected rejection')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    base=Path('\\\\?\\'+str(HERE)) if os.name=='nt' else HERE
    checks=[]
    with tempfile.TemporaryDirectory(prefix='formal_review_cpu_',dir=base) as temporary:
        spec=enriched_fixture(Path(temporary))
        report=review.review_case(spec)
        assert report['framework_calls']==280 and not report['concerns']
        checks.append('formal_ten_step_raw_recomputation_and_280_call_hash_bindings')
        # Fixture has a deliberately invalid private binary and no genuine logits bytes.
        # Success proves this metadata review does not read those files.
        assert (spec['case_dir']/'DO_NOT_OPEN.bin').read_bytes()==b'PRIVATE_NOT_LOCALIZATION_INPUT'
        checks.append('private_bytes_not_required_or_opened_by_reviewer')
        first=next(x for x in report['source_only_statistical_inference'] if x['comparison_id']=='native_vs_fault_a')
        assert first['first_observed_model_api_difference']['node']['B']['name']=='Functional.scaled_dot_product_attention.0.forward'
        checks.append('source_only_global_first_remains_layer0_not_hash_collision_location')
        fault=next(x for x in report['separate_runtime_hash_review'] if x['comparison_id']=='native_vs_fault_a' and x['stage']=='prefill')
        assert fault['statistical_summary_collisions']==[{'layer_index':10,'changed_request_slots':[0],'classification':'SUMMARY_COLLISION'}]
        checks.append('hash_change_with_identical_four_statistics_is_explicit_summary_collision')
        assert all(x['all_input_output_logits_hashes_match'] for x in report['separate_runtime_hash_review'] if x['restoration_or_repeat_required'])
        checks.append('native_repeat_off_recovery_hashes_stay_separate_from_accuracy_claim')
        assert not report['source_only_statistical_inference'][0]['alignment_complete']
        checks.append('calibration_source_stack_mismatch_is_not_forced_into_alignment')
        loaded=load_capture(spec['case_dir'],spec['completion'])
        for step in (2,4):
            loaded['steps'][step]['sdpa'][0]['node']['output'][0]['Mean']=None
        nullable=compare_phase(loaded,2,4)
        assert nullable['layers'][0]['output_unobservable_fields']==1
        assert nullable['layers'][0]['output_difference'] is False
        assert nullable['tensor_equality_proven'] is False
        checks.append('null_statistic_remains_unobservable_not_evidence_of_tensor_equality')
        path=spec['statistics_report'].parent/'native_vs_fault_a_prefill.json'
        payload=path.read_bytes(); changed=json.loads(payload);changed['first_difference_boundary']='input';write(path,changed)
        reject(lambda:review.review_case(spec));path.write_bytes(payload)
        checks.append('modified_independent_first_difference_report_rejected')
        path=spec['case_dir']/'native_calls/call_0000/call.json'
        payload=path.read_bytes();changed=json.loads(payload);changed['output']['full_sha256']='0'*64;write(path,changed)
        reject(lambda:review.review_case(spec));path.write_bytes(payload)
        checks.append('changed_original_call_metadata_rejected_by_result_binding')
        a=snap([1,16,1,128],'a');b=copy.deepcopy(a);b['full_sha256']='0'*64
        reject(lambda:review.changed_slots(a,b))
        checks.append('contradictory_full_and_request_hashes_rejected')
    report={'schema':'nes.model_coverage.local_formal_review_cpu.v1','status':'PASS_CPU_FIXTURES',
        'checks':checks,'native_calls':0,'network_calls':0,
        'analysis_source_sha256':review.source_hashes(('review_formal_summary.py','test_review_formal_summary.py'))}
    write(args.output,report)
    print(json.dumps({'status':report['status'],'checks':len(checks),'sha256':sha(args.output)}))


if __name__=='__main__':
    main()
