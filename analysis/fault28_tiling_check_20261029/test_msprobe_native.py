"""CPU tests of native evidence interpretation; no tensors/NPU/control inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_analysis_tested', HERE / 'analyze_msprobe_native.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def tensor(maximum=4, null=False):
    return {'type': 'torch.Tensor', 'dtype': 'torch.float16', 'shape': [1, 2, 4, 64],
            'Max': None if null else maximum, 'Min': None if null else -2,
            'Mean': None if null else 0.25, 'Norm': None if null else 16,
            'requires_grad': False}


def fixture():
    config = {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': [0], 'dump_path': '/B/observation',
              'async_dump': False, 'statistics': {'scope': [], 'list': [], 'data_mode': ['all'],
                                                 'summary_mode': 'statistics', 'tensor_list': []}}
    data, stack = {}, {}
    root = 'Module.model.Qwen3ForCausalLM.forward.0'
    construct = {root: None}
    for layer in range(2):
        module = f'Module.model.layers.{layer}.self_attn.Qwen3Attention.forward.0'
        api = f'Functional.scaled_dot_product_attention.{layer}.forward'
        construct.update({module: root, api: module})
        data[api] = {'input_args': [tensor(), tensor(), tensor()],
                     'input_kwargs': {'dropout_p': {'type': 'float', 'value': 0.0},
                                      'is_causal': {'type': 'bool', 'value': True}},
                     'output': [tensor()]}
        data[module] = {'input_args': [tensor()], 'input_kwargs': {}, 'output': [tensor()]}
        stack[str(len(stack))] = [[api], [
            'File /audit/hf_runtime_probe.py, line 245, in observe,\n result = self.original(*args, **kwargs)',
            'File /site/transformers/integrations/sdpa_attention.py, line 60, in sdpa_attention_forward,\n output = F.scaled_dot_product_attention(...)']]
        stack[str(len(stack))] = [[module], [
            'File /site/torch/nn/modules/module.py, line 1841, in inner,\n hook_result = hook(...)']]
    return {'config': config, 'dump': {'task': 'statistics', 'level': 'mix', 'framework': 'pytorch', 'data': data},
            'stack': stack, 'construct': construct}


def run(left, right, expected=2):
    with tempfile.TemporaryDirectory(dir=HERE, prefix='.native_analyzer_cpu_') as td:
        assert Path(td).resolve().parent == HERE  # Bound automatic cleanup to this workspace test directory.
        roots = [Path(td) / 'B', Path(td) / 'F']
        for root, source in zip(roots, [left, right]):
            root.mkdir()
            for kind, value in source.items():
                (root / (kind + '.json')).write_text(json.dumps(value), encoding='utf-8')
            # These are deliberately invalid. Any attempt to consume other evidence fails.
            (root / 'control.json').write_text('NOT JSON: expected_fault_point=layer999', encoding='utf-8')
            (root / 'runtime.json').write_text('NOT JSON: injection_hit=true', encoding='utf-8')
        report = a.analyze(roots[0], roots[1], roots[0] / 'config.json', roots[1] / 'config.json',
                           expected_sdpa_calls=expected)
        json.dumps(report, allow_nan=False)
        return report


def reject(left, right):
    try:
        run(left, right)
    except ValueError:
        return
    raise AssertionError('Expected malformed evidence rejection')


def main():
    done = []
    b = fixture()
    report = run(b, copy.deepcopy(b))
    assert report['status'] == 'ALIGNED_AVAILABLE_STATISTICS_MATCH'
    assert report['tensor_equality_proven'] is False and report['fault_mechanism_accepted'] is False
    assert report['attention_boundary_coverage']['B']['complete_hf_sdpa_module_coverage']
    assert not report['attention_boundary_coverage']['B']['explicit_flash_attention_score_recorded']
    done.append('matching_statistics_not_tensor_equality_and_sdpa_not_cann_launch')

    f = copy.deepcopy(b)
    f['dump']['data']['Functional.scaled_dot_product_attention.1.forward']['output'][0]['Max'] = 5
    report = run(b, f)
    first = report['first_observed_model_api_output_difference']
    assert first['B']['name'] == 'Functional.scaled_dot_product_attention.1.forward'
    assert first['input_summaries_match_where_observable'] and not first['input_values_proven_equal']
    done.append('first_attention_output_found_without_reading_control_or_runtime')

    diagnostic = 'Tensor.cpu.0.forward'
    for item in (b, f):
        node = {'input_args': [tensor()], 'input_kwargs': {}, 'output': [tensor()]}
        item['dump']['data'] = {diagnostic: node, **item['dump']['data']}
        item['construct'][diagnostic] = None
        item['stack']['99'] = [[diagnostic], [
            'File /audit/hf_runtime_probe.py, line 156, in record,\n tensor.detach().to(cpu)',
            'File /audit/hf_runtime_probe.py, line 220, in observe,\n before = self.record(...)']]
    f['dump']['data'][diagnostic]['output'][0]['Max'] = 99
    report = run(b, f)
    assert report['raw_first_recorded_difference_including_diagnostics']['B']['name'] == diagnostic
    assert report['first_observed_model_api_numerical_difference']['B']['name'].endswith('.1.forward')
    assert len(report['diagnostic_filter']['B_excluded']) == 1
    assert len(report['attention_boundary_coverage']['B']['nodes']) == 2
    done.append('raw_first_diagnostic_separated_while_true_observe_call_retained')

    b = fixture()
    f = copy.deepcopy(b)
    api0 = 'Functional.scaled_dot_product_attention.0.forward'
    b['dump']['data'][api0]['output'][0] = tensor(null=True)
    f['dump']['data'][api0]['output'][0] = tensor()
    report = run(b, f)
    assert report['first_observed_model_api_numerical_difference'] is None
    assert report['unobservable_statistic_field_count'] == 4
    done.append('null_to_finite_is_unobservable_not_numerical_difference')

    f['dump']['data']['Functional.scaled_dot_product_attention.1.forward']['output'][0]['Max'] = 6
    report = run(b, f)
    assert report['unobservable_statistic_fields_before_first_observed_difference'] == 4
    done.append('earlier_nulls_are_disclosed_before_first_observed_difference')

    b = fixture()
    f = copy.deepcopy(b)
    f['dump']['data'][api0]['input_args'][0]['shape'][0] = 8
    f['dump']['data'][api0]['output'][0]['Max'] = 9
    report = run(b, f)
    assert report['status'] == 'INCONCLUSIVE_ALIGNMENT_OR_METADATA'
    assert report['first_observed_model_api_numerical_difference'] is None
    assert not report['independent_target_slice_localization_supported']
    done.append('batch_shape_mismatch_cannot_be_target_slice_comparison')

    f = copy.deepcopy(b)
    f['dump']['data'][api0]['input_args'][0]['Max'] = 9
    f['dump']['data'][api0]['output'][0]['Max'] = 10
    report = run(b, f)
    assert not report['first_observed_model_api_output_difference']['input_summaries_match_where_observable']
    done.append('changed_input_summaries_are_not_claimed_local_output_fault')

    f = copy.deepcopy(b)
    f['dump']['data'] = dict(reversed(list(f['dump']['data'].items())))
    report = run(b, f)
    assert report['status'] == 'INCONCLUSIVE_ALIGNMENT_OR_METADATA'
    assert report['first_observed_model_api_numerical_difference'] is None
    done.append('reordered_native_records_suppress_first_difference_claim')

    f = copy.deepcopy(b)
    del f['stack']['0']
    report = run(b, f)
    assert report['status'] == 'INCONCLUSIVE_ALIGNMENT_OR_METADATA'
    done.append('missing_native_stack_fails_alignment')

    f = copy.deepcopy(b)
    module = f['construct'][api0]
    f['construct'][module] = module
    reject(b, f)
    done.append('construct_cycle_rejected')

    f = copy.deepcopy(b)
    f['config']['statistics']['summary_mode'] = 'md5'
    reject(b, f)
    done.append('md5_mode_not_misrepresented_as_pure_statistics')

    f = copy.deepcopy(b)
    f['config']['statistics']['data_mode'] = ['output']
    report = run(b, f)
    assert not report['alignment']['config_and_framework_match']
    done.append('different_capture_config_rejected_for_full_alignment')

    report = run(b, copy.deepcopy(b), expected=28)
    assert not report['attention_boundary_coverage']['B']['complete_hf_sdpa_module_coverage']
    done.append('incomplete_model_attention_coverage_detected')

    f = copy.deepcopy(b)
    f['dump']['data'][api0]['output'][0]['Max'] = float('inf')
    report = run(b, f)
    assert report['first_observed_model_api_output_difference']['output']['numerical_differences'][0]['kind'] == 'nonfinite_summary_state_difference'
    done.append('finite_to_infinity_reported_without_invalid_json')

    print(json.dumps({'status': 'PASS_CPU_NATIVE_STATISTICS_ANALYZER', 'count': len(done), 'tests': done,
                      'NPU_executed': False, 'synthetic_fixtures_are_not_fault_evidence': True}, indent=2))


if __name__ == '__main__':
    main()
