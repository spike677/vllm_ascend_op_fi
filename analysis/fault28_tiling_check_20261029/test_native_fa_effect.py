"""CPU diagnostic-oracle regressions with synthetic mutations, never NPU proof."""
import copy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return True
    raise AssertionError('Invalid evidence accepted')


def main():
    mod = load(HERE / 'verify_native_fa_effect.py', 'effect')
    util = load(HERE / 'verify_stock_off.py', 'stock')
    source = next(p for p in (HERE / 'native_hf_fa_calibration/raw').rglob('call_000.json') if 'stock_b1' in p.parts)
    call = json.loads(source.read_bytes())
    result_path = next(p for p in (HERE / 'native_hf_fa_calibration/raw').rglob('result.json') if 'stock_b1' in p.parts)
    forward = json.loads(result_path.read_bytes())['forward_results'][0]
    baseline = []
    for layer in range(28):
        row = copy.deepcopy(call)
        row.update(layer_index=layer, layer_name=f'model.layers.{layer}.self_attn')
        row['selected_layer_full_inputs_before'] = {}
        row['selected_layer_full_inputs_after'] = {}
        if layer == 1:
            for where, input_key in [('selected_layer_full_inputs_before', 'target_inputs_before'),
                                     ('selected_layer_full_inputs_after', 'target_inputs_after')]:
                row[where] = copy.deepcopy(row[input_key])
                for r in row[where].values():
                    r['hashed_slice'] = 'entire logical tensor'
                row[where]['attn_mask'] = None
        baseline.append(row)
    changed = copy.deepcopy(baseline)
    for layer in range(1, 28):
        changed[layer]['target_output']['logical_sha256'] = 'a' * 64
        if layer > 1:
            for phase in ('target_inputs_before', 'target_inputs_after'):
                changed[layer][phase]['query']['logical_sha256'] = 'b' * 64
    changed_forward = copy.deepcopy(forward)
    changed_forward['target_logits']['logical_sha256'] = 'c' * 64
    tests = []
    def check(name, fn):
        assert fn()
        tests.append(name)
    check('actual_first_derived_as_output', lambda: mod.first_value_difference(baseline, changed)['first_observed_value_difference']['phase'] == 'output')
    check('synthetic_causal_contract_passes', lambda: mod.model_forward(util, baseline, changed, forward, changed_forward, True)['status'] == 'PASS_MODEL_VALUE_CONTRACT')
    check('unchanged_on_fails', lambda: mod.model_forward(util, baseline, baseline, forward, forward, True)['errors'])
    check('healthy_guard_passes', lambda: mod.model_forward(util, baseline, baseline, forward, forward, False)['status'] == 'PASS_MODEL_VALUE_CONTRACT')
    check('guard_with_effect_fails', lambda: mod.model_forward(util, baseline, changed, forward, changed_forward, False)['errors'])
    upstream = copy.deepcopy(changed)
    upstream[0]['target_output']['logical_sha256'] = 'd' * 64
    check('upstream_actual_first_not_overwritten', lambda: mod.first_value_difference(baseline, upstream)['first_observed_value_difference']['layer_index'] == 0)
    check('upstream_difference_rejects_mechanism', lambda: mod.model_forward(util, baseline, upstream, forward, changed_forward, True)['errors'])
    wrong_input = copy.deepcopy(changed)
    for phase in ('target_inputs_before', 'target_inputs_after'):
        wrong_input[1][phase]['key']['logical_sha256'] = 'e' * 64
    check('selected_input_first_derived', lambda: mod.first_value_difference(baseline, wrong_input)['first_observed_value_difference']['phase'] == 'input')
    check('changed_selected_input_rejected', lambda: mod.model_forward(util, baseline, wrong_input, forward, changed_forward, True)['errors'])
    full_peer_changed = copy.deepcopy(changed)
    for phase in ('selected_layer_full_inputs_before', 'selected_layer_full_inputs_after'):
        full_peer_changed[1][phase]['value']['logical_sha256'] = 'f' * 64
    check('full_input_change_rejected_even_if_target_same', lambda: mod.model_forward(util, baseline, full_peer_changed, forward, changed_forward, True)['full_native_input_errors'])
    check('missing_logits_propagation_fails', lambda: mod.model_forward(util, baseline, changed, forward, forward, True)['errors'])
    check('incomplete_recovery_fails', lambda: mod.model_forward(util, baseline, changed, forward, forward, False)['errors'])
    bad_precision = copy.deepcopy(changed)
    bad_precision[1]['scalar_arguments']['scale'] *= 2
    check('changed_precision_scalar_rejected', lambda: mod.model_forward(util, baseline, bad_precision, forward, changed_forward, True)['errors'])
    root = HERE.parent.parent
    oracle = load(root / 'experiments/fault28_native_tiling_20261029/source_matching/fa_policy/derive_query_trace.py', 'trace')
    a1 = json.loads((HERE / 'stock_off_compat_a1_verified.json').read_bytes())
    row = a1['cases'][0]['host_comparisons'][0]
    prefix = next(p for p in (HERE / 'native_hf_fa_calibration/raw').rglob('call_001.declared_prefix.bin') if 'stock_b1.fa_log_profile' in p.parts)
    event = {'schema': 'fault28.fa.typed_host.v1', 'callback_status': 0, 'workspace_count': 1,
             'workspace_sizes': row['workspace_sizes_off'], 'raw_size': 1464, 'raw_hex': prefix.read_bytes().hex(),
             'tiling_key': row['tiling_key'], 'schedule_mode': row['schedule_mode_off'],
             'typed': row['typed'], 'mode': 'off', 'eligible': False, 'policy_applied': False}
    check('real_off_source_coverage_passes', lambda: mod.derive_without_selector(oracle, event)['every_real_query_owned_once'])
    forged_hit = copy.deepcopy(event)
    forged_hit.update(mode='on', eligible=True, policy_applied=True)
    check('trace_ignores_mode_and_hits', lambda: mod.derive_without_selector(oracle, event) == mod.derive_without_selector(oracle, forged_hit))
    invalid_range = copy.deepcopy(event)
    invalid_range['typed']['sparse_start_idx'] = [0] * 48
    # Empty early ranges plus final core can still own all work; create an overlap/non-monotonic interval instead.
    invalid_range['typed']['sparse_start_idx'][2] = 12
    invalid_range['typed']['sparse_start_idx'][4] = 4
    check('invalid_causal_work_partition_rejected', lambda: rejected(lambda: mod.derive_without_selector(oracle, invalid_range)))
    check('unchanged_reduction_does_not_prove_effect', lambda: mod.trace_pair(oracle, event, event, True)['errors'])
    print(json.dumps({'status': 'PASS', 'scope': 'CPU-only synthetic mutation/recorded-profile checks; not ON execution evidence',
                      'checks': len(tests), 'tests': tests}, indent=2))


if __name__ == '__main__':
    main()
