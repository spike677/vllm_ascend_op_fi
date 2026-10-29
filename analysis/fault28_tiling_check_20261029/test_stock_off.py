"""CPU-only regressions; synthetic mutations are not NPU execution evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('verify_stock_off', HERE / 'verify_stock_off.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return True
    raise AssertionError('Invalid evidence was accepted')


def main():
    # Real previously retained stock metadata, copied in memory; evidence remains immutable.
    case = HERE / 'native_hf_fa_calibration/raw/stock_b1/native_calls/call_000.json'
    if not case.exists():
        matches = list((HERE / 'native_hf_fa_calibration/raw').rglob('call_000.json'))
        case = next(p for p in matches if 'stock_b1' in p.parts)
    original = json.loads(case.read_bytes())
    tests = []
    def check(name, fn):
        assert fn()
        tests.append(name)
    check('same_numeric_trace_accepts', lambda: m.compare_calls(original, original)['target_inputs_and_output_equal'])
    missing = copy.deepcopy(original)
    missing['target_output'] = None
    check('both_missing_outputs_rejected', lambda: rejected(lambda: m.compare_calls(missing, missing)))
    changed = copy.deepcopy(original)
    for key in ('target_inputs_before', 'target_inputs_after'):
        for value in changed[key].values():
            value['object_id'] += 17
            value['tensor_ptr'] += 4096
    changed['target_output']['object_id'] += 17
    changed['target_output']['tensor_ptr'] += 4096
    check('cross_process_identity_not_compared', lambda: m.compare_calls(original, changed)['target_inputs_and_output_equal'])
    changed = copy.deepcopy(original)
    changed['target_output']['logical_sha256'] = 'a' * 64
    check('output_bit_difference_fails', lambda: not m.compare_calls(original, changed)['target_inputs_and_output_equal'])
    changed = copy.deepcopy(original)
    changed['target_inputs_after']['query']['logical_sha256'] = 'b' * 64
    check('api_mutated_input_rejected', lambda: rejected(lambda: m.compare_calls(original, changed)))
    changed = copy.deepcopy(original)
    for phase in ('target_inputs_before', 'target_inputs_after'):
        changed[phase]['query']['logical_sha256'] = 'c' * 64
    check('changed_before_input_reported_before_output', lambda: m.compare_calls(original, changed)['differences'][0]['field'] == 'target_inputs_before.query.logical_sha256')
    changed = copy.deepcopy(original)
    changed['target_inputs_after']['query']['actual_format'] = 29
    check('format29_rejected', lambda: rejected(lambda: m.compare_calls(original, changed)))
    changed = copy.deepcopy(original)
    changed['target_inputs_before']['query']['actual_format'] = 2
    changed['target_inputs_after']['query']['actual_format'] = 0
    check('reverse_normalization_rejected', lambda: rejected(lambda: m.compare_calls(original, changed)))
    changed = copy.deepcopy(original)
    changed['target_inputs_before']['query']['logical_bytes'] -= 2
    check('incomplete_target_hash_rejected', lambda: rejected(lambda: m.compare_calls(original, changed)))
    changed = copy.deepcopy(original)
    changed['scalar_arguments']['is_causal'] = False
    check('causal_metadata_change_fails', lambda: not m.compare_calls(original, changed)['target_inputs_and_output_equal'])
    changed = copy.deepcopy(original)
    changed['target_mapping']['last_token_position'] -= 1
    check('wrong_target_position_fails', lambda: not m.compare_calls(original, changed)['target_inputs_and_output_equal'])
    root = HERE.parent.parent
    schema = json.loads((root / 'experiments/fault28_native_tiling_20261029/source_matching/fa_policy/build_evidence/fa_pod_schema.json').read_bytes())
    prefix = next(p for p in (HERE / 'native_hf_fa_calibration/raw').rglob('call_001.declared_prefix.bin') if 'stock_b1.fa_log_profile' in p.parts)
    raw = prefix.read_bytes()
    decoded = m.decode_pod(raw, schema)
    check('public_pod105_decodes', lambda: len(decoded) == 105 and decoded['coreParams.s1BaseSize'] == 128)
    check('raw_length_rejected', lambda: rejected(lambda: m.decode_pod(raw[:-1], schema)))
    typed = {k: decoded[v] for k, v in m.TYPED_MAP.items()}
    typed.update(n1=typed['n2'] * typed['g'], s2_group=typed['s2_base'] * typed['n_ratio'])
    typed['scale'] = float(format(typed['scale'], '.10g'))
    check('float32_projection_roundtrips', lambda: not m.typed_matches(typed, decoded))
    typed['m'] //= 2
    check('typed_m_disagreement_rejected', lambda: 'm' in m.typed_matches(typed, decoded))
    bad = bytearray(raw)
    bad[576] ^= 1
    check('one_raw_byte_changes_pod', lambda: m.decode_pod(bad, schema) != decoded)
    # Actual a1 must stay failed: numeric/raw equality does not waive selector control.
    report = json.loads((HERE / 'stock_off_compat_a1.json').read_bytes())
    check('real_a1_selector_failure_preserved', lambda: report['status'] == 'FAIL_STOCK_OFF_COMPATIBILITY' and
          all(c['numerical_comparison_pass'] for c in report['cases']) and
          report['cases'][1]['host_comparisons'][0]['errors'] == ['Expected B8 eligible and B1 ineligible'])
    print(json.dumps({'status': 'PASS', 'scope': 'CPU-only synthetic mutations and immutable a1 report inspection',
                      'checks': len(tests), 'tests': tests}, indent=2))


if __name__ == '__main__':
    main()
