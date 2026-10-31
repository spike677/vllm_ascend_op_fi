"""Targeted synthetic CPU acceptance regressions; never real model evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import accept_model_execution as a
import native_evidence


def snapshot(shape, data=None, tag='stable'):
    length = 2 * __import__('math').prod(shape)
    if data is None:
        whole = hashlib.sha256((tag + repr(shape)).encode()).hexdigest()
        slots = [hashlib.sha256((tag + repr(shape[1:]) + str(i)).encode()).hexdigest() for i in range(shape[0])]
    else:
        size = len(data) // shape[0]
        whole = hashlib.sha256(data).hexdigest()
        slots = [hashlib.sha256(data[i * size:(i + 1) * size]).hexdigest() for i in range(shape[0])]
    stride, n = [], 1
    for dimension in reversed(shape):
        stride.insert(0, n); n *= dimension
    return {'shape': shape, 'stride': stride, 'dtype': 'torch.float16', 'device': 'npu:0',
        'object_id': 123, 'data_ptr': 456, 'storage_offset': 0, 'actual_format': 0,
        'full_sha256': whole, 'bytes': length,
        'per_batch_sha256': slots,
        'raw_order': 'C-contiguous logical elements', 'byteorder': 'little'}


def call():
    shapes = {'query': [1, 16, 1, 128], 'key': [1, 8, 2049, 128], 'value': [1, 8, 2049, 128]}
    before = {k: snapshot(v) for k, v in shapes.items()}
    return {'inputs_before': before, 'inputs_after': copy.deepcopy(before),
        'sdpa_metadata': {**{k: {'shape': v, 'dtype': 'torch.float16'} for k, v in shapes.items()},
                          'attn_mask': None, 'dropout_p': 0.0, 'is_causal': False, 'enable_gqa': True, 'scale': None},
        'allowed_input_format_normalizations': [], 'output': snapshot(shapes['query']),
        'original_api_calls': 1, 'original_return_identity_preserved': True}


class AcceptanceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.accept_cpu_', dir=a.HERE)
        self.root = Path(self.temp.name) / 'caseA'; self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_supported_input_integrity(self):
        a.input_integrity(call(), 1, 'decode')

    def test_only_one_way_format_normalization(self):
        row = call(); row['inputs_after']['query']['actual_format'] = 2
        row['allowed_input_format_normalizations'] = [{'input': 'query', 'from': 0, 'to': 2, 'logical_bytes_unchanged': True}]
        a.input_integrity(row, 1, 'decode')
        row['inputs_before']['query']['actual_format'] = 2; row['inputs_after']['query']['actual_format'] = 0
        with self.assertRaises(ValueError): a.input_integrity(row, 1, 'decode')

    def test_input_hash_changed_rejected(self):
        row = call(); row['inputs_after']['key']['full_sha256'] = '1' * 64
        with self.assertRaises(ValueError): a.input_integrity(row, 1, 'decode')

    def test_pointer_replacement_rejected(self):
        row = call(); row['inputs_after']['value']['data_ptr'] += 2
        with self.assertRaises(ValueError): a.input_integrity(row, 1, 'decode')

    def test_request_axis_not_inferred_from_one_hash(self):
        row = call(); row['output']['per_batch_sha256'] = []
        with self.assertRaises(ValueError): a.input_integrity(row, 1, 'decode')

    def test_repeat_original_call_rejected(self):
        row = call(); row['original_api_calls'] = 2
        with self.assertRaises(ValueError): a.input_integrity(row, 1, 'decode')

    def test_in_memory_container_path_mapping(self):
        folder = self.root / 'native_calls/call_0000'; folder.mkdir(parents=True)
        path = folder / 'host_events.original.jsonl'; path.write_bytes(b'{"unaltered":true}\n')
        original = {'host_event': {'raw_hex': '00'}, 'evidence': [{'path': '/evidence/caseA/native_calls/call_0000/host_events.original.jsonl', 'sha256': a.sha(path)}]}
        frozen = copy.deepcopy(original)
        mapped = a.remap_record(self.root, original, folder)
        self.assertEqual(original, frozen)
        self.assertEqual(mapped['host_event'], original['host_event'])
        self.assertEqual(Path(mapped['evidence'][0]['path']), path.resolve())

    def test_cross_case_path_rejected(self):
        with self.assertRaises((ValueError, OSError)):
            a.mapped_path(self.root, '/evidence/caseB/native_calls/call_0000/x.json')

    def test_parent_path_rejected(self):
        with self.assertRaises(ValueError): a.mapped_path(self.root, '/evidence/caseA/../caseB/x.json')

    def test_changed_original_host_bytes_rejected(self):
        folder = self.root / 'native_calls/call_0000'; folder.mkdir(parents=True)
        path = folder / 'host_events.original.jsonl'; path.write_bytes(b'changed')
        record = {'evidence': [{'path': str(path.resolve()), 'sha256': '0' * 64}]}
        with self.assertRaises(ValueError): a.remap_record(self.root, record, folder)

    def test_complete_logits_validate_every_slot(self):
        data = b'\x00\x3c\x00\x40\x00\x42\x00\x44'
        path = self.root / 'private_output/logits.bin'; path.parent.mkdir(); path.write_bytes(data)
        phase = {'logits': snapshot([2, 1, 2], data), 'private_logits_file': 'private_output/logits.bin'}
        self.assertEqual(a.logits_bytes(self.root, phase, 2), [data[:4], data[4:]])
        phase['logits']['per_batch_sha256'][1] = '0' * 64
        with self.assertRaises(ValueError): a.logits_bytes(self.root, phase, 2)

    def test_logits_private_bytes_changed_rejected(self):
        path = self.root / 'logits.bin'; path.write_bytes(b'\0\0\0\0')
        phase = {'logits': snapshot([1, 1, 2]), 'private_logits_file': 'logits.bin'}
        path.write_bytes(b'\0\0\0\1')
        with self.assertRaises(ValueError): a.logits_bytes(self.root, phase, 1)

    def numeric_fixture(self):
        phases, calls = {}, {}
        for step in range(10):
            data = b'\0\0\0\0' if step // 2 not in (2, 3) else b'\0\0\0\1'
            path = self.root / f'logits{step}.bin'; path.write_bytes(data)
            phases[step] = {'logits': snapshot([1, 1, 2], data), 'private_logits_file': path.name}
            for layer in range(28):
                row = call()
                if step // 2 in (2, 3):
                    # Downstream changed inputs are legitimate propagation, not an acceptance failure.
                    if layer > 0:
                        for key in row['inputs_before']:
                            row['inputs_before'][key]['full_sha256'] = '1' * 64
                    row['output']['full_sha256'] = '2' * 64
                    row['output']['per_batch_sha256'] = ['2' * 64]
                calls[step, layer] = row
        return calls, phases

    def test_fault_input_propagation_allowed_and_first_derived(self):
        calls, phases = self.numeric_fixture()
        report = a.compare_output_rows(calls, phases, self.root, 1, False)
        first = report['comparisons'][1]['stages'][0]['first_runtime_hash_difference']
        self.assertEqual(first['layer_index'], 0); self.assertEqual(first['boundary'], 'output')
        self.assertTrue(report['numerics_observed']); self.assertTrue(report['model_logits_changed'])

    def test_first_difference_not_assigned_from_policy(self):
        calls, phases = self.numeric_fixture()
        for step in (4, 5, 6, 7):
            calls[step, 0] = copy.deepcopy(calls[2 + step % 2, 0])
        report = a.compare_output_rows(calls, phases, self.root, 1, False)
        first = report['comparisons'][1]['stages'][0]['first_runtime_hash_difference']
        self.assertEqual((first['layer_index'], first['boundary']), (1, 'input'))

    def test_off_output_change_rejected(self):
        calls, phases = self.numeric_fixture()
        calls[8, 27]['output']['full_sha256'] = '3' * 64
        calls[8, 27]['output']['per_batch_sha256'] = ['3' * 64]
        with self.assertRaises(ValueError): a.compare_output_rows(calls, phases, self.root, 1, False)

    def test_fault_repeat_instability_rejected(self):
        calls, phases = self.numeric_fixture()
        calls[6, 18]['inputs_before']['query']['full_sha256'] = '4' * 64
        with self.assertRaises(ValueError): a.compare_output_rows(calls, phases, self.root, 1, False)

    def test_equal_fault_numerics_are_valid_observation(self):
        calls, phases = self.numeric_fixture()
        for step in (4, 5, 6, 7):
            for layer in range(28): calls[step, layer] = copy.deepcopy(calls[2 + step % 2, layer])
            phases[step] = copy.deepcopy(phases[2 + step % 2])
        report = a.compare_output_rows(calls, phases, self.root, 1, False)
        self.assertFalse(report['numerics_observed']); self.assertFalse(report['model_logits_changed'])

    def test_pilot_never_grants_formal_acceptance(self):
        fixture = {'framework_calls': 56, 'eligible_prefill_calls': 28, 'unsupported_decode_calls': 28,
                   'actual_strategy_switches': 0, 'switched_layer_request_pairs': 0}
        with patch.object(a, 'accept_case', return_value=fixture):
            report = a.accept_suite([{}], True)
        self.assertFalse(report['formal_coverage_accepted'])
        self.assertNotIn('PASS', report['status'])

    def native_fixture(self):
        """Minimal synthetic CANN log; actual reviewed parser performs framing."""
        call = {'case_id': 'caseA', 'call_id': 'token_a', 'api': 'torch.nn.functional.scaled_dot_product_attention',
                'pid': 123, 'thread_id': 9, 'scope_token': 'token_a', 'phase': 'operator_execution', 'batch_size': 1, 'step': 0}
        op = 'aclnnFlashAttentionScore_3_FlashAttentionScore'
        lines = [b'Original unrelated header line\n',
            (json.dumps({'event': 'native_api_enter', **call}) + '\n').encode(),
            f'OP(123,device) [NNOP][Launch][9] OpName:[{op}] Tiling Key: 22619644976, len: 8, numBlocks: 20, Workspace Num: 1.\n'.encode(),
            f'OP(123,device) [NNOP][SplitDataAndPrint][9] OpName:[{op}] PrintTilingData[0]: Aligned TilingData Size: 8, data(uint32_t): 102\n'.encode(),
            f'OP(123,device) [NNOP][SplitDataAndPrint][9] OpName:[{op}] PrintTilingData[1]: 4 7 \n'.encode(),
            b'OP(123,device) [NNOP][CommonOpExecutorRun][9] OpName:[aclnnFlashAttentionScore_3] Workspace addr: 0x1000, size: 2048,\n',
            (json.dumps({'event': 'native_api_exit', 'status': 'returned', **call}) + '\n').encode()]
        log = self.root / 'caseA.log'; log.write_bytes(b''.join(lines))
        window = self.root / 'native_window.log'
        observed = native_evidence.actual_launch(log, call, window)
        def marker(index):
            start = sum(len(x) for x in lines[:index]); payload = lines[index]
            return {'line_number': index + 1, 'byte_range': [start, start + len(payload)],
                    'line_sha256': hashlib.sha256(payload).hexdigest()}
        census = {'enter': marker(1), 'exit': marker(6)}
        record = {'native_launch': observed, 'host_event': {'pid': 123, 'tid': 9, 'token': 'token_a', 'raw_hex': observed['raw_hex']},
                  'evidence': [{'path': str(window), 'sha256': a.sha(window)}]}
        return log, call, census, record

    def reparse(self, log, call, census, record):
        verifier = a.OriginalLogVerifier(log, native_evidence, self.root / 'temporary_reparse')
        try:
            return verifier.verify(call, census, record)
        finally:
            verifier.close()

    def test_raw_reparsed_with_character_split_and_no_original_rewrite(self):
        log, call, census, record = self.native_fixture()
        before = a.sha(log)
        result = self.reparse(log, call, census, record)
        self.assertTrue(result['raw_reparsed_from_original_log'])
        self.assertEqual(record['native_launch']['raw_hex'], '0004000007000000')
        self.assertEqual(a.sha(log), before)

    def test_joint_claim_same_key_block_but_original_baseline_raw_rejected(self):
        log, call, census, record = self.native_fixture()
        # Both mutually agreeing derived JSON rows claim another recipe while
        # the original CANN line/bytes retain the first recipe and same key/block.
        other = '4000000004000000'
        record['host_event']['raw_hex'] = other
        record['native_launch']['raw_hex'] = other
        record['native_launch']['raw_sha256'] = hashlib.sha256(bytes.fromhex(other)).hexdigest()
        with self.assertRaisesRegex(ValueError, 'ORIGINAL launch bytes'):
            self.reparse(log, call, census, record)

    def test_wrong_host_token_rejected(self):
        log, call, census, record = self.native_fixture()
        record['host_event']['token'] = 'another_call'
        with self.assertRaisesRegex(ValueError, 'token/PID/TID'):
            self.reparse(log, call, census, record)

    def test_wrong_native_marker_token_rejected(self):
        log, call, census, record = self.native_fixture()
        call['call_id'] = call['scope_token'] = 'other_token'
        record['host_event']['token'] = 'other_token'
        with self.assertRaisesRegex(ValueError, 'Incomplete call window'):
            self.reparse(log, call, census, record)

    def test_wrong_window_byte_range_rejected(self):
        log, call, census, record = self.native_fixture()
        record['native_launch']['window_byte_range'][0] += 1
        with self.assertRaisesRegex(ValueError, 'marker range'):
            self.reparse(log, call, census, record)

    def test_wrong_window_line_range_rejected(self):
        log, call, census, record = self.native_fixture()
        record['native_launch']['window_line_range'][0] += 1
        with self.assertRaisesRegex(ValueError, 'marker range'):
            self.reparse(log, call, census, record)

    def test_wrong_original_marker_sha_rejected(self):
        log, call, census, record = self.native_fixture()
        census['enter']['line_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'marker byte/line/SHA'):
            self.reparse(log, call, census, record)

    def test_wrong_original_prefix_sha_rejected(self):
        log, call, census, record = self.native_fixture()
        record['native_launch']['source_prefix_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'prefix SHA'):
            self.reparse(log, call, census, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=a.HERE / 'accept_model_execution_cpu_checks.json')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AcceptanceChecks)
    run = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'schema': 'nes.model_coverage.execution_acceptance.cpu.v1',
        'status': 'PASS_TARGETED_SYNTHETIC_CPU_CHECKS' if run.wasSuccessful() else 'FAIL_CPU_CHECKS',
        'tests_run': run.testsRun, 'failures': len(run.failures), 'errors': len(run.errors),
        'native_calls': 0, 'ssh_connections': 0, 'formal_model_coverage_accepted': False,
        'complete_real_capture_schema_executed': False,
        'scope': 'Acceptance helper/schema/negative regressions only; full recorded-model pipeline awaits pilot and formal original data.',
        'sources': {p.name: a.sha(p) for p in (Path(__file__), Path(a.__file__))}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    return 0 if run.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
