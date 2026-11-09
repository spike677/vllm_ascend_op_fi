"""CPU-only original-log cursor tests; no device or framework imports."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))
import native_evidence as native
from model_native_stream import NativeLogCursor

ORIGINAL_FUNCTION_SHA = 'd22511c5e0384ea8316d48a0815b63cc8dad21ae7fa66d5987f33a969ef1e504'


def call(step=0, layer=0, token='p0'):
    stage = 'prefill' if step % 2 == 0 else 'decode'
    return {'case_id': 'caseA', 'call_id': token, 'api': 'torch.nn.functional.scaled_dot_product_attention',
        'pid': 123, 'thread_id': 9, 'scope_token': token if stage == 'prefill' else None,
        'phase': 'operator_execution', 'batch_size': 1, 'step': step, 'mode': 'calibration',
        'layer_index': layer, 'model_stage': stage, 'run_index': step // 2}


def mark(event, row):
    return (json.dumps({'event': event, **row, **({'status': 'returned'} if event.endswith('exit') else {})}) + '\n').encode()


def window(row):
    op = 'aclnnFlashAttentionScore_3_FlashAttentionScore'
    return b''.join([mark('native_api_enter', row),
        f'OP(123,device) [NNOP][Launch][79] OpName:[{op}] Tiling Key: 22619644976, len: 8, numBlocks: 20, Workspace Num: 1.\n'.encode(),
        f'OP(123,device) [NNOP][SplitDataAndPrint][79] OpName:[{op}] PrintTilingData[0]: Aligned TilingData Size: 8, data(uint32_t): 102\n'.encode(),
        f'OP(123,device) [NNOP][SplitDataAndPrint][79] OpName:[{op}] PrintTilingData[1]: 4 7 \n'.encode(),
        b'OP(123,device) [NNOP][CommonOpExecutorRun][79] OpName:[aclnnFlashAttentionScore_3] Workspace addr: 0x1000, size: 2048,\n',
        mark('native_api_exit', row)])


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.stream_cpu_', dir=HERE)
        self.root = Path(self.temp.name)
        self.log = self.root / 'caseA.log'
        self.first, self.second, self.decode = call(), call(2, 0, 'p1'), call(1, 0, 'd0')
        self.log.write_bytes(b'Original header\nLoadOppPackage load so, so_name is /tmp//a.so\n' + window(self.first))
        self.cursor = NativeLogCursor(self.log)

    def tearDown(self):
        self.cursor.close(); self.temp.cleanup()

    def capture(self, row, name='stream.log'):
        return self.cursor.actual_launch(self.log, row, self.root / name)

    def append(self, data):
        with self.log.open('ab') as stream: stream.write(data)

    def assert_full_equal(self, row, streamed, name='full.log'):
        old = native.actual_launch(self.log, row, self.root / name)
        actual = copy.deepcopy(streamed)
        self.assertEqual(Path(actual['window_file']).read_bytes(), Path(old['window_file']).read_bytes())
        actual['window_file'] = old['window_file']
        self.assertEqual(actual, old)

    def test_original_full_scanner_source_unchanged(self):
        source = (EXPERIMENT / 'native_evidence.py').read_text()
        node = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == 'actual_launch')
        self.assertEqual(hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(), ORIGINAL_FUNCTION_SHA)
        self.assertIsNone(inspect.signature(native.verified_record).parameters['launch_reader'].default)

    def test_two_calls_decode_between_and_appended_log_are_exactly_equal(self):
        initial = self.log.stat().st_size
        first = self.capture(self.first)
        self.assert_full_equal(self.first, first)
        self.append(b'LoadOppPackage load so, so_name is /tmp//b.so\n' + window(self.decode) + b'gap\n' + window(self.second))
        second = self.capture(self.second, 'stream2.log')
        self.assert_full_equal(self.second, second, 'full2.log')
        self.assertEqual(self.cursor.bytes_consumed, self.log.stat().st_size)
        self.assertEqual(first['source_prefix_bytes'], initial)
        self.assertEqual(len(first['loaded_libraries_before_exit']), 1)
        self.assertEqual(len(second['loaded_libraries_before_exit']), 2)
        self.assertEqual(second['raw_hex'], '0004000007000000')
        self.assertEqual(second['tid'], 79)  # Worker launch TID may differ from caller9.

    def test_consumed_prefix_rewrite_is_not_falsely_claimed_detected(self):
        self.capture(self.first)
        with self.log.open('r+b') as stream: stream.write(b'Altered! header\n')
        self.append(window(self.decode) + window(self.second))
        streamed = self.capture(self.second, 'stream2.log')
        full = native.actual_launch(self.log, self.second, self.root / 'full2.log')
        self.assertEqual(streamed['raw_hex'], full['raw_hex'])
        self.assertNotEqual(streamed['source_prefix_sha256'], full['source_prefix_sha256'])
        # Independent acceptance recomputes this full prefix and rejects the mismatch.

    def test_repeat_requested_call_rejected(self):
        self.capture(self.first)
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            self.capture(self.first, 'again.log')

    def test_receipt_and_idempotent_close_preserve_limits(self):
        captured = self.capture(self.first)
        self.cursor.close(); self.cursor.close()
        receipt = self.cursor.receipt()
        self.assertEqual(receipt['mode'], 'sequential_original_log_prefix')
        self.assertEqual(receipt['bytes_consumed'], captured['source_prefix_bytes'])
        self.assertEqual(receipt['prefix_sha256'], captured['source_prefix_sha256'])
        self.assertEqual(receipt['completed_prefill_windows'], 1)
        self.assertTrue(receipt['closed']); self.assertFalse(receipt['failed'])
        self.assertFalse(receipt['full_log_rehashed']); self.assertTrue(receipt['independent_acceptance_required'])

    def test_out_of_order_request_rejected(self):
        self.capture(self.first)
        self.append(window(self.decode) + window(self.second)); self.capture(self.second, 'second.log')
        earlier = call(0, 1, 'earlier')
        with self.assertRaisesRegex(ValueError, 'Out-of-order'):
            self.capture(earlier, 'earlier.log')

    def test_skipped_prefill_rejected(self):
        self.append(window(self.second))
        with self.assertRaisesRegex(ValueError, 'Unrequested prefill'):
            self.capture(self.second)

    def test_decode_inherited_scope_rejected(self):
        self.capture(self.first)
        bad = dict(self.decode, scope_token='unexpected')
        self.append(window(bad) + window(self.second))
        with self.assertRaisesRegex(ValueError, 'Unrequested prefill'):
            self.capture(self.second, 'second.log')

    def test_foreign_pid_marker_rejected(self):
        bad = dict(self.first, pid=124)
        with self.assertRaisesRegex(ValueError, 'Cross-case/PID/TID'):
            self.capture(bad)

    def test_foreign_caller_tid_marker_rejected(self):
        bad = dict(self.first, thread_id=10)
        with self.assertRaisesRegex(ValueError, 'Cross-case/PID/TID'):
            self.capture(bad)

    def test_wrong_token_marker_rejected(self):
        bad = dict(self.first, scope_token='wrong')
        with self.assertRaisesRegex(ValueError, 'individually scoped'):
            self.capture(bad)

    def rewrite_fresh_log(self, payload):
        self.cursor.close(); self.log.write_bytes(payload); self.cursor = NativeLogCursor(self.log)

    def test_cross_call_marker_preserved_in_failed_original_window(self):
        cross = call(0, 1, 'cross')
        self.rewrite_fresh_log(mark('native_api_enter', self.first) + mark('native_api_enter', cross))
        before = native.sha(self.log)
        with self.assertRaisesRegex(ValueError, 'Cross-call or nested'):
            self.capture(self.first)
        self.assertEqual((self.root / 'stream.log').read_bytes(), self.log.read_bytes())
        self.assertEqual(native.sha(self.log), before)
        with self.assertRaisesRegex(ValueError, 'closed or failed'):
            self.capture(self.first, 'retry.log')

    def test_wrong_exit_identity_rejected(self):
        self.rewrite_fresh_log(mark('native_api_enter', self.first) + mark('native_api_exit', dict(self.first, scope_token='wrong')))
        with self.assertRaisesRegex(ValueError, 'mismatched PID/TID/token'):
            self.capture(self.first)

    def test_truncation_rejected(self):
        self.capture(self.first)
        with self.log.open('r+b') as stream: stream.truncate(3)
        with self.assertRaisesRegex(ValueError, 'truncated'):
            self.capture(self.second, 'second.log')

    def test_inode_replacement_rejected(self):
        # Portable Windows fixture: simulate the path stat after an inode swap.
        original = Path.stat
        def different(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if Path(path) == self.log:
                fields = list(info); fields[1] += 1
                return os.stat_result(fields)
            return info
        with patch.object(Path, 'stat', different):
            with self.assertRaisesRegex(ValueError, 'inode/path was replaced'):
                self.capture(self.first)

    def test_different_log_path_rejected(self):
        other = self.root / 'other.log'; other.write_bytes(self.log.read_bytes())
        with self.assertRaisesRegex(ValueError, 'different original log'):
            self.cursor.actual_launch(other, self.first, self.root / 'other_window.log')

    def test_malformed_raw_preserves_original_evidence(self):
        data = window(self.first).replace(b'PrintTilingData[1]: 4 7 ', b'PrintTilingData[1]: 4 X ')
        self.rewrite_fresh_log(data)
        with self.assertRaisesRegex(ValueError, 'Malformed selected native raw'):
            self.capture(self.first)
        self.assertEqual((self.root / 'stream.log').read_bytes(), data)

    def test_incomplete_selected_window_is_not_retried(self):
        data = mark('native_api_enter', self.first) + b'ordinary line\n'
        self.rewrite_fresh_log(data)
        with self.assertRaisesRegex(ValueError, 'exit marker missing'):
            self.capture(self.first)
        self.assertEqual((self.root / 'stream.log').read_bytes(), data)
        self.append(mark('native_api_exit', self.first))
        with self.assertRaisesRegex(ValueError, 'closed or failed'):
            self.capture(self.first, 'retry.log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StreamTests))
    report = {'schema': 'nes.model_native_stream.cpu.v1',
        'status': 'PASS_SYNTHETIC_CPU_CHECKS' if result.wasSuccessful() else 'FAIL_CPU_CHECKS',
        'tests_run': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
        'npu_calls': 0, 'ssh_connections': 0, 'original_full_scanner_function_sha256': ORIGINAL_FUNCTION_SHA,
        'limitations': ['Consumed-prefix same-length rewriting is not detected by cursor alone; independent whole-log acceptance must rehash original bytes.',
                        'Synthetic raw fixture verifies framing only; it is not a real tiling/kernel/mechanism result.'],
        'sources': {str(p.relative_to(EXPERIMENT)): native.sha(p) for p in
                    (EXPERIMENT / 'model_native_stream.py', EXPERIMENT / 'native_evidence.py', EXPERIMENT / 'model_probe.py', Path(__file__))}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
