"""CPU raw-byte diagnostic contracts; no Worker or device imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'experiments/c2c3_closure_20261117/diagnose_attention33_baselines.py'
spec = importlib.util.spec_from_file_location('attention33_diagnostic', SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class BaselineDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.count = 0

    def row(self, values, dtype='torch.float16', shape=None):
        self.count += 1
        values = np.asarray(values, dtype=diagnostic.FORMATS[dtype])
        raw = values.tobytes(); path = self.folder/(str(self.count)+'.bin'); path.write_bytes(raw)
        return dict(path=str(path), dtype=dtype, shape=shape or list(values.shape),
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def test_every_byte_is_checked_and_signed_zero_remains_nonexact(self):
        left, right = self.row([0., 1.]), self.row([-0., 1.])
        result = diagnostic.raw_difference(left, right)
        self.assertFalse(result['exact'])
        self.assertEqual((result['changed_elements'], result['numeric_changed_elements'], result['max_abs_difference']), (1, 0, 0))
        self.assertEqual(result['left_path'], left['path'])

    def test_float32_difference_uses_wide_cpu_arithmetic(self):
        maximum = np.finfo(np.float32).max
        result = diagnostic.raw_difference(self.row([maximum], 'torch.float32'), self.row([-maximum], 'torch.float32'))
        self.assertEqual(result['max_abs_difference'], float(maximum)*2)
        json.dumps(result, allow_nan=False)

    def test_bfloat16_is_interpreted_as_float_bits_not_uint_values(self):
        result = diagnostic.raw_difference(self.row([0x3f80, 0x4000], 'torch.bfloat16'),
                                           self.row([0x4000, 0x4000], 'torch.bfloat16'))
        self.assertEqual(result['max_abs_difference'], 1.)
        self.assertEqual(result['changed_elements'], 1)

    def test_prefill_cache_rows_and_last_decode_row_are_separately_hashed(self):
        left = self.row(np.arange(12).reshape(3, 2, 2))
        values = np.arange(12).reshape(3, 2, 2); values[-1, 0, 0] += 3
        result = diagnostic.raw_difference(left, self.row(values), split_cache=True)
        before, decode = result['cache_segments']['prefill_rows'], result['cache_segments']['last_decode_row']
        self.assertEqual(before['left_sha256'], before['right_sha256'])
        self.assertEqual(before['changed_elements'], 0)
        self.assertEqual((decode['changed_elements'], decode['max_abs_difference']), (1, 3.))

    def test_mutated_hash_trailing_file_or_nonfinite_is_refused(self):
        left, right = self.row([1., 2.]), self.row([1., 3.])
        right['sha256'] = left['sha256']
        with self.assertRaisesRegex(ValueError, 'HASH_CHANGED'):
            diagnostic.raw_difference(left, right)
        right = self.row([1., 2.]); Path(right['path']).write_bytes(Path(right['path']).read_bytes()+b'xx')
        with self.assertRaisesRegex(ValueError, 'TRAILING'):
            diagnostic.raw_difference(left, right)
        with self.assertRaisesRegex(ValueError, 'NONFINITE'):
            diagnostic.raw_difference(left, self.row([1., float('nan')]))

    def test_first_layer_follows_actual_occurrence_not_lexical_file_number(self):
        order = ['model.layers.'+str(i)+'.self_attn.attn' for i in range(28)]
        exact, changed = self.row([1., 2.]), self.row([1., 3.])
        left = {name: exact for name in ('hidden', 'logits_hidden', 'logits')}
        left.update({path+'/'+kind: exact for path in order for kind in ('output', 'key', 'value')})
        right = dict(left)
        right['hidden'] = changed
        right[order[10]+'/output'] = changed
        right[order[2]+'/output'] = changed
        result = diagnostic.compare(left, right, order)
        self.assertEqual(result['first_recorded_attention_output_difference'], order[2])
        self.assertEqual(result['changed_attention_layers']['output'], [2, 10])
        self.assertEqual(result['differing_payloads'], 3)
        right.pop(order[27]+'/key')
        with self.assertRaisesRegex(ValueError, '87_PAYLOAD'):
            diagnostic.compare(left, right, order)

    def test_diagnostic_guard_blocks_device_imports_and_library_open(self):
        code = '''import sys, ctypes, numpy
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from diagnose_attention33_baselines import prohibit_runtime
sys.addaudithook(prohibit_runtime)
from demoncat.runtime.vllm_ascend.worker_extension import paged33_tnd, paged33_loading
for action in (lambda: __import__('torch'), lambda: ctypes.CDLL('never-open-device.so')):
    try: action()
    except RuntimeError: pass
    else: raise AssertionError('runtime action was allowed')
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
print('CPU_ONLY_IMPORT_GUARD_PASS')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(SCRIPT.parent), str(ROOT/'src')],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CPU_ONLY_IMPORT_GUARD_PASS', result.stdout)

    def test_same_run_repeat_is_not_misreported_as_cross_run_or_bridge_pass(self):
        phases = ('baseline_before', 'baseline_repeat', 'measured', 'off_after', 'healthy_after')
        order = ['model.layers.'+str(i)+'.self_attn.attn' for i in range(28)]
        exact, changed = self.row([1., 2.]), self.row([1., 3.])
        names = ['hidden', 'logits_hidden', 'logits'] + [path+'/'+kind for path in order for kind in ('output', 'key', 'value')]
        geometry = dict(shape=[1, 16, 128], stride=[2048, 128, 1], storage_offset=0,
                        dtype='torch.float16', device='npu:0', storage_nbytes=4096)
        scheduler = dict(request_count=1, token_row_count=1, counts=[1], pure_decode=True,
                         row_token_positions=[1024], input_ids=[7], prefixes=[dict(prompt=[1], output=[7])])
        boundary = {path: dict(query=geometry, scheduler=scheduler) for path in order}
        for label in ('stock', 'private'):
            run = self.folder/label; run.mkdir(); rows = []
            for phase in phases:
                capture = dict(payloads={name: exact for name in names}, boundaries=boundary, native_calls=[])
                if label == 'private' and phase == 'baseline_repeat':
                    capture['payloads'][order[2]+'/output'] = changed
                rows.append(dict(phase=phase, finish=dict(capture=capture),
                    response=dict(prompt_token_ids=[1], outputs=[7]), setup=dict(source='same', module_sha256='same')))
                (run/phase).mkdir(); (run/phase/'runtime_result.json').write_text(json.dumps(capture))
            (run/'result.json').write_text(json.dumps(dict(phases=rows, status='fixture',
                execution_contract={'same': True}, warm_healthy=dict(prompt_token_ids=[1]))))
        def refuse(*args):
            raise RuntimeError('ORIGINAL_AUDIT_FAILED_FIXTURE')
        bridge = SimpleNamespace(LAYERS=set(order), audit_run=refuse, audit_native=refuse,
            payload_rows=lambda path, capture: capture['payloads'],
            boundary_rows=lambda capture, row, prompt: capture['boundaries'])
        contract = SimpleNamespace(PHASES=phases, read_json=lambda path: json.loads(path.read_bytes()))
        result = diagnostic.diagnose(bridge, contract, self.folder/'stock', self.folder/'private', self.folder/'build')
        self.assertEqual(result['within_run']['private']['baseline_repeat']['differing_payloads'], 1)
        self.assertEqual(result['cross_run']['baseline_before']['differing_payloads'], 0)
        self.assertEqual(result['cross_run']['baseline_repeat']['first_recorded_attention_output_difference'], order[2])
        self.assertNotIn('payload_details_identical_to_phase', result['cross_run']['baseline_repeat'])
        self.assertEqual(result['cross_run']['measured']['payload_details_identical_to_phase'], 'baseline_before')
        self.assertEqual(result['private_original_native_audit']['status'], 'FAILED')
        self.assertEqual(result['private']['strict_original_run_audit']['status'], 'FAILED')
        self.assertFalse(result['bridge_pass'] or result['fault_authorized'] or result['c2'] or result['c3'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
