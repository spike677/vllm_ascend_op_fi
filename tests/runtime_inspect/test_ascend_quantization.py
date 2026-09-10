import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from ascend_fi.inspect import ascend_quantization as aq


def instance(qualified):
    module, _, name = qualified.rpartition('.')
    return type(name, (), {'__module__': module})()


class PinnedContractTests(unittest.TestCase):
    def contract(self, scheme_name):
        wrapper = instance(aq.WRAPPER)
        wrapper.quant_method = instance(scheme_name)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'fixture.py'
            path.write_bytes(b'# explicit fixture, not vendor source\n')
            hashes = {k: hashlib.sha256(path.read_bytes()).hexdigest() for k in aq.SOURCE_HASHES}
            with patch.object(aq, 'SOURCE_HASHES', hashes), patch.object(aq.inspect, 'getfile', return_value=str(path)):
                return aq.audited_ascend_contract(wrapper, 'fixture-runtime')

    def test_static_roles_and_reference_boundary(self):
        c = self.contract(aq.STATIC)
        self.assertFalse(c['activation_scale']['dynamic'])
        self.assertEqual(c['fields']['input_scale']['target'], 'activation')
        self.assertEqual(c['fields']['deq_scale']['target'], 'dequantization')
        self.assertEqual(c['fields']['weight_scale']['providers'], [])
        self.assertIn('UNSUPPORTED', c['activation_scale']['model_scale_fake_requant'])

    def test_dynamic_is_local_temporary_not_static_field(self):
        c = self.contract(aq.SCHEME)
        self.assertTrue(c['activation_scale']['dynamic'])
        self.assertNotIn('input_scale', c['fields'])

    def test_source_mismatch_remains_unknown(self):
        m = instance(aq.WRAPPER)
        m.quant_method = instance(aq.STATIC)
        with patch.object(aq.inspect, 'getfile', return_value=__file__):
            self.assertIsNone(aq.audited_ascend_contract(m, 'runtime'))

    def test_wrong_scheme_not_guessed_from_wrapper(self):
        m = instance(aq.WRAPPER)
        m.quant_method = object()
        self.assertIsNone(aq.audited_ascend_contract(m, 'runtime'))

    def test_no_duplicate_native_methods(self):
        path = Path(aq.__file__).with_name('operators.py')
        tree = ast.parse(path.read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
            self.assertEqual(len(methods), len(set(methods)))


class FactoryGuards(unittest.TestCase):
    def test_other_worker_observer_blocks_identity(self):
        from ascend_fi.inspect.qwen_r1_adapter import factory
        with tempfile.TemporaryDirectory() as td:
            args = {'session_dir': td, 'suite': {}, 'fault_configuration': {'world_size': 2},
                    'plan': {'service_epoch': 'rank0'}, 'probe': {}}
            rows = [{'result': {'service_epoch': 'rank0', 'active': False, 'installed_handles': 0}},
                    {'result': {'service_epoch': 'rank1', 'active': True, 'installed_handles': 1}}]
            with patch('ascend_fi.inspect.qwen_r1_adapter.create_owner', return_value=SimpleNamespace()), \
                 patch('ascend_fi.inspect.qwen_r1_adapter.InspectClient') as client:
                client.return_value.call.return_value = rows
                adapter = factory(args)
                with self.assertRaisesRegex(RuntimeError, 'ALL_WORKERS'):
                    adapter.current_identity()

    def test_validation_output_cannot_escape_session(self):
        from ascend_fi.inspect.qwen_r1_adapter import factory
        with patch('ascend_fi.inspect.qwen_r1_adapter.create_owner', return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ValueError, 'INVALID_VALIDATION_RUN'):
                factory({'session_dir': '.', 'suite': {}, 'validation_run': '../escape'})

    def test_prior_validation_preserved(self):
        from ascend_fi.inspect.qwen_r1_adapter import factory
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / 'inspect_identity/identity_adapter_v1').mkdir(parents=True)
            with patch('ascend_fi.inspect.qwen_r1_adapter.create_owner', return_value=SimpleNamespace()):
                with self.assertRaisesRegex(ValueError, 'PRESERVE_PRIOR'):
                    factory({'session_dir': td, 'suite': {}})


if __name__ == '__main__':
    unittest.main()
