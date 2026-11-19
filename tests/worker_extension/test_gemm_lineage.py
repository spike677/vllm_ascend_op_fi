"""Real CPU custom operators; fixture source hashes never attest an NPU run."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import gemm_lineage as lineage
from demoncat.runtime.vllm_ascend.worker_extension.serving39_coverage import public_pair


PATCH_SOURCE = '''import torch
def unquantized_gemm(x, weight, bias=None):
    return torch.nn.functional.linear(x, weight, bias)
def default_unquantized_gemm(layer, x, weight, bias=None):
    return torch.ops.vllm.unquantized_gemm(x, weight, bias)
'''


class GemmLineageTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        patch_path = self.folder / 'patch_unquantized_gemm.py'
        patch_path.write_text(PATCH_SOURCE)
        utils_path = self.folder / 'utils.py'
        utils_path.write_text('# Explicit CPU source-identity fixture.\n')
        self.module = ModuleType(lineage.PATCH_MODULE)
        self.module.__file__ = str(patch_path)
        exec(compile(patch_path.read_bytes(), str(patch_path), 'exec'), self.module.__dict__)
        utils = ModuleType(lineage.UTILS_MODULE)
        utils.__file__ = str(utils_path)
        utils.default_unquantized_gemm = self.module.default_unquantized_gemm
        register_path = self.folder / 'torch_utils.py'
        register_path.write_text('import torch\nvllm_lib = torch.library.Library("vllm", "FRAGMENT")\n'
                                 'def direct_register_custom_op():\n    pass\n')
        register = ModuleType(lineage.REGISTER_MODULE)
        register.__file__ = str(register_path)
        exec(compile(register_path.read_bytes(), str(register_path), 'exec'), register.__dict__)
        self.library = register.vllm_lib
        self.stack.callback(self.library._destroy)
        self.library.define('unquantized_gemm(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor')
        for key in ('CPU', 'PrivateUse1'):
            self.library.impl('unquantized_gemm', lambda x, weight, bias=None:
                              torch.nn.functional.linear(x, weight, bias), key)
        self.module.direct_register_custom_op = register.direct_register_custom_op
        self.stack.enter_context(patch.dict(sys.modules, {
            lineage.PATCH_MODULE: self.module, lineage.UTILS_MODULE: utils,
            lineage.REGISTER_MODULE: register,
            'torch_npu': NS(get_npu_format=lambda tensor: 2)}))
        from fault_injection.cpu_build_identity import cpu_test_build_identity
        self.stack.enter_context(cpu_test_build_identity(
            name='CPU_FIXTURE_GEMM_SOURCE_AND_DISPATCH', torch_version=torch.__version__,
            torch_npu_version='CPU_ONLY', source_hashes={
                lineage.PATCH_MODULE:hashlib.sha256(patch_path.read_bytes()).hexdigest(),
                lineage.UTILS_MODULE:hashlib.sha256(utils_path.read_bytes()).hexdigest(),
                lineage.REGISTER_MODULE:hashlib.sha256(register_path.read_bytes()).hexdigest()}, registrations={}))
        self.function = torch.ops.vllm.unquantized_gemm.default
        self.binding = lineage.ReviewedGemmBinding()
        self.x = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        self.weight = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.bfloat16), requires_grad=False)

    def tearDown(self):
        self.stack.close()

    def pair(self, args=(), kwargs=None, expected_weight=None):
        return public_pair(str(self.function), args, kwargs or {}, function=self.function,
                           gemm_binding=self.binding, expected_weight=self.weight if expected_weight is None else expected_weight)

    def test_actual_custom_operator_matches_owned_transposed_storage_without_copy(self):
        pair = self.pair((self.x, self.weight))
        a, b, values = pair
        self.assertIs(a, self.x)
        self.assertEqual(b.untyped_storage().data_ptr(), self.weight.untyped_storage().data_ptr())
        self.assertEqual(list(b.shape), [4, 3])
        self.assertEqual(b.stride(), (1, 4))
        self.assertTrue(torch.equal(torch.mm(a, b), self.function(self.x, self.weight)))
        self.assertEqual(values[4:7], [2, 4, 3])
        self.assertEqual(self.binding.last_proof['schema'], str(self.function._schema))
        self.assertTrue(self.binding.last_proof['source_and_loaded_code_verified'])
        self.assertIs(self.pair(kwargs=dict(x=self.x, weight=self.weight, bias=None))[0], self.x)

    def test_unowned_weight_bias_and_unsupported_geometry_remain_unqualified(self):
        self.assertIsNone(self.pair((self.x, self.weight.clone())))
        self.assertIsNone(self.binding.last_proof)
        self.assertIsNone(self.pair((self.x, self.weight, torch.ones(3, dtype=torch.bfloat16))))
        self.assertIsNone(self.pair((self.x.float(), self.weight)))
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_PUBLIC_GEMM_ARGUMENT'):
            self.pair((self.x, self.weight), dict(x=self.x))

    def test_loaded_code_source_and_schema_drift_fail_closed(self):
        self.pair((self.x, self.weight))
        original = self.module.unquantized_gemm
        self.module.unquantized_gemm = lambda x, weight, bias=None: x
        with self.assertRaisesRegex(RuntimeError, 'IMPLEMENTATION_CHANGED'):
            self.pair((self.x, self.weight))
        self.module.unquantized_gemm = original
        source = Path(self.module.__file__)
        source.write_text(source.read_text() + '# source drift\n')
        with self.assertRaisesRegex(RuntimeError, 'UNREVIEWED_PUBLIC_GEMM_SOURCE'):
            self.pair((self.x, self.weight))
        with self.assertRaisesRegex(RuntimeError, 'UNREVIEWED_PUBLIC_GEMM_SCHEMA'):
            self.binding.arguments(torch.ops.aten.mm.default, (self.x, self.weight.t()), {}, self.weight)

    def test_whole_request_observer_binds_both_actual_custom_gemm_modules(self):
        import test_serving39_coverage as hooks

        class CustomLinear(torch.nn.Linear):
            def forward(self, x):
                return torch.ops.vllm.unquantized_gemm(x, self.weight)

        class CustomModel(hooks.TwoProjectionModel):
            def __init__(self):
                super().__init__()
                self.projection_a = CustomLinear(4, 4, bias=False, dtype=torch.bfloat16)
                self.projection_b = CustomLinear(4, 4, bias=False, dtype=torch.bfloat16)

        fixture = hooks.WholeRequestHookTests()
        fixture.setUp()
        try:
            with patch.object(hooks, 'TwoProjectionModel', CustomModel):
                session = fixture.session()
            session.expect(fixture.actual_request_id)
            hidden = hooks.life.ServingSessionLifecycleTests.root(fixture, session)
            session.processor(session.head, hidden[-1:].contiguous())
            rows = [row for row in session.backend.calls
                    if row['row']['operator'] == 'vllm.unquantized_gemm.default']
            self.assertEqual([row['row']['module_path'] for row in rows], ['projection_a', 'projection_b'])
            self.assertTrue(all(row['pair'] is not None for row in rows))
            self.assertEqual(len({row['row']['public_lowering']['module_weight_object_identity'] for row in rows}), 2)
            session.close()
            self.assertEqual(session.backend.owned_hooks, 0)
        finally:
            fixture.tearDown()
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
