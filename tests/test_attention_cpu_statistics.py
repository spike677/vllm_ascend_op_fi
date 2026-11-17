"""CPU contracts for the explicitly different, fresh-only statistics backend."""
import hashlib
import math
from pathlib import Path
import sys
import tempfile
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from experiments.c2c3_closure_20261117 import attention_cpu_statistics as backend


SOURCE = '''import torch
class TensorStatInfo:
    def __init__(self):
        self.max = self.min = self.mean = self.norm = self.check_sum = None
class PytorchDataProcessor:
    def get_stat_info(self, data, async_dump=False, precision='low'):
        self.delegated += 1
        result = TensorStatInfo()
        if data.numel():
            if data.dtype == torch.bool:
                result.max, result.min = torch.any(data), torch.all(data)
            elif not data.shape:
                result.max = result.min = result.mean = result.norm = data.clone()
        return result
'''


class CpuStatisticsTests(unittest.TestCase):
    def setUp(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = folder/'pytorch_processor.py'
        self.path.write_text(SOURCE, encoding='utf-8')
        self.module = ModuleType(backend.MODULE)
        self.module.__file__ = str(self.path)
        exec(compile(self.path.read_bytes(), str(self.path), 'exec'), self.module.__dict__)
        self.enterContext(patch.dict(sys.modules, {backend.MODULE: self.module}))
        self.enterContext(patch.object(backend, 'REVIEWED_PROCESSOR_SHA256',
                                      hashlib.sha256(self.path.read_bytes()).hexdigest()))
        self.processor = self.module.PytorchDataProcessor()
        self.processor.config = SimpleNamespace(task='statistics', summary_mode='statistics')
        self.processor.tensor_handler = SimpleNamespace(is_empty_data=lambda t:t.is_meta,
            is_batchedtensor=lambda t:False, is_gradtrackingtensor=lambda t:False)
        self.processor.delegated = 0
        self.original = self.module.PytorchDataProcessor.get_stat_info

    def collect(self, tensor):
        return self.processor.get_stat_info(tensor, False, 'high')

    def test_complete_noncontiguous_tensor_is_cpu_fp64_without_mutating_input(self):
        value = torch.tensor([[1e8, 1, -1e8], [3, 4, 0]], dtype=torch.float32).t()
        before = value.clone()
        observer = backend.CpuFp64Statistics().install()
        actual = self.collect(value)
        observer.close()
        self.assertTrue(torch.equal(before, value))
        self.assertAlmostEqual(actual.mean.item(), 8/6)
        self.assertAlmostEqual(actual.norm.item(), math.sqrt(2e16+26))
        self.assertEqual(actual.max.dtype, torch.float64)
        self.assertEqual(actual.max.device.type, 'cpu')
        self.assertEqual(observer.report()['copied_bytes'], value.numel()*value.element_size())
        self.assertEqual(observer.report()['status'], 'NEW_COLLECTION_COMPLETE')
        self.assertEqual(self.processor.delegated, 0)
        self.assertIs(self.module.PytorchDataProcessor.get_stat_info, self.original)

    def test_original_null_scalar_bool_and_nonfinite_values_remain_distinct(self):
        observer = backend.CpuFp64Statistics().install()
        empty = self.collect(torch.empty(0))
        boolean = self.collect(torch.tensor([True, False]))
        scalar = self.collect(torch.tensor(-2.))
        nan = self.collect(torch.tensor([float('nan'), 1.]))
        infinity = self.collect(torch.tensor([float('inf'), 1.]))
        observer.close()
        self.assertIsNone(empty.max)
        self.assertIsNone(boolean.mean)
        self.assertFalse(boolean.min)
        self.assertEqual(scalar.norm.item(), -2.)  # the reviewed collector's scalar convention
        self.assertTrue(math.isnan(nan.mean.item()))
        self.assertTrue(math.isnan(nan.norm.item()))
        self.assertTrue(math.isinf(infinity.mean.item()))
        self.assertTrue(math.isinf(infinity.norm.item()))
        self.assertEqual(self.processor.delegated, 3)

    def test_copy_limit_rejects_complete_tensor_instead_of_sampling(self):
        observer = backend.CpuFp64Statistics().install()
        with patch.object(backend, 'MAX_TENSOR_BYTES', 4):
            with self.assertRaisesRegex(RuntimeError, 'COMPLETE_COPY_BUDGET_EXCEEDED'):
                self.collect(torch.arange(8, dtype=torch.float32))
        with self.assertRaisesRegex(RuntimeError, 'COLLECTION_FAILED'):
            observer.close()
        self.assertEqual(observer.total_bytes, 0)
        self.assertIs(self.module.PytorchDataProcessor.get_stat_info, self.original)
        self.assertEqual(observer.report()['status'], 'NEW_COLLECTION_INCOMPLETE')

    def test_total_copy_limit_and_high_precision_scope_are_enforced(self):
        observer = backend.CpuFp64Statistics().install()
        with patch.object(backend, 'MAX_TOTAL_BYTES', 12):
            self.collect(torch.ones(2))
            with self.assertRaisesRegex(RuntimeError, 'COMPLETE_COPY_BUDGET_EXCEEDED'):
                self.collect(torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, 'SYNCHRONOUS_HIGH_STATISTICS_REQUIRED'):
            self.processor.get_stat_info(torch.ones(2), True, 'high')
        with self.assertRaisesRegex(RuntimeError, 'COLLECTION_FAILED'):
            observer.close()
        self.assertEqual(observer.total_bytes, 8)

    def test_changed_source_code_or_globals_cannot_be_installed(self):
        self.path.write_text(SOURCE+'# changed\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'SOURCE_UNREVIEWED'):
            backend.CpuFp64Statistics().install()
        self.path.write_text(SOURCE, encoding='utf-8')
        self.module.PytorchDataProcessor.get_stat_info = FunctionType(self.original.__code__, dict(self.module.__dict__))
        with self.assertRaisesRegex(RuntimeError, 'METHOD_UNPROVEN'):
            backend.CpuFp64Statistics().install()

    def test_foreign_method_is_never_overwritten_during_cleanup(self):
        observer = backend.CpuFp64Statistics().install()
        foreign = lambda *args, **kwargs: None
        self.module.PytorchDataProcessor.get_stat_info = foreign
        with self.assertRaisesRegex(RuntimeError, 'OWNERSHIP_CHANGED'):
            observer.close()
        self.assertIs(self.module.PytorchDataProcessor.get_stat_info, foreign)
        self.assertFalse(observer.report()['original_method_restored'])


if __name__ == '__main__':
    unittest.main()
