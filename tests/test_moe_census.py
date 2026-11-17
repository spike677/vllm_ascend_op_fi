"""CPU-only proof that a census sees the real API inside an opaque wrapper."""
from dataclasses import dataclass
import sys
import unittest

import torch
from test_route8_index22 import LIB  # Explicit CPU implementation, never NPU evidence.
from fault_injection.moe_census import NaturalMoETrace
from fault_injection.tensor_observer import NaturalTensorObserver


@dataclass
class CombineMetadata:
    expanded_row_idx: torch.Tensor
    topk_weights: torch.Tensor


class Dispatcher:
    def token_combine(self, hidden_states, combine_metadata):
        return torch.ops.npu.npu_moe_token_unpermute(
            hidden_states, combine_metadata.expanded_row_idx.abs(), combine_metadata.topk_weights)


DISPATCHER = Dispatcher()
OUTER = torch.library.Library('moe_census_cpu_fixture', 'DEF')
OUTER.define('opaque(Tensor hidden, Tensor indices, Tensor weights) -> Tensor')

def opaque_cpu(hidden, indices, weights):
    return DISPATCHER.token_combine(hidden, CombineMetadata(indices, weights))

OUTER.impl('opaque', opaque_cpu, 'CPU')


class Model(torch.nn.Module):
    def forward(self, hidden, indices, weights):
        return torch.ops.moe_census_cpu_fixture.opaque(hidden, indices, weights) + 1


class MoECensusTests(unittest.TestCase):
    def test_actual_unpermute_is_visible_inside_opaque_call_and_not_reexecuted(self):
        model = Model().eval()
        args = (torch.arange(24.).reshape(6, 4), torch.arange(6, dtype=torch.int32),
                torch.arange(6.).reshape(3, 2)/10)
        baseline = model(*args)
        trace = NaturalMoETrace(model, [(Dispatcher.token_combine, 'native_combine')],
                               {}, simulation=True)
        with trace, NaturalTensorObserver(model) as observer:
            for request in ('real-cpu-forward-1', 'real-cpu-forward-2'):
                trace.request_id = request
                self.assertTrue(torch.equal(model(*args), baseline))
        report = trace.report()
        self.assertTrue(report['complete'], report['errors'])
        self.assertEqual(report['observed_unpermute_calls'], 2)
        self.assertEqual(report['observed_unpermute_per_request'],
                         {'real-cpu-forward-1': 1, 'real-cpu-forward-2': 1})
        native = [row for row in report['native_operators'] if 'npu_moe_token_unpermute(' in row['schema']]
        self.assertTrue(all(row['original_calls'] == 1 for row in native))
        self.assertEqual(native[0]['input'][2]['object_id'], id(args[2]))
        self.assertEqual(report['python_boundaries'][0]['input']['combine_metadata']['topk_weights']['object_id'],
                         native[0]['input'][2]['object_id'])
        abs_row = next(row for row in report['native_operators'] if row['operator'] == 'aten.abs.default')
        self.assertEqual(abs_row['input'][0]['object_id'], id(args[1]))
        self.assertEqual(abs_row['output'][0]['object_id'], native[0]['input'][1]['object_id'])
        self.assertTrue(observer.report()['complete'])
        self.assertIsNone(sys.getprofile())
        self.assertFalse(report['qualification_granted'])
        self.assertFalse(report['complete_permutation_value_proven'])

    def test_foreign_profiler_is_preserved(self):
        trace = NaturalMoETrace(Model(), [], {}, simulation=True)
        foreign = lambda *args: None
        sys.setprofile(foreign)
        try:
            with self.assertRaisesRegex(RuntimeError, 'FOREIGN_PYTHON_PROFILE'):
                trace.__enter__()
            self.assertIs(sys.getprofile(), foreign)
        finally:
            sys.setprofile(None)


if __name__ == '__main__':
    unittest.main()
