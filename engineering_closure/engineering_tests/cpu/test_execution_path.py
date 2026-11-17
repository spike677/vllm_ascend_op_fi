import json
import unittest

from engineering_closure.path_discovery import discover_execution_path, observe_execution_path


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch CPU environment required')
class ExecutionPathTests(unittest.TestCase):
    def test_live_discovery_records_all_operator_details_and_cleans_up(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.activation = torch.nn.ReLU()
                self.unused = torch.nn.ReLU()

            def forward(self, value):
                return self.activation(value + 2)

        model = Model()
        source = torch.tensor([[-3.0, 2.0]])
        output, report = discover_execution_path(model, lambda: model(source))
        self.assertEqual(output.tolist(), [[0.0, 4.0]])
        self.assertEqual(report['context']['scope_kind'], 'MODEL_LOCAL')
        self.assertEqual(report['forwards'][0]['request_status'], 'NOT_VISIBLE')
        self.assertTrue(report['all_dispatch_details'])
        nodes = {node['id']: node for node in report['nodes']}
        self.assertTrue(any(node.get('name') == 'aten.add.Tensor' and
                            node.get('parameters', {}).get('positional', [None, None])[1] == 2
                            for node in nodes.values()))
        self.assertTrue(any(node.get('name') == 'aten.relu.default' and node.get('tensors')
                            for node in nodes.values()))
        self.assertTrue(any(edge['target'] == 'layer:unused' and edge['status'] == 'NOT_EXECUTED'
                            for edge in report['edges']))
        self.assertFalse(any(edge['target'].startswith('native:') and edge['status'] == 'OBSERVED'
                             for edge in report['edges']))
        self.assertTrue(any(edge['source'].startswith('output:') and
                            edge['evidence'] == 'SAME_LIVE_TENSOR_OBJECT_READ_BY_DISPATCH'
                            for edge in report['edges']))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model.activation._forward_hooks), 0)
        json.dumps(report, allow_nan=False)

    def test_budget_failure_preserves_partial_report_and_original_exception(self):
        class Model(torch.nn.Module):
            def forward(self, value):
                return (value + 1) * 2

        model = Model()
        source = torch.tensor([1.0])
        with self.assertRaisesRegex(RuntimeError, 'INDEPENDENT_TENSOR_OBSERVER_BUDGET_EXHAUSTED'):
            with observe_execution_path(model, max_events=1) as capture:
                model(source)
        self.assertFalse(capture['report']['observation_complete'])
        self.assertEqual(capture['report']['dropped_calls'], 1)
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._forward_hooks), 0)
