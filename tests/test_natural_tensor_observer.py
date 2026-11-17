"""Independent observer contracts; CPU only, no natural/NPU qualification."""
import unittest
import torch
from fault_injection.tensor_observer import NaturalTensorObserver


class TwoNorms(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.RMSNorm(8)
        self.second = torch.nn.RMSNorm(8)
    def forward(self, x):
        return self.second(self.first(x) + 1) * 2


class WriteChain(torch.nn.Module):
    def forward(self, x):
        base = x.clone()
        view = torch.as_strided(base, (2, 2), (4, 1), 1)
        view.copy_(torch.ones_like(view))
        return torch.cat((base, base), dim=0)


class NaturalTensorObserverTests(unittest.TestCase):
    def test_same_shape_modules_have_separate_independent_consumers(self):
        model = TwoNorms().eval(); x = torch.arange(16.).reshape(2, 8)
        baseline = model(x)
        with NaturalTensorObserver(model) as observer:
            first, second = model(x), model(x)
        self.assertTrue(torch.equal(baseline, first) and torch.equal(first, second))
        report = observer.report()
        self.assertTrue(report['complete'])
        rows = [row for row in report['events'] if row['kind'] == 'module_return']
        self.assertEqual([row['path'] for row in rows], ['first', 'second', 'first', 'second'])
        self.assertTrue(all(row['consumers'] for row in rows))
        self.assertEqual(report['tensor_mutations'], 0)
        self.assertEqual(report['observer_hooks'], 0)
        self.assertEqual(report['inflight'], 0)
        self.assertFalse(report['qualification_granted'])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_actual_view_copy_base_consumer_no_added_writes(self):
        model = WriteChain().eval(); x = torch.arange(16.).reshape(4, 4)
        before = x.clone(); baseline = model(x)
        with NaturalTensorObserver(model) as observer:
            actual = model(x)
        self.assertTrue(torch.equal(actual, baseline) and torch.equal(before, x))
        rows = [row for row in observer.report()['events'] if row['kind'] == 'view']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['writes'][0]['operator'], 'aten.copy_.default')
        self.assertEqual(rows[0]['consumers'][0]['operator'], 'aten.cat.default')

    def test_model_exception_unwinds_dispatch_and_hooks(self):
        class Broken(TwoNorms):
            def forward(self, x):
                super().forward(x)
                raise RuntimeError('intentional')
        model = Broken().eval()
        observer = NaturalTensorObserver(model)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with observer:
                model(torch.ones(8))
        self.assertFalse(observer.report()['complete'])
        self.assertEqual(observer.report()['inflight'], 0)
        self.assertFalse(model._forward_hooks)


if __name__ == '__main__':
    unittest.main()
