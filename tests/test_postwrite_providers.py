"""Real CPU tensor/operator contract tests. These grant no NPU qualification."""
import unittest

import torch

from fault_injection.deployments.postwrite import (
    PostwriteProvider, PostwriteRefused, _storage, _validate_tensor,
)


class CpuTransport:
    transport_kind = 'cpu_test'

    def validate(self, tensor):
        if tensor.device.type != 'cpu' or tensor.dtype != torch.bfloat16:
            raise PostwriteRefused('CPU_TEST_BF16_ONLY')
        _validate_tensor(tensor)

    def open(self, tensor, owner, generation):
        self.validate(tensor)
        return CpuLease(tensor)


class CpuLease:
    def __init__(self, tensor):
        self.tensor = tensor
        self.storage = _storage(tensor)
        self.original = tensor.clone()
        self.expected = self.original.clone()
        self.closed = False

    def apply(self, mode, indices):
        if not torch.isfinite(self.tensor).all():
            raise PostwriteRefused('NONFINITE_ORIGINAL')
        flat = self.tensor.view(-1)
        if mode in (1, 2, 3):
            flat[list(indices)] = {1: float('inf'), 2: -float('inf'), 3: float('nan')}[mode]
        elif mode == 4:
            flat[list(indices)] = -flat[list(indices)]
        self.expected = self.tensor.clone()
        return len(indices) if mode else 0

    def recover(self):
        with torch.inference_mode():
            # Equal including NaN payloads, since qNaN is a legitimate proxy mode.
            if not torch.equal(self.tensor.view(torch.uint8), self.expected.view(torch.uint8)):
                raise PostwriteRefused('EXTERNAL_WRITE')
            self.tensor.copy_(self.original)
        self.closed = True

    def live(self):
        return not self.closed


class Projection(torch.nn.Module):
    def __init__(self, consumer=True, weight=1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4, 4), weight, dtype=torch.bfloat16), requires_grad=False)
        self.consumer = consumer

    def forward(self, x):
        out = torch.mm(x, self.weight)
        return out + 1 if self.consumer else out


class Cache(torch.nn.Module):
    def __init__(self, external_write=False):
        super().__init__()
        self.register_buffer('state', torch.zeros(3, 4, dtype=torch.bfloat16))
        self.external_write = external_write

    def forward(self, source, rows):
        self.state.index_copy_(0, rows, source)
        out = self.state + 1
        if self.external_write:
            self.state.zero_()
        return out


class PostwriteTests(unittest.TestCase):
    def test_actual_matmul_lossless_fault_off_and_restore(self):
        model = Projection()
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        provider = PostwriteProvider(model, CpuTransport(), '39.output_proxy')
        selected = provider.qualify(x)
        self.assertEqual(selected['operator'], 'aten.mm.default')
        self.assertIn('aten.add.Tensor', selected['consumers'])
        baseline, b = provider.run('baseline', x)
        fault, f = provider.run('fault', x)
        off, o = provider.run('off', x)
        self.assertEqual(torch.isinf(fault).sum().item(), 1)
        self.assertTrue(torch.equal(baseline, off))
        self.assertEqual(f['intervention']['changed_elements'], 1)
        self.assertEqual(f['new_fault_hits'], 1)
        self.assertEqual(o['new_fault_hits'], 0)
        self.assertTrue(f['recovery_confirmed'])
        self.assertFalse(f['public_worker_qualified'])
        with self.assertRaisesRegex(PostwriteRefused, 'ONCE_GATE'):
            provider.run('fault', x)

    def test_same_cache_storage_target_slot_and_non_target_preserved(self):
        model = Cache()
        source = torch.arange(1, 5, dtype=torch.bfloat16).reshape(1, 4)
        rows = torch.tensor([1])
        p = PostwriteProvider(model, CpuTransport(), '41.tail_column', tail_elements=2)
        selected = p.qualify(source, rows)
        self.assertEqual(selected['operator'], 'aten.index_copy_.default')
        baseline, _ = p.run('baseline', source, rows)
        before = model.state.clone()
        fault, f = p.run('fault', source, rows)
        self.assertTrue(torch.equal(model.state, before))
        self.assertTrue(torch.equal(fault[[0, 2]], baseline[[0, 2]]))
        self.assertTrue(torch.equal(fault[1, :2], baseline[1, :2]))
        self.assertTrue(torch.equal(fault[1, 2:], torch.tensor([-2, -3], dtype=torch.bfloat16)))
        off, o = p.run('off', source, rows)
        self.assertTrue(torch.equal(baseline, off))
        self.assertEqual(f['intervention']['selected_elements'], 2)
        self.assertEqual(o['new_fault_hits'], 0)

    def test_non_consumed_output_not_eligible(self):
        p = PostwriteProvider(Projection(False), CpuTransport(), '39.output_proxy')
        with self.assertRaisesRegex(PostwriteRefused, 'NO_STABLE'):
            p.qualify(torch.ones(1, 4, dtype=torch.bfloat16))

    def test_changed_request_and_model_refused(self):
        m = Projection()
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        p = PostwriteProvider(m, CpuTransport(), '39.output_proxy')
        p.qualify(x)
        with self.assertRaisesRegex(PostwriteRefused, 'INPUT_CHANGED'):
            p.run('baseline', x + 1)
        with torch.no_grad():
            m.weight.add_(1)
        with self.assertRaisesRegex(PostwriteRefused, 'IDENTITY'):
            p.run('baseline', x)

    def test_lossless_required_and_mutable_storage_refused(self):
        m = Cache(True)
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        rows = torch.tensor([1])
        p = PostwriteProvider(m, CpuTransport(), '41.tail_column')
        p.qualify(x, rows)
        with self.assertRaisesRegex(PostwriteRefused, 'LOSSLESS'):
            p.run('fault', x, rows)
        with self.assertRaisesRegex(PostwriteRefused, 'WRITE_AFTER'):
            p.run('baseline', x, rows)
        self.assertEqual(p.query()['leases'], 0)

    def test_nonfinite_original_refused_before_mutation(self):
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        p = PostwriteProvider(Projection(weight=float('inf')), CpuTransport(), '39.output_proxy')
        p.qualify(x)
        with self.assertRaisesRegex(PostwriteRefused, 'NONFINITE'):
            p.run('baseline', x)


if __name__ == '__main__':
    unittest.main()
