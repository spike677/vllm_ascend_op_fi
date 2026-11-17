"""Executed CPU contracts; canonical source and real return/consumer boundaries."""
import unittest
import torch
from fault_injection.rmsnorm_return import RMSNormReturnSession, eager_rms_contract

LIB = torch.library.Library('npu', 'FRAGMENT')
LIB.define('npu_add_rms_norm(Tensor input, Tensor residual, Tensor gamma, float epsilon) -> (Tensor, Tensor)')
def fused_cpu(input, residual, gamma, epsilon):
    total = input + residual
    normalized = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + epsilon)
    return normalized * gamma, total
LIB.impl('npu_add_rms_norm', fused_cpu, 'CPU')


class FusedNorm(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.ones(8))
    def forward(self, x, residual):
        return torch.ops.npu.npu_add_rms_norm(x, residual, self.weight, 1e-6)


class CanonicalNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.variance_epsilon = 1e-6
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeceptiveNorm(CanonicalNorm):
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return 2 * self.weight * hidden_states.to(input_dtype)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = CanonicalNorm(8)
        self.right = CanonicalNorm(8)
    def forward(self, x):
        return self.left(x) + self.right(x * 2)


class NaturalRMSNormTests(unittest.TestCase):
    def test_serving_span_tracks_actual_last_token_gemm_without_post_root_injection(self):
        class Hidden(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.norm = CanonicalNorm(8)
            def forward(self, x):
                return self.norm(x)
        model = Hidden().eval(); x = torch.arange(24.).reshape(3, 8)
        baseline = model(x)
        with RMSNormReturnSession(model, arm='fault', simulation=True,
                                  observe_post_root_consumers=True) as session:
            hidden = model(x)
            last = hidden[torch.tensor([2])]
            logits = torch.mm(last, torch.eye(8))
            # An unrelated normalizer outside the actual root is untouched.
            outside = model.norm(x)
        event, = session.report()['events']
        self.assertTrue(torch.equal(outside, baseline))
        self.assertTrue(torch.equal(logits, baseline[-1:] * .5))
        self.assertTrue(any(row['post_root'] and row['operator'] == 'aten.mm.default'
                            for row in event['consumers']))
        self.assertIn('aten.index.Tensor', event['lineage_operators'])

    def test_fused_return_lineage_halves_only_normalized_output(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.norm = FusedNorm()
            def forward(self, x, residual):
                y, live_residual = self.norm(x, residual)
                return y * 3, live_residual
        model = Model().eval(); x = torch.arange(8.); residual = torch.ones(8)
        before = model(x, residual)
        with RMSNormReturnSession(model, arm='fault', simulation=True) as session:
            after = model(x, residual)
        self.assertTrue(torch.equal(after[0], before[0] * .5))
        self.assertTrue(torch.equal(after[1], before[1]))
        row, = session.report()['events']
        self.assertEqual(row['native_calls'], 1)
        self.assertTrue(row['auxiliary_return_storage_unchanged'])
        self.assertTrue(row['consumers'])

    def test_source_contract_rejects_similar_incorrect_normalizer(self):
        self.assertIsNotNone(eager_rms_contract(CanonicalNorm(8)))
        self.assertIsNone(eager_rms_contract(DeceptiveNorm(8)))

    def test_multiple_natural_return_gain_consumers_and_off(self):
        model = Network().eval(); x = torch.arange(16.).reshape(2, 8)
        native = model(x)
        with RMSNormReturnSession(model, arm='baseline', simulation=True) as baseline:
            lossless = model(x)
        self.assertTrue(torch.equal(native, lossless))
        rows = baseline.report()['events']
        self.assertEqual([row['path'] for row in rows], ['left', 'right'])
        self.assertTrue(all(row['consumers'] for row in rows))
        selected = [(row['path'], row['ordinal']) for row in rows]
        with RMSNormReturnSession(model, arm='fault', selected_calls=selected, simulation=True) as fault:
            changed = model(x)
        self.assertTrue(torch.equal(changed, native * .5))
        hits = fault.report()['events']
        self.assertTrue(all(row['applied'] and row['return_readback_exact'] for row in hits))
        self.assertTrue(all(not row['native_reduce_error_claimed'] for row in hits))
        self.assertTrue(torch.equal(model(x), native))
        self.assertEqual(fault.runtime_state()['leases'], 0)


if __name__ == '__main__':
    unittest.main()
