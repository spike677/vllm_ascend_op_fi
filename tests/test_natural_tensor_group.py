"""Real CPU forwards exercise bounded independent group admission and recovery."""
import unittest
import torch
from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural import NaturalTensorGroupProvider
from test_rmsnorm_natural import Network
from test_route8_index22 import RopeWorkload


class NaturalTensorGroupTests(unittest.TestCase):
    def provider(self, model, mechanism, inputs):
        return NaturalTensorGroupProvider(model.eval(), mechanism, replay=lambda m, phase: m(*inputs),
            synchronize=lambda: None, simulation=True, workload_sha256='a'*64,
            environment_sha256='b'*64, model_weights_sha256='c'*64, model_source_sha256='d'*64)

    def test_actual_multinorm_c2_c3_observer_and_same_model_recovery(self):
        model = Network().eval(); x = torch.arange(16.).reshape(2, 8)
        provider = self.provider(model, '18.api_half', (x,))
        qualification = provider.qualify()
        self.assertEqual(qualification['candidate_set']['eligible'], 2)
        c2 = provider.run()
        self.assertTrue(c2['contract_satisfied'], c2)
        proof = provider.group.c2_proof()
        c3 = provider.run('every_match', forwards=2, c2_proof=proof)
        self.assertTrue(c3['contract_satisfied'], c3)
        self.assertTrue(c3['recovery']['complete'])
        self.assertEqual(c3['observed_eligible_calls'], 4)
        self.assertFalse(c3['public_qualification_granted'])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_rope_actual_lookups_each_match_then_exact_off(self):
        class Two(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.left = RopeWorkload(); self.right = RopeWorkload()
            def forward(self, x, indices):
                return self.left(x, indices) + self.right(x + 1, indices)
        model = Two().eval(); inputs = (torch.arange(16.).reshape(4, 4), torch.arange(4))
        provider = self.provider(model, '22.wrong_row', inputs)
        provider.qualify()
        c2 = provider.run()
        self.assertTrue(c2['contract_satisfied'], c2)
        proof = provider.group.c2_proof()
        c3 = provider.run('every_match', forwards=2, c2_proof=proof)
        self.assertTrue(c3['contract_satisfied'], c3)
        self.assertEqual(c3['observed_eligible_calls'], 4)


if __name__ == '__main__':
    unittest.main()
