"""CPU contracts for the real opaque-boundary protocol, never NPU evidence."""
from dataclasses import dataclass, field
import unittest
from unittest.mock import patch
import torch

from test_route8_index22 import LIB
import test_serving_tensor_group as host_fixture
from fault_injection.moe_census import NaturalMoETrace
from demoncat.runtime.vllm_ascend.worker_extension.serving_moe import ServingMoEGroup


@dataclass
class Routing:
    apply_router_weight_on_input: bool = False

@dataclass
class ExpertInput:
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    routing: Routing = field(default_factory=Routing)

@dataclass
class Combine:
    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor
    restore_shape: tuple

@dataclass
class Dispatched:
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    dynamic_scale: torch.Tensor
    combine_metadata: Combine

@dataclass
class MlpInput:
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    dynamic_scale: torch.Tensor

@dataclass
class Routed:
    routed_out: torch.Tensor

def select_experts(hidden_states, router_logits, top_k):
    count = hidden_states.shape[0]
    return torch.arange(1., count*top_k+1).reshape(count, top_k)/10, torch.zeros(count, top_k, dtype=torch.int32)

class Dispatcher:
    def token_dispatch(self, token_dispatch_input):
        x = token_dispatch_input.hidden_states; k = token_dispatch_input.topk_weights.shape[1]
        n = x.shape[0]*k
        return Dispatched(x.repeat_interleave(k, dim=0).to(torch.int8), torch.tensor([n]), torch.ones(n),
            Combine(token_dispatch_input.topk_weights, torch.arange(n, dtype=torch.int32).flip(0), tuple(x.shape)))
    def token_combine(self, hidden_states, combine_metadata):
        return torch.ops.npu.npu_moe_token_unpermute(hidden_states, combine_metadata.expanded_row_idx.abs(),
                                                  combine_metadata.topk_weights)

class Pipeline:
    def __init__(self):
        self.dispatcher = Dispatcher()
    def fused_experts(self, fused_experts_input):
        dispatched = self.dispatcher.token_dispatch(fused_experts_input)
        hidden = self._apply_mlp(MlpInput(dispatched.hidden_states, dispatched.group_list, dispatched.dynamic_scale))
        return Routed(self.dispatcher.token_combine(hidden, dispatched.combine_metadata))
    def _apply_mlp(self, mlp_compute_input):
        return mlp_compute_input.hidden_states.float() * .75 + .125

REGISTRY = {}
OUTER = torch.library.Library('moe_group_cpu_fixture', 'DEF')
OUTER.define('opaque(Tensor hidden, Tensor router_logits, int key) -> Tensor')
def opaque_cpu(hidden, router_logits, key):
    return REGISTRY[key].forward_impl(hidden, router_logits)
OUTER.impl('opaque', opaque_cpu, 'CPU')

class Expert(torch.nn.Module):
    def __init__(self, key):
        super().__init__(); self.top_k = 2; self.key = key; self.pipeline = Pipeline(); REGISTRY[key] = self
    def forward(self, hidden, router_logits):
        return torch.ops.moe_group_cpu_fixture.opaque(hidden, router_logits, self.key)
    def forward_impl(self, hidden, router_logits):
        weights, ids = select_experts(hidden, router_logits, self.top_k)
        return self.pipeline.fused_experts(ExpertInput(hidden, weights, ids)).routed_out + hidden * .125

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.first = Expert(1); self.last = Expert(2); self.processor = host_fixture.Processor()
        self.register_buffer('embedding', torch.arange(64.).reshape(8, 8)/8)
    def forward(self, input_ids, positions):
        hidden = self.embedding[input_ids]; logits = hidden[:, :4]
        return self.last(self.first(hidden, logits), logits)
    def logits(self, hidden):
        return self.processor(hidden[torch.tensor([hidden.shape[0]-1])])

def template(model):
    return NaturalMoETrace(model, [(select_experts, 'topk_producer'), (Pipeline.fused_experts, 'expert_pipeline'),
        (Pipeline._apply_mlp, 'expert_mlp'), (Dispatcher.token_dispatch, 'routing_producer'),
        (Dispatcher.token_combine, 'native_combine')], {}, simulation=True)

class ServingMoEGroupTests(unittest.TestCase):
    def setUp(self):
        self.host = host_fixture.ServingTensorGroupTests(); self.host.setUp(); self.host.session.close()
        self.host.model = Model().eval(); self.host.worker.model_runner.model = self.host.model
        self.host.session = ServingMoEGroup(self.host.worker, prompt=[1, 2, 3], weight_manifest={'cpu_fixture': True},
            workload_sha256='a'*64, environment_sha256='b'*64, _simulation=True,
            _synchronize=lambda: None, _trace_factory=template)
        self.session = self.host.session
    def tearDown(self):
        self.host.tearDown(); REGISTRY.clear()
    def healthy(self):
        for phase, arm in (('native_baseline', None), ('baseline', 'baseline'),
                           ('baseline_repeat', 'baseline'), ('lossless', 'baseline')):
            self.host.request(phase, arm)
        return self.session.qualify()
    def test_all_natural_pipeline_targets_c2_c3_and_original_dtype_recovery(self):
        found = self.healthy()
        self.assertEqual(found['candidate_set']['eligible'], 2)
        self.session.provider.open_group('once', forwards=1)
        fault, _ = self.host.request('c2_fault', 'fault'); self.host.request('c2_off')
        c2 = self.session.finish_group()
        self.assertTrue(c2['contract_satisfied'], c2['rejection_reasons'])
        self.assertEqual(c2['hits'], 2)
        trace = fault['independent']['moe_trace']
        self.assertTrue(trace['complete'])
        self.assertEqual(trace['observed_unpermute_calls'], 2)
        self.assertTrue(all(p['actual_routed_return_same_object'] for p in trace['producer_joins']))
        self.assertTrue(all(e['complete_native_output_oracle_exact'] and e['observed_original_native_calls'] == 1
                            for e in fault['mechanism']['events']))
        proof = self.session.provider.group.c2_proof()
        self.session.provider.open_group('every_match', forwards=2, c2_proof=proof)
        self.host.request('c3_fault_0', 'fault'); self.host.request('c3_fault_1', 'fault'); self.host.request('c3_off')
        c3 = self.session.finish_group()
        self.assertTrue(c3['contract_satisfied'], c3['rejection_reasons'])
        self.assertEqual(c3['hits'], 4)
        self.assertFalse(c3['real_worker_evidence']); self.assertFalse(c3['public_qualification_granted'])
        self.session.close()

    def test_incomplete_real_permutation_refuses_before_native_remap(self):
        original = Dispatcher.token_dispatch
        def incomplete(dispatcher, token_dispatch_input):
            result = original(dispatcher, token_dispatch_input)
            result.combine_metadata.expanded_row_idx[0] = result.combine_metadata.expanded_row_idx[1]
            return result
        with patch.object(Dispatcher, 'token_dispatch', incomplete):
            with self.assertRaisesRegex(RuntimeError, 'complete contribution permutation'):
                self.host.request('baseline', 'baseline')
        self.assertIsNone(self.session.provider.candidates)
        self.session.abort()

    def test_structural_module_missing_from_workload_cannot_shrink_denominator(self):
        self.session.close()
        self.host.model.unused = Expert(3)
        self.host.session = ServingMoEGroup(self.host.worker, prompt=[1, 2, 3], weight_manifest={'cpu_fixture': True},
            workload_sha256='a'*64, environment_sha256='b'*64, _simulation=True,
            _synchronize=lambda: None, _trace_factory=template)
        self.session = self.host.session
        with self.assertRaisesRegex(RuntimeError, 'STRUCTURAL_DENOMINATOR_INCOMPLETE'):
            self.healthy()
        self.assertIsNone(self.session.provider.candidates)
        self.session.close()

    def test_changed_probability_producer_is_refused_before_any_fault(self):
        original = Dispatcher.token_dispatch
        def replaced(dispatcher, token_dispatch_input):
            result = original(dispatcher, token_dispatch_input)
            result.combine_metadata.topk_weights = result.combine_metadata.topk_weights.clone()
            return result
        with patch.object(Dispatcher, 'token_dispatch', replaced):
            with self.assertRaisesRegex(RuntimeError, 'ORIGINAL_TOPK_PROBABILITY_PRODUCER_IDENTITY_REQUIRED'):
                self.host.request('baseline', 'baseline')
        report = self.session.abort()
        self.assertEqual(report['failed_phase']['mechanism']['events'], [])
        self.assertFalse(report['failed_request_recovery_claimed'])

if __name__ == '__main__':
    unittest.main()
