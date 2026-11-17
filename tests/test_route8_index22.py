"""CPU dispatch contracts for rebuilt mechanism code, never native acceptance."""
from types import SimpleNamespace
import unittest

import torch

from fault_injection.route8 import RestoreOffsetSession
from fault_injection.index22 import WrongRowSession
from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_modules import (
    TensorModuleProvider, install_tensor_replay)


LIB = torch.library.Library('npu', 'FRAGMENT')
LIB.define('npu_moe_token_unpermute(Tensor permuted_tokens, Tensor sorted_indices, Tensor? probs=None, bool padded_mode=False, int[]? restore_shape=None) -> Tensor')


def cpu_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None):
    tokens, topk = probs.shape
    return (permuted_tokens[sorted_indices.long()].reshape(tokens,topk,-1)
            *probs.unsqueeze(-1)).sum(1)


LIB.impl('npu_moe_token_unpermute',cpu_unpermute,'CPU')


class MoeWorkload(torch.nn.Module):
    def forward(self, hidden, indices, probs):
        return torch.ops.npu.npu_moe_token_unpermute(hidden,indices,probs) + 1


class RopeWorkload(torch.nn.Module):
    def __init__(self):
        super().__init__()
        positions=torch.arange(16.).unsqueeze(1)
        frequency=torch.tensor([.1,.2,.3,.4]).unsqueeze(0)
        self.register_buffer('cos_sin_cache',torch.cat(((positions*frequency).cos(),(positions*frequency).sin()),1))
    def forward(self, x, positions):
        lookup=self.cos_sin_cache.index_select(0,positions)
        cos,sin=lookup.chunk(2,dim=-1)
        rotated=torch.cat((-x[:,2:],x[:,:2]),dim=-1)
        return x*cos+rotated*sin


class RouteAndIndexTests(unittest.TestCase):
    def moe_inputs(self,tokens=3):
        return (torch.arange(tokens*2*4.).reshape(tokens*2,4),
                torch.arange(tokens*2,dtype=torch.int32).flip(0),
                torch.arange(1.,tokens*2+1).reshape(tokens,2)/10)

    def test_moe_moves_groups_and_matching_probabilities(self):
        model=MoeWorkload().eval();inputs=self.moe_inputs()
        original=[x.clone() for x in inputs]
        baseline=model(*inputs)
        with RestoreOffsetSession(model,arm='baseline',simulation=True) as lossless:
            self.assertTrue(torch.equal(model(*inputs),baseline))
        with RestoreOffsetSession(model,simulation=True) as fault:
            result=model(*inputs)
            self.assertTrue(torch.equal(result,baseline[torch.tensor([1,2,2])]))
            event=fault.report()['events'][0]
            self.assertEqual(event['native_calls'],1)
            self.assertEqual(event['source_groups'],[1,2,2])
            self.assertTrue(event['paired_group_readback_exact'])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(original,inputs)))
        self.assertTrue(torch.equal(model(*inputs),baseline))

    def test_t1_passthrough_and_bad_complete_permutation_refused(self):
        model=MoeWorkload().eval();inputs=self.moe_inputs(1);baseline=model(*inputs)
        with RestoreOffsetSession(model,simulation=True) as session:
            self.assertTrue(torch.equal(model(*inputs),baseline))
            self.assertEqual(session.events[0]['status'],'passthrough')
        hidden,indices,probs=self.moe_inputs();indices[0]=indices[1]
        with RestoreOffsetSession(model,simulation=True):
            with self.assertRaisesRegex(RuntimeError,'complete contribution permutation'):
                model(hidden,indices,probs)

    def test_rope_table_wrong_last_row_source_unchanged_and_real_consumers(self):
        model=RopeWorkload().eval();x=torch.arange(16.).reshape(4,4)
        positions=torch.tensor([0,1,2,3]);cache=model.cos_sin_cache.clone()
        baseline=model(x,positions)
        with WrongRowSession(model,simulation=True) as fault:
            actual=model(x,positions)
            event=fault.events[0]
            self.assertTrue(event['replacement_readback_exact'])
            self.assertEqual(event['correct_source_row'],3)
            self.assertEqual(event['replacement_source_row'],0)
            self.assertTrue(event['consumers'])
            self.assertTrue(torch.equal(actual[:-1],baseline[:-1]))
            self.assertFalse(torch.equal(actual[-1],baseline[-1]))
        self.assertTrue(torch.equal(cache,model.cos_sin_cache))
        self.assertTrue(torch.equal(baseline,model(x,positions)))

    def test_unrelated_indexselect_is_not_selected(self):
        class Other(torch.nn.Module):
            def forward(self,x,i):return x.index_select(0,i)
        model=Other().eval();x=torch.arange(12.).reshape(3,4);i=torch.tensor([0,2])
        with WrongRowSession(model,simulation=True) as session:
            self.assertTrue(torch.equal(model(x,i),x[i]))
            self.assertFalse(session.events)

    def test_public_provider_c1_replays_for_both_mechanisms(self):
        for mechanism,model,inputs in (
            ('8.restore_offset',MoeWorkload(),self.moe_inputs()),
            ('22.wrong_row',RopeWorkload(),(torch.arange(16.).reshape(4,4),torch.arange(4)))):
            with self.subTest(mechanism=mechanism):
                model.eval();worker=SimpleNamespace(model_runner=SimpleNamespace(model=model))
                install_tensor_replay(worker,mechanism,replay=lambda m,p:m(*inputs),synchronize=lambda:None,
                                      workload_kind='constructed_module',simulation=True)
                provider=TensorModuleProvider(worker,model,{},mechanism)
                for action in ('prepare','register','check','execute','finish','unregister'):
                    state=provider.dispatch(action,{},owner='cpu-contract',generation=1)
                effect=state['qualification']['effect']
                self.assertTrue(effect['output_changed'] and effect['off_exact'])
                self.assertFalse(effect['native_execution_claimed'])
                self.assertFalse(state['hook_owned'])


if __name__=='__main__': unittest.main()
