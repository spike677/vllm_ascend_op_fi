"""Actual forward/controller protocol on CPU fixtures, never Worker evidence."""
import copy
import unittest
import torch

import test_serving_tensor_group as host_fixture
from test_rope_shared_triton import Model as Body,JIT,triton_reader
from fault_injection.rope_census import NaturalRoPETrace
from fault_injection.rope_serving import ServingRoPEGroup


class DirectModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Embedding(8,4);self.body=Body()
        with torch.no_grad():self.embed.weight.copy_(torch.arange(32.).reshape(8,4)/8)
    def __call__(self,*args,**kwargs):return self.forward(*args,**kwargs)
    def forward(self,input_ids,positions):return self.body(self.embed(input_ids),positions)


class WorkerModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=DirectModel();self.lm_head_alias=self.model.embed
        self.processor=host_fixture.Processor();self.processor.weight=torch.arange(16.).reshape(4,4)/16
    def forward(self,input_ids,positions):return self.model(input_ids,positions)
    def logits(self,hidden):return self.processor(hidden[-1:])


def template(model):
    return NaturalRoPETrace(model,triton_reader,{'model.body.layers.0.rope':model.model.body.layers[0].rope},{},
        simulation=True,outer_operator='rope_shared_triton_fixture::opaque',triton_kernel=JIT)


class RoPEWorkerGroupTests(unittest.TestCase):
    def setUp(self):
        JIT.skip_launch=False;JIT.replace_position=False
        self.host=host_fixture.ServingTensorGroupTests();self.host.setUp();self.host.session.close()
        self.host.model=WorkerModel().eval();self.host.worker.model_runner.model=self.host.model
        self.host.session=ServingRoPEGroup(self.host.worker,prompt=[1,2,3],weight_manifest={'cpu_fixture':True},
            workload_sha256='a'*64,environment_sha256='b'*64,_simulation=True,
            _synchronize=lambda:None,_trace_factory=template)
        self.session=self.host.session
    def tearDown(self):self.host.tearDown()
    def healthy(self):
        for phase,arm in (('native_baseline',None),('baseline','baseline'),
                          ('baseline_repeat','baseline'),('lossless','baseline')):
            self.host.request(phase,arm)
        return self.session.qualify()
    def test_real_scheduled_denominator_shared_sites_c2_live_c3_and_off(self):
        found=self.healthy();self.assertEqual(found['candidate_set']['eligible'],3)
        self.assertEqual(len({t.module_object_identity for t in self.session.provider.selected.values()}),1)
        self.session.provider.open_group('once',forwards=1)
        fault,_=self.host.request('c2_fault','fault');self.host.request('c2_off')
        c2=self.session.finish_group();self.assertTrue(c2['contract_satisfied'],c2['rejection_reasons'])
        self.assertEqual(c2['hits'],3);self.assertFalse(c2['real_worker_evidence'])
        self.assertTrue(fault['independent']['rope_trace']['complete'])
        proof=self.session.provider.group.c2_proof()
        self.session.provider.open_group('every_match',forwards=2,c2_proof=proof)
        self.host.request('c3_fault_0','fault');self.host.request('c3_fault_1','fault');self.host.request('c3_off')
        c3=self.session.finish_group();self.assertTrue(c3['contract_satisfied'],c3['rejection_reasons'])
        self.assertEqual(c3['hits'],6);self.assertFalse(c3['public_qualification_granted'])
        self.assertTrue(c3['recovery']['complete']);self.assertFalse(c3['raw_device_launch_payload_observed'])
        self.session.close()
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in self.host.model.modules()))
    def test_missing_shared_site_is_not_qualified(self):
        self.host.model.model.body.skip_last=True
        with self.assertRaisesRegex(RuntimeError,'INDEPENDENT_ROPE_CENSUS_INCOMPLETE'):self.healthy()
        self.assertIsNone(self.session.provider.candidates)
    def test_missing_actual_launcher_prevents_group_qualification(self):
        JIT.skip_launch=True
        with self.assertRaisesRegex(RuntimeError,'INDEPENDENT_ROPE_CENSUS_INCOMPLETE'):self.healthy()
        self.assertIsNone(self.session.provider.candidates)
    def test_equal_value_replacement_cache_invalidates_qualified_lifetime(self):
        self.healthy();owner=self.host.model.model.body.layers[0].rope
        owner.cos_sin_cache=owner.cos_sin_cache.clone()
        with self.assertRaisesRegex(RuntimeError,'CACHE_LIFETIME_CHANGED'):
            self.session.expect('changed-cache','c2_fault','fault')
        self.assertIsNone(self.session.adapter)
    def test_every_match_without_live_c2_is_refused(self):
        self.healthy()
        with self.assertRaisesRegex(RuntimeError,'C2'):
            self.session.provider.open_group('every_match',forwards=2)
    def test_copied_c2_capability_is_not_reconstructed_as_authority(self):
        self.healthy();self.session.provider.open_group('once',forwards=1)
        self.host.request('c2_fault','fault');self.host.request('c2_off')
        self.assertTrue(self.session.finish_group()['contract_satisfied'])
        proof=self.session.provider.group.c2_proof()
        with self.assertRaisesRegex(RuntimeError,'C2'):
            self.session.provider.open_group('every_match',forwards=2,c2_proof=copy.copy(proof))
        self.session.close()
    def test_root_manifest_is_one_bounded_group_without_fixed_layer_count(self):
        from experiments.c2c3_closure_20261117.prepare_rope22_group import group_cases
        case,=group_cases('/home/frozen-source',2)
        self.assertIn('natural_rope_worker_probe.py',case['command'])
        self.assertEqual(case['review']['physical_cards'],[2])
        self.assertIn('original live C2 proof',case['review']['expected_hits'])
        self.assertIn('No raw statistics',case['review']['recovery_plan'])


if __name__=='__main__':unittest.main()
