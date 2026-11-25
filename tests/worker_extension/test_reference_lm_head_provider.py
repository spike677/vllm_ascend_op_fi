"""Exercise the real v2 factory admission, not a replacement provider double."""
import unittest,torch
from types import SimpleNamespace
from unittest.mock import patch
from reference_adapters.lm_head_provider import LmHeadCollapseProvider,LmHeadSwapProvider
from demoncat.runtime.vllm_ascend.worker_extension import external_persistent_runtime as v2
import test_reference_lm_head_rows as row_fixtures

class ProviderContractTests(unittest.TestCase):
    def make(self,factory,**context):
        model=object();worker=SimpleNamespace(model_runner=SimpleNamespace(model=model),
            _reference_lm_context=dict(reader_calls=4,provider=None,**context))
        spec=v2.PersistentSite('external.test','persistent_weight',factory,
                               v2.factory_sha256(factory),{'tp':2})
        setattr(worker,v2.ATTRIBUTE,{'external.test':(spec,model)})
        return worker,model

    def test_both_named_factories_meet_real_persistent_runtime_protocol(self):
        for factory in (LmHeadCollapseProvider,LmHeadSwapProvider):
            worker,model=self.make(factory)
            with patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance.observe_deployment',return_value={'tp':2}),patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance.deployment_acceptance_report',return_value={'can_prepare':True,'acceptance_permit':{'id':'fixture'}}):
                provider=v2.create_persistent_provider(worker,model,{'rank':0},'external.test')
                self.assertIsInstance(provider,factory)
                self.assertEqual(provider.region_kind,'persistent_weight')
                self.assertEqual(provider.persistent_protocol_version,2)
                self.assertIsNone(provider.site) # constructing must not write

    def test_absent_actual_reader_remains_structured_preparation_refusal(self):
        worker,model=self.make(LmHeadCollapseProvider,reader_failure={'operators':['aten::t']})
        with self.assertRaisesRegex(RuntimeError,'ROW12_ACTUAL_READER_NOT_OBSERVED'):
            LmHeadCollapseProvider(worker,model,{'rank':0},mechanism='external.test',region_kind='persistent_weight')

    def test_actual_provider_lifecycle_two_scopes_both_shard_roles(self):
        stream=SimpleNamespace(synchronize=lambda:None)
        with patch.object(torch,'npu',SimpleNamespace(current_stream=lambda:stream),create=True):
            for factory in (LmHeadCollapseProvider,LmHeadSwapProvider):
                for rank,start in ((0,0),(1,16)):
                    site,weight,holder,boundary=row_fixtures.RowTests().make(factory.variant,start=start,end=start+16)
                    original=weight.clone();module=SimpleNamespace(weight=weight)
                    context=dict(module=module,admission=vars(site.admission),provider=None,
                                 reader_calls=4,boundary=lambda:boundary)
                    worker=SimpleNamespace(_reference_lm_context=context)
                    for i in range(2):
                        p=factory(worker,object(),{'rank':rank},mechanism='external.test',region_kind='persistent_weight')
                        for action in ('prepare','register','check','enable'):
                            p.dispatch(action,{},owner=str(i),generation=i+1)
                        for step in range(4):
                            torch.nn.functional.linear(torch.ones(1,8,dtype=weight.dtype),weight)
                            p.consumed('request-'+str(i))
                        for action in ('disable','unregister'):
                            state=p.dispatch(action,{},owner=str(i),generation=i+1)
                        self.assertTrue(state['recovery']['bytes_restored'])
                        self.assertTrue(state['recovery']['resources_zero'])
                        self.assertEqual(state['reader_observations'],4 if rank==0 else 0)
                        self.assertTrue(torch.equal(weight,original));self.assertIsNone(context['provider'])
