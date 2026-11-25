"""Exercise the real v2 factory admission, not a replacement provider double."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from reference_adapters.lm_head_provider import LmHeadCollapseProvider,LmHeadSwapProvider
from demoncat.runtime.vllm_ascend.worker_extension import external_persistent_runtime as v2

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
