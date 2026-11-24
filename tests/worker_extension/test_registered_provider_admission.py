import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from demoncat.runtime.vllm_ascend.worker_extension.providers import product
from demoncat.runtime.vllm_ascend.worker_extension import deployment_authorization as da


class RegisteredProviderAdmissionTests(unittest.TestCase):
    def test_real_provider_keeps_mechanism_for_each_live_admission_check(self):
        model=object();profile={'mode':'PRODUCT_REGISTERED'}
        worker=NS(model_runner=NS(model=model),_demoncat_product_profiles={'norm':profile})
        b=NS(product={'scope':{'mode':'all'},'profile':'norm'},mechanism='18.api_half',check=Mock())
        with patch.object(product,'current_binding',return_value=b):
            provider=product.ProductProvider(worker,model,{'rank':0},'18.api_half')
        provider._deployment_admission=(worker,model,dict(mode='PRODUCT_REGISTERED',product_registration={'record_sha256':'a'*64},registration_source='BLUE_COMBINATION',accepted=True,actual={},reviewed_combination='a'*64))
        with patch('demoncat.runtime.vllm_ascend.worker_extension.continuous_binding.current_binding',return_value=b) as binding, patch('demoncat.runtime.vllm_ascend.worker_extension.product_registration.registered_for') as registered:
            self.assertTrue(da.acceptance_use(provider,'prepare')['scope_valid'])
            binding.assert_called_once_with(worker,model,'18.api_half')
            registered.assert_called_once_with(worker,'18.api_half',profile,expected_sha='a'*64)
            registered.side_effect=ValueError('PRODUCT_REGISTRATION_IDENTITY_MISMATCH')
            self.assertFalse(da.acceptance_use(provider,'enable')['scope_valid'])
            with self.assertRaisesRegex(RuntimeError,'SCOPE_CHANGED'):
                da.acceptance_use(provider,'enable',enforce=True)
