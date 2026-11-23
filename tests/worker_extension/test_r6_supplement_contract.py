import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension import product_binding as pb
from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import deployment_acceptance_report

class SupplementContractTests(unittest.TestCase):
    def test_string_fault_and_unknown_mechanism_are_not_profile_lookup(self):
        with self.assertRaisesRegex(ValueError,'positive integer'):FaultConfig('99','unknown_profile')
        with self.assertRaisesRegex(ValueError,'explicit mechanism variant'):FaultConfig(99,'unknown_profile')
        self.assertEqual(FaultConfig(18,'api_half').mechanism,'18.api_half')

    def test_missing_profile_at_trusted_rpc_is_specific_error(self):
        worker=NS(_demoncat_product_profiles={})
        authority=NS(authorize_operator=lambda *args:True)
        with patch.object(pb,'authority_for',return_value=authority),patch.object(pb,'install_continuous_binding') as install:
            with self.assertRaisesRegex(RuntimeError,'PRODUCT_PROFILE_UNKNOWN'):
                pb.product_binding_rpc(worker,'capabilities',{'operator':'trusted','profile':'not_configured_negative_probe'})
            install.assert_not_called()

    def test_continuous_product_still_requires_scoped_permit(self):
        from demoncat.runtime.vllm_ascend.worker_extension import deployment_acceptance as da
        worker=NS();model=object();worker.model_runner=NS(model=model)
        b=NS(product={'mode':'PRODUCT_ACCEPTANCE_ONLY'},native_library=None,arithmetic_device='npu',prompts=(),reference_mode='immediate')
        authority=NS(source='YELLOW_DEDICATED_INSTANCE')
        with patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization.authority_for',return_value=authority),patch.object(da,'observe_deployment',return_value={}),patch.object(da,'_cpu_fixture',return_value=False),patch('demoncat.runtime.vllm_ascend.worker_extension.continuous_binding.current_binding',return_value=b):
            report=deployment_acceptance_report('18.api_half',worker,model)
            self.assertFalse(report['can_prepare']);self.assertIsNone(report['acceptance_permit'])

if __name__=='__main__':unittest.main()
