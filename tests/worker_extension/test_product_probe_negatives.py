"""The acceptance probe must not report negative success with leaked owners."""
import types, unittest
from unittest.mock import patch
from experiments.worker_core_v11_20261118.product_mode_probe import trusted_bootstrap

class ProbeNegativeTests(unittest.TestCase):
    def run_probe(self, owners):
        module=types.ModuleType('vllm.v1.worker.worker_base')
        module.WorkerWrapperBase=type('WorkerWrapperBase',(),{})
        worker=types.SimpleNamespace(rank=1,model_runner=types.SimpleNamespace(model=object()),
            _demoncat_worker_fault_runtime=types.SimpleNamespace(_leases={},_tenants=owners))
        with patch.dict('sys.modules',{'vllm.v1.worker.worker_base':module}), \
             patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization.authority_for',return_value=None), \
             patch('demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission.ordinary_route_report',return_value={'can_prepare':False}), \
             patch('demoncat.runtime.vllm_ascend.worker_extension.product_binding.product_binding_rpc',side_effect=RuntimeError('PROFILE_REJECTED')):
            return trusted_bootstrap(worker,'negative',{'mechanism':'18.api_half'})
    def test_real_catalog_rejects_unknown_mechanism(self):
        row=self.run_probe({})
        self.assertIn('explicit mechanism',row['unknown_mechanism'])
        self.assertEqual(row['resources'],{'leases':0,'owners':0})
    def test_residual_owner_cannot_be_reported_clean(self):
        with self.assertRaises(AssertionError):self.run_probe({'unfinished':object()})
if __name__=='__main__':unittest.main()
