"""Acceptance certificates are distinct from installed adapters/structural fit."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
from demoncat.runtime.vllm_ascend.worker_extension import deployment_acceptance as a
from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report
from demoncat.runtime.vllm_ascend.worker_extension.registry import create_worker_provider
from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import TensorServingBinding

class DeploymentAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.model=object();self.worker=NS(model_runner=NS(model=self.model))
    def actual(self,key):
        record=a._CONFIGS[key]
        return dict(a._COMMON,model_config_sha256=record[0],model_class=record[1],live_config_matches_file=True)
    def test_implemented_unaccepted_unimplemented_are_separate(self):
        with patch.object(a,'observe_deployment',return_value=self.actual('8.restore_offset')):
            self.assertEqual(a.deployment_acceptance_report('8.restore_offset',self.worker,self.model)['status'],'ACCEPTED_DEPLOYMENT')
            self.assertEqual(a.deployment_acceptance_report('22.wrong_row',self.worker,self.model)['status'],'IMPLEMENTED_NOT_ACCEPTED_FOR_DEPLOYMENT')
            self.assertEqual(a.deployment_acceptance_report('20.buffer_substitute',self.worker,self.model,implemented=False)['status'],'NOT_IMPLEMENTED')
    def test_reviewed8_17_22_only_accept_exact_deployment(self):
        for key in ('8.restore_offset','17.spatial_padding','22.wrong_row'):
            actual=self.actual(key)
            with self.subTest(key=key),patch.object(a,'observe_deployment',return_value=actual):
                report=a.deployment_acceptance_report(key,self.worker,self.model)
                self.assertTrue(report['accepted']);self.assertEqual(report['mode'],'DEFAULT')
                self.assertIsNone(report['pending_revalidation'])
                self.assertIn('4c56701f/' if key=='22.wrong_row' else '16f40f1f/',report['reviewed_combination'])
            for field,value in [('model_config_sha256','f'*64),('tp',2),('cann',['9.0.0'])]:
                changed=dict(actual);changed[field]=value
                with self.subTest(key=key,field=field),patch.object(a,'observe_deployment',return_value=changed):
                    self.assertFalse(a.deployment_acceptance_report(key,self.worker,self.model)['can_prepare'])

    def test_new_batch_model_or_runtime_not_accepted(self):
        for field,value in [('model_config_sha256','f'*64),('model_class','models.DeepseekV4'),
                ('tp',4),('cann',['9.0.0']),('torch','9'),('live_config_matches_file',False)]:
            actual=self.actual('quant.batch_shared_scale');actual[field]=value
            with self.subTest(field=field),patch.object(a,'observe_deployment',return_value=actual):
                row=a.deployment_acceptance_report('quant.batch_shared_scale',self.worker,self.model)
                self.assertFalse(row['can_prepare']);self.assertFalse(row['accepted'])
                self.assertTrue(any(c['status']=='mismatch' and c['actual']==value for c in row['checks']))
    def test_binding_alone_does_not_admit22_query_and_create_agree(self):
        b=TensorServingBinding('22.wrong_row',self.worker,self.worker.model_runner,self.model,(1,2),'{}','a'*64,'b'*64,1,1,runtime_only=True)
        self.worker._demoncat_natural_tensor_serving_bindings={'22.wrong_row':b}
        with patch.object(a,'observe_deployment',return_value={}):
            report=ordinary_route_report('22.wrong_row',self.worker,self.model)
            self.assertFalse(report['can_prepare'])
            with self.assertRaisesRegex(RuntimeError,'ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT') as caught:
                create_worker_provider('22.wrong_row','1',self.worker,self.model,{})
            self.assertEqual(report,caught.exception.prerequisites)
        self.assertFalse(b.leases)
    def test_explicit_blue_permit_does_not_create_acceptance(self):
        a.install_blue_acceptance_permit(self.worker,mechanisms=['22.wrong_row'],attempt='cpu-contract',source_manifest_sha256='a'*64)
        with patch.object(a,'observe_deployment',return_value={}):
            row=a.deployment_acceptance_report('22.wrong_row',self.worker,self.model)
            self.assertTrue(row['can_prepare']);self.assertFalse(row['accepted']);self.assertEqual(row['mode'],'BLUE_ACCEPTANCE_ONLY')
            self.assertFalse(a.deployment_acceptance_report('8.restore_offset',self.worker,self.model)['can_prepare'])
            self.worker.model_runner.model=object()
            self.assertFalse(a.deployment_acceptance_report('22.wrong_row',self.worker,self.model)['can_prepare'])
        a.remove_blue_acceptance_permit(self.worker)
    def test_permit_cannot_open_unimplemented_route_or_release_live_owner(self):
        a.install_blue_acceptance_permit(self.worker,mechanisms=['20.buffer_substitute'],attempt='test',source_manifest_sha256='b'*64)
        with patch.object(a,'observe_deployment',return_value={}):
            self.assertFalse(ordinary_route_report('20.buffer_substitute',self.worker,self.model)['can_prepare'])
        self.worker._demoncat_worker_fault_runtime=NS(_leases={'20.buffer_substitute':object()})
        with self.assertRaisesRegex(RuntimeError,'OWNER_CLEAN'):a.remove_blue_acceptance_permit(self.worker)
    def test_public_configuration_has_no_acceptance_bypass(self):
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        with self.assertRaises(TypeError):FaultConfig(22,'wrong_row',blue_acceptance=True)
    def test_loaded_torch_version_subclass_is_canonical_string(self):
        class Version(str):pass
        import sys
        with patch.dict(sys.modules,{'torch':NS(__version__=Version('2.9.0+cpu'))}):
            actual=a.observe_deployment(self.worker,self.model)
        self.assertIs(type(actual['torch']),str)
        self.assertEqual(actual['torch'],'2.9.0+cpu')

    def test_actual_observer_does_not_import_or_execute_models(self):
        with patch('importlib.import_module',side_effect=AssertionError('no dynamic import')):
            result=a.observe_deployment(self.worker,self.model)
        self.assertIsNone(result['model_config_sha256']);self.assertFalse(result['live_config_matches_file'])

if __name__=='__main__':unittest.main()
