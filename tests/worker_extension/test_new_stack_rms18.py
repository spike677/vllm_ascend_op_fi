import json,sys,unittest
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import patch
import torch
from demoncat.runtime.vllm_ascend.worker_extension import new_stack_rms18 as n
from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import remove_tensor_serving_binding
from demoncat.runtime.vllm_ascend.worker_extension.registry import create_worker_provider

class BindingTests(unittest.TestCase):
    def fixture(self,tp=2):
        model=torch.nn.Linear(3,3).eval();runner=NS(model=model,supports_mm_inputs=False)
        identity=dict(rank=0,pid=123);worker=NS(model_runner=runner,demoncat_worker_identity=lambda:dict(status='WORKER_IDENTIFIED',worker=identity))
        members=[identity,dict(rank=1,pid=124)]
        contract=dict(mechanism='18.api_half',tp=2,injection_policy='ONE_AUTOMATIC_RANK',group_generation=1,owner='worker-group-test',members=members,members_sha256=n.digest(members))
        module=ModuleType('vllm.distributed.parallel_state');module.get_tp_group=lambda:NS(world_size=tp)
        self.enterContext(patch.dict(sys.modules,{'vllm.distributed.parallel_state':module}))
        validator=self.enterContext(patch.object(n,'SourceBoundTextObserver'))
        validator.return_value.validate.return_value.snapshot.return_value=dict(owned_hooks=0,owned_wrappers=0)
        options=dict(prompt_token_ids=[1,2],weight_manifest={'cpu_fixture':True},workload_sha256='a'*64,environment_sha256='b'*64,group_contract=contract if tp==2 else None)
        return worker,model,options,validator
    def test_inert_typed_binding_preserves_no_qualification_and_exact_group(self):
        w,m,options,validator=self.fixture();r=n.install_rms18_newstack_binding(w,**options)
        self.assertFalse(r['qualification_granted']);self.assertEqual(r['hooks_installed'],0)
        validator.return_value.install.assert_not_called()
        binding=w._demoncat_natural_tensor_serving_bindings['18.api_half']
        self.assertEqual(binding.execution_profile,n.PROFILE);self.assertEqual(binding.group_context.contract,options['group_contract'])
        with patch('demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission.require_ordinary_route',return_value={'deployment_acceptance':{'cpu_fixture':True}}),patch.object(n,'NewStackTP2Rms18Provider',return_value=NS()) as factory:
            result=create_worker_provider('18.api_half','1',w,m,{})
            factory.assert_called_once_with(w,m,{})
        remove_tensor_serving_binding(w,'18.api_half')
    def test_wrong_group_member_refused_without_binding(self):
        w,m,options,validator=self.fixture();options['group_contract']['members_sha256']='f'*64
        with self.assertRaisesRegex(RuntimeError,'CURRENT_MEMBERS'):n.install_rms18_newstack_binding(w,**options)
        self.assertFalse(hasattr(w,'_demoncat_natural_tensor_serving_bindings'))
    def test_tp1_requires_no_tp2_contract(self):
        w,m,options,validator=self.fixture(1);r=n.install_rms18_newstack_binding(w,**options)
        self.assertEqual(r['tp'],1);self.assertIsNone(w._demoncat_natural_tensor_serving_bindings['18.api_half'].group_context)
        remove_tensor_serving_binding(w,'18.api_half')
    def test_missing_typed_permit_is_not_bypassed_by_binding(self):
        w,m,options,validator=self.fixture();n.install_rms18_newstack_binding(w,**options)
        with patch('demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission.require_ordinary_route',side_effect=RuntimeError('NO_DEPLOYMENT_ACCEPTANCE')):
            with self.assertRaisesRegex(RuntimeError,'NO_DEPLOYMENT_ACCEPTANCE'):create_worker_provider('18.api_half','1',w,m,{})
        remove_tensor_serving_binding(w,'18.api_half')

if __name__=='__main__':unittest.main()
