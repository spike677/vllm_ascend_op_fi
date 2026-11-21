import json,os,threading,unittest
from pathlib import Path
from unittest.mock import patch
from test_extension import _extended_worker
from demoncat.runtime.vllm_ascend.worker_extension import deployment_acceptance as a
from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import configure_deployment_acceptance
from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import ContinuousBinding,ATTRIBUTE


class ContinuousDeploymentTests(unittest.TestCase):
    def test_reviewed_torch_binding_never_grants_native_or_a3_or_tp1(self):
        record=json.loads(Path(a.__file__).with_name('continuous_blue_records.json').read_text())['records'][0]
        w=_extended_worker();w.load_model();m=w.model_runner.model
        configure_deployment_acceptance(w,source='BLUE_COMBINATION',authorize_operator=lambda *a:True)
        expected=record['expected'];key=record['mechanism']
        with patch.object(a,'observe_deployment',return_value=expected):
            self.assertFalse(a.deployment_acceptance_report(key,w,m)['accepted'])
            b=ContinuousBinding(w,m,key,((1,2),),dict(members=[dict(rank=0)]),dict(rank=0),
                'input',os.getpid(),threading.get_ident())
            setattr(w,ATTRIBUTE,{key:b})
            report=a.deployment_acceptance_report(key,w,m)
            self.assertTrue(report['accepted']);self.assertFalse(report['can_prepare'])
            a.install_blue_acceptance_permit(w,mechanisms=[key],attempt='cpu-scope',source_manifest_sha256='a'*64)
            self.assertTrue(a.deployment_acceptance_report(key,w,m)['can_prepare'])
            a.remove_blue_acceptance_permit(w)
        for changes in ({'hardware_family':'A3'},{'tp':1},{'vllm':'0.27.0'}):
            with patch.object(a,'observe_deployment',return_value=dict(expected,**changes)):
                self.assertFalse(a.deployment_acceptance_report(key,w,m)['accepted'])
        b.arithmetic_device='cpu'
        with patch.object(a,'observe_deployment',return_value=expected):
            self.assertFalse(a.deployment_acceptance_report(key,w,m)['accepted'])
        b.arithmetic_device='input';b.reference_mode='forward'
        with patch.object(a,'observe_deployment',return_value=expected):
            self.assertFalse(a.deployment_acceptance_report(key,w,m)['accepted'])
