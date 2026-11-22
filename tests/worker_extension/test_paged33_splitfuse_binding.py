import unittest
from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import validate_paged_partition,paged_site_type
from fault_injection.sites.paged33 import Paged33Site
from fault_injection.sites.paged33_splitfuse import NativeBoundarySplitFuse33Site
class BindingTests(unittest.TestCase):
    def test_explicit_reviewed_pair_routes_without_monkeypatch(self):
        validate_paged_partition('33.paged_decode',[512,256],'input',[[1]*2304])
        self.assertIs(paged_site_type((512,256)),NativeBoundarySplitFuse33Site)
        self.assertIs(paged_site_type(None),Paged33Site)
        validate_paged_partition('33.paged_decode',[128,32],'input',[[1]*768])
    def test_control_identity_is_observed_but_not_a_grant(self):
        import os,threading
        from unittest.mock import patch
        from test_extension import _extended_worker
        from demoncat.runtime.vllm_ascend.worker_extension import deployment_acceptance as a
        from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import ContinuousBinding,ATTRIBUTE
        w=_extended_worker();w.load_model();model=w.model_runner.model
        binding=ContinuousBinding(w,model,'33.paged_decode',((1,2),),dict(members=[dict(rank=0)]),dict(rank=0),'input',os.getpid(),threading.get_ident(),paged_partition=(512,256))
        setattr(w,ATTRIBUTE,{'33.paged_decode':binding})
        with patch.object(a,'observe_deployment',return_value={}):r=a.deployment_acceptance_report('33.paged_decode',w,model)
        self.assertEqual(r['actual']['paged_reference']['control_kind'],'BF16_INPUT_FP32_OUTPUT_SPLITFUSE')
        self.assertFalse(r['accepted'])

    def test_unsupported_length_backend_or_pair_refuses(self):
        for pair,device,n in [([512,256],'cpu',2304),([512,256],'input',768),([512,128],'input',2304)]:
            with self.subTest(pair=pair,device=device,n=n),self.assertRaises(RuntimeError):validate_paged_partition('33.paged_decode',pair,device,[[1]*n])
if __name__=='__main__':unittest.main()
