import unittest
from types import SimpleNamespace
import torch
from fault_injection.sites.paged33_splitfuse import SplitFuse33Site
from fault_injection.engines.paged33_native_control import split_fuse_control

class SplitFuseTests(unittest.TestCase):
    def test_geometry_key_set_and_whole_output_replacement(self):
        torch.manual_seed(53);q=torch.randn(12,256).bfloat16();k=torch.randn(2305,2,256).bfloat16();v=torch.randn_like(k)
        backup=split_fuse_control(q[None],k,v,scale=.0625,split_spans=[(i,min(i+512,len(k))) for i in range(0,len(k),512)])[0]
        site=object.__new__(SplitFuse33Site);event={'site':'actual.attn','page_mapping':{'block_size':128,'logical_keys':2305}}
        result=site.calculate(q,k,v,SimpleNamespace(scale=.0625),backup,event)
        p=event['partition_reference'];self.assertEqual(p['control_agreement']['bitwise_ratio'],1)
        self.assertEqual(result.shape,backup.shape);self.assertEqual(p['key_visits'],2305)
        self.assertEqual(p['replacement_policy'],'WHOLE_TARGET_ROW_NO_VALUE_SELECTION')
        with self.assertRaisesRegex(RuntimeError,'CONTROL_BELOW_99'):site.calculate(q,k,v,SimpleNamespace(scale=.0625),backup+10,event)
    def test_failed_control_preserves_native_and_defers_to_common_boundary(self):
        from unittest.mock import patch
        from fault_injection.sites.gqa28 import GQA28Site
        from fault_injection.sites.paged33_splitfuse import ControlMismatch
        site=object.__new__(SplitFuse33Site);site.pending={};site.events=[]
        native=torch.ones(1,3);calls=[]
        def original(*args):calls.append(args);return native
        def base(self,name,impl,original,args):
            original(*args);self.events.append(dict(site=name,eligible=True))
            raise ControlMismatch('CONTROL_BELOW_99')
        with patch.object(GQA28Site,'call_gqa',base):result=site.call_gqa('real',None,original,(1,))
        self.assertIs(result,native);self.assertEqual(calls,[(1,)])
        self.assertTrue(site.events[-1]['failed']);self.assertFalse(site.events[-1]['applied'])

    def test_unsupported_deployment_refused_and_prefill_not_eligible(self):
        site=object.__new__(SplitFuse33Site);self.assertFalse(site.eligible_call(2304,2304)[0])
        with self.assertRaisesRegex(RuntimeError,'UNMEASURED_LENGTH'):site.eligible_call(1,100)

if __name__=='__main__':unittest.main()
