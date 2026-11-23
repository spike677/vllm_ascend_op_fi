import unittest
from copy import deepcopy
from fault_injection.sites.allgather_output8 import validate_product_scope
from fault_injection.product_contract import PRODUCT_CANDIDATES, summarize_request
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from tools.review_product_mode import review
from test_product_review import receipt

class AllGatherProductTests(unittest.TestCase):
    def test_no_legacy_tensor_fallback_or_unbound_public_route(self):
        from demoncat.runtime.vllm_ascend.worker_extension import registry
        from types import SimpleNamespace
        with self.assertRaisesRegex(RuntimeError,'TYPED_BLUE_PRODUCT_BINDING'):
            registry.create_research_provider('8.allgather_output_row','1',None,None,{})
        with self.assertRaisesRegex(RuntimeError,'ORDINARY_TASK_ADAPTER_PENDING'):
            registry.create_worker_provider('8.allgather_output_row','1',SimpleNamespace(),object(),{})
    def test_separate_named_variant_and_no_ep_v4_or_yellow_qualification(self):
        self.assertIn('8.allgather_output_row',PRODUCT_CANDIDATES)
        self.assertNotIn('8.restore_offset',PRODUCT_CANDIDATES)
        self.assertEqual(FaultConfig(8,'allgather_output_row').mechanism,'8.allgather_output_row')
        args=dict(family='deepseek_v2',source='BLUE_COMBINATION',tp=2,ep=False,soc={'family':'A2'})
        validate_product_scope(**args)
        for k,v in [('family','deepseek_v4'),('source','YELLOW_DEDICATED_INSTANCE'),('tp',1),('ep',True),('soc',{'family':'A3'})]:
            with self.assertRaisesRegex(RuntimeError,'ALLGATHER8_BLUE'):validate_product_scope(**dict(args,**{k:v}))
    def test_summary_retains_real_routing_join(self):
        e=dict(site='moe',eligible=False,actual_producer_join=True,row_request_ids=['decode'],native_calls=1,metadata_unchanged=True,reference_variant='allgather_output_row_proxy')
        r=summarize_request('decode',1,[e],completed=True)
        self.assertEqual(r['sites'][0]['row_request_ids'],['decode']);self.assertEqual(r['coverage']['eligible'],0)
    def test_reviewer_rejects_old_index_evidence(self):
        d=receipt();d['mechanism']='8.allgather_output_row'
        with self.assertRaisesRegex(ValueError,'ALLGATHER_PRODUCT_ROUTING'):review(d)
if __name__=='__main__':unittest.main()
