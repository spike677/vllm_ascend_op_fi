import unittest
from copy import deepcopy
from fault_injection.dsv4_product_v030 import prepare_route,request_policy,enable_prepared_route,validate_group
class DRouteTests(unittest.TestCase):
    def groups(self):
        return [dict(global_rank=i,**{n:dict(members=list(range(8)),local_rank=i,source='fixture.'+n) for n in ('tp','ep','mc2')}) for i in range(8)]
    def identity(self):
        return dict(vllm='0.30.0+empty',vllm_ascend='0.30.0',cann='9.2.1',tp=8,ep=True,dp=1,
                    model_config_sha256='a'*64,quant_model_description_sha256='b'*64,runtime_source_sha256='c'*64,safetensors_index_sha256={'model.safetensors.index.json':'d'*64})
    def facts(self,n):
        return dict(prompt_length=n,max_tokens=4,request_count=1,single_chunk=True,fresh_cache=True,speculative=False,lora=False,prompt_embeds=False,image_count=0,video_count=0)
    def test_exact_long_request_boundaries(self):
        d=dict(max_model_len=4096,max_num_batched_tokens=4092)
        for n in (2500,4092):self.assertTrue(request_policy(self.facts(n),d)['eligible'])
        for n in (2499,4093):self.assertFalse(request_policy(self.facts(n),d)['eligible'])
    def test_real_scheduler_budget_and_flags(self):
        f=self.facts(4092)
        self.assertFalse(request_policy(f,dict(max_model_len=4096,max_num_batched_tokens=4091))['eligible'])
        for key in ('fresh_cache','single_chunk'):
            x=deepcopy(f);x[key]=False
            self.assertFalse(request_policy(x,dict(max_model_len=4096,max_num_batched_tokens=4096))['eligible'])
    def test_preparation_does_not_grant_execution(self):
        r=prepare_route(self.identity(),self.groups())
        self.assertFalse(r['executable']);self.assertFalse(r['qualification_granted'])
        r['executable']=True
        with self.assertRaisesRegex(ValueError,'NOT_YET_REVIEWED'):enable_prepared_route(r)
    def test_missing_or_drifting_groups(self):
        for key in ('ep','mc2','tp'):
            rows=self.groups();del rows[0][key]
            with self.assertRaises(ValueError):validate_group(rows)
        rows=self.groups();rows[1]['mc2']['members']=[1,0,2,3,4,5,6,7]
        with self.assertRaises(ValueError):validate_group(rows)
    def test_no_cross_stack_or_missing_identity(self):
        for key,value in [('vllm','0.26.0'),('tp',2),('ep',False),('runtime_source_sha256',None),('safetensors_index_sha256',{})]:
            identity=self.identity();identity[key]=value
            with self.assertRaises(ValueError):prepare_route(identity,self.groups())
    def test_original_qwen_range_not_widened(self):
        from fault_injection.product_contract import request_range
        self.assertFalse(request_range(self.facts(2500))['eligible'])
if __name__=='__main__':unittest.main()

