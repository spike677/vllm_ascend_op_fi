"""Native trusted product deployment gates; no device qualification."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension import product_native as pn
from demoncat.runtime.vllm_ascend.worker_extension import product_binding as pb
from demoncat.runtime.vllm_ascend.worker_extension.product_config import load_product_deployment

class NativeProductDeploymentTests(unittest.TestCase):
    def test_library_bytes_source_tp_and_non_native_misuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lib=root/'fixture.so';lib.write_bytes(b'explicit CPU fixture')
            sha=hashlib.sha256(lib.read_bytes()).hexdigest();manifest=root/'source.json';manifest.write_bytes(b'{}')
            cfg=dict(schema='demoncat.product.deployment.v1',profile='native',fault={'fault_id':41,'variant':'tail_column','implementation_version':'1'},model='/model',tp=2,mode='PRODUCT_ACCEPTANCE_ONLY',source='BLUE_COMBINATION',source_manifest=str(manifest),source_manifest_sha256=hashlib.sha256(b'{}').hexdigest(),native_library=str(lib),native_sha256=sha,canary=dict(prompt='a canary',image=None,max_tokens=4),evidence_directory='/receipts')
            file=root/'deployment.json'
            with patch.object(pn,'BLUE_NATIVE_LIBRARY_SHA256',sha):
                file.write_text(json.dumps(cfg));self.assertEqual(load_product_deployment(file)['native_sha256'],sha)
                for edit in ({'source':'YELLOW_DEDICATED_INSTANCE'},{'tp':1},{'native_sha256':'0'*64},{'native_library':None},{'fault':{'fault_id':18,'variant':'api_half','implementation_version':'1'}}):
                    file.write_text(json.dumps(dict(cfg,**edit)))
                    with self.assertRaises((ValueError,RuntimeError)):load_product_deployment(file)
                lib.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'HASH_MISMATCH'):pn.validate_blue_native_library(str(lib),sha,'BLUE_COMBINATION',2)
    def test_observed_worker_family_and_toolkit_are_required(self):
        worker=NS(model_runner=NS(vllm_config=NS(parallel_config=NS(tensor_parallel_size=2))))
        tk=dict(status='FOUND',installations=[dict(fields={'version':'9.1.0'})])
        with patch('fault_injection.environment._live_soc',return_value=({'family':'A2','observed_name':'Ascend910B4'},'live')),patch('fault_injection.environment._toolkit',return_value=tk):
            pn.validate_blue_native_worker(worker,'BLUE_COMBINATION')
            for source in ['YELLOW_DEDICATED_INSTANCE',None]:
                with self.assertRaises(RuntimeError):pn.validate_blue_native_worker(worker,source)
            tk['installations'][0]['fields']['version']='9.2.1'
            with self.assertRaisesRegex(RuntimeError,'CANN91'):pn.validate_blue_native_worker(worker,'BLUE_COMBINATION')
        with patch('fault_injection.environment._live_soc',return_value=({'family':'A3','observed_name':'Ascend910_9382'},'live')):
            with self.assertRaisesRegex(RuntimeError,'A2_TP2'):pn.validate_blue_native_worker(worker,'BLUE_COMBINATION')
    def test_only_trusted_profile_transports_native_library(self):
        profile=dict(mechanism='41.tail_column',canary_prompts=[[1,2]],mode='PRODUCT_ACCEPTANCE_ONLY',native_library=('/native/reviewed.so','a'*64),native_source='BLUE_COMBINATION')
        worker=NS(model_runner=NS(model=object()),_demoncat_product_profiles={'native':profile})
        authority=NS(source='BLUE_COMBINATION',authorize_operator=lambda *a:True)
        b=NS(product=None)
        with patch.object(pb,'authority_for',return_value=authority),patch.object(pb,'validate_blue_native_worker') as validate,patch.object(pb,'current_binding',side_effect=[None,b]),patch.object(pb,'install_continuous_binding',return_value={'qualification_granted':False}) as install:
            pb.product_binding_rpc(worker,'attach',dict(operator='trusted',profile='native',scope={'mode':'all'},group_contract={}))
            self.assertEqual(install.call_args.kwargs['native_library'],profile['native_library']);validate.assert_called_once_with(worker,'BLUE_COMBINATION')
        with patch.object(pb,'authority_for',return_value=authority):
            with self.assertRaisesRegex(RuntimeError,'UNKNOWN_FIELD'):
                pb.product_binding_rpc(worker,'attach',dict(operator='trusted',profile='native',scope={'mode':'all'},group_contract={},native_library='/attacker.so'))

class NativeProductReviewTests(unittest.TestCase):
    def test_native_proof_and_negative_controls_cannot_be_omitted(self):
        from test_product_review import receipt
        from tools.review_product_mode import review
        from copy import deepcopy
        data=receipt();data['mechanism']='26.fia_destination'
        for scope in data['scopes']:
            scope['negative_controls']=[dict(rank=r,permit_remaining=False,missing_permit={'can_prepare':False},unknown_profile='PRODUCT_PROFILE_UNKNOWN',user_native_path='PRODUCT_RPC_UNKNOWN_FIELD') for r in range(2)]
            for request in scope['requests'][:3]:
                for rank in request['result']['ranks']:
                    rank['native_memory_effect']=dict(leases=0,request_ids=[rank['request_id']],library_sha256='a'*64,
                        regions=[dict(region_elements=4,consumer_operator='aten::add',before_sha256='b'*64,restored_sha256='b'*64,source_sha256='c'*64,after_sha256='c'*64)])
        self.assertFalse(review(data)['qualification_granted'])
        for field,value in [('negative_controls',[]),('negative_controls',[dict(rank=r,permit_remaining=True,missing_permit={'can_prepare':False}) for r in range(2)])]:
            bad=deepcopy(data);bad['scopes'][0][field]=value
            with self.assertRaises(ValueError):review(bad)
        bad=deepcopy(data);bad['scopes'][0]['requests'][0]['result']['ranks'][0]['native_memory_effect']['regions'][0]['after_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'SOURCE_PROOF'):review(bad)

if __name__=='__main__':unittest.main()
