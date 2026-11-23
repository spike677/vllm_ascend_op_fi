import hashlib,types,sys,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_alias_output_census import GatedModel
from fault_injection.engines.region_lease import RegionLeases
from fault_injection.product_contract import summarize_request

class Alias15Tests(unittest.TestCase):
    def test_actual_writer_detached_base_consumed_and_release_two_scopes(self):
        from fault_injection.sites.alias15 import Alias15Site
        import test_alias_output_census as fixture
        model=torch.nn.Sequential(GatedModel());x=torch.arange(8,dtype=torch.bfloat16).reshape(2,4)
        expected=model(x);runner=types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['live-request']))
        original_empty=torch.empty
        def initialized_fixture(*args,**kw):
            # Deterministic CPU allocator substitute, never part of NPU code.
            return original_empty(*args,**kw).fill_(7)
        for generation in (1,2):
            leases=RegionLeases('owner',generation,stream=lambda:0,complete=lambda stream:None)
            with patch.dict(sys.modules,torch_npu=types.SimpleNamespace(get_npu_format=lambda t:2)):
                site=Alias15Site(model,runner,fault=True,region_leases=leases)
            site.source_sha=hashlib.sha256(Path(fixture.__file__).read_bytes()).hexdigest()
            with patch.object(torch,'empty',initialized_fixture):
                site.install()
                try:
                    for step in range(4):
                        out=model(x)
                        self.assertTrue(torch.equal(out,torch.full_like(x,7)*torch.sigmoid(x)))
                    report=summarize_request('live-request',0,site.events,completed=True)
                    self.assertEqual(report['coverage'],dict(observed=4,eligible=4,attempted=4,applied=4,changed=4,consumed=4,restored=4,failed=0))
                    self.assertEqual(report['changed_elements'],28)
                    self.assertEqual(site.pending,0)
                    for e in site.events:
                        self.assertEqual(e['before_sha256'],e['restored_sha256'])
                        self.assertTrue(e['initial_view_exact']);self.assertTrue(e['detached_storage'])
                        self.assertNotEqual(e['after_sha256'],e['intended_sha256'])
                finally:site.close()
            self.assertTrue(torch.equal(model(x),expected))

    def test_lease_refuses_later_legal_write_without_restoring_old_bytes(self):
        l=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None);x=torch.ones(4)
        token=l.acquire(x);x.add_(3);l.consumed(token)
        with self.assertRaisesRegex(RuntimeError,'BASE_CHANGED_NO_WRITEBACK'):l.release_unmodified(token)
        self.assertTrue(torch.equal(x,torch.full_like(x,4)));self.assertEqual(l.count(),1)

if __name__=='__main__':unittest.main()


class Alias15ReviewTests(unittest.TestCase):
    def fixture(self):
        from test_product_review import receipt
        d=receipt();d['mechanism']='15.alias_detach'
        for scope in d['scopes']:
            for request in scope['requests'][:3]:
                for r in request['result']['ranks']:
                    e=dict(site='natural',request_ids=[r['request_id']],dtype='torch.bfloat16',format=2,initial_view_exact=True,detached_storage=True,before_sha256='old',after_sha256='old',restored_sha256='old',gate_input_sha256='old',intended_sha256='new',gate_result_sha256='gate',consumer_sha256='gate',gate_device_read_completed=True,projection_reads_exact_produced_value=True,device_consumer_completed=True,restore_kind='UNCHANGED_BASE_BYTES_TEMPORARY_RELEASED',changed=1,changed_elements=1,initial_all_finite=True)
                    r['alias_detach_effect']=dict(semantics='ND_VIEW_ALIAS_DETACH_NO_PRIVATE_FORMAT_TRIGGER',pending=0,calls=[e])
        return d
    def test_independent_review_rejects_missing_consumer_or_non_detached_base(self):
        from tools.review_alias15_product import review
        self.assertEqual(review(self.fixture())['status'],'ALIAS15_PRODUCT_MECHANISM_REVIEW_PASSED')
        for k,v in [('detached_storage',False),('after_sha256','new'),('changed',2),('consumer_sha256','different'),('gate_device_read_completed',False)]:
            d=self.fixture();d['scopes'][0]['requests'][0]['result']['ranks'][0]['alias_detach_effect']['calls'][0][k]=v
            with self.assertRaisesRegex(ValueError,'INDEPENDENT_MECHANISM_PROOF'):review(d)

class Alias15DeploymentTests(unittest.TestCase):
    def test_attach_tp2_observed_identity_only_and_no_native_library(self):
        from demoncat.runtime.vllm_ascend.worker_extension import product_binding as pb
        NS=types.SimpleNamespace
        profile=dict(mechanism='15.alias_detach',canary_prompts=[[1,2]],mode='PRODUCT_ACCEPTANCE_ONLY')
        w=NS(model_runner=NS(model=object(),vllm_config=NS(parallel_config=NS(tensor_parallel_size=2))),_demoncat_product_profiles={'alias':profile})
        payload=dict(operator='trusted',profile='alias',scope={'mode':'all'},group_contract={})
        for source in ('BLUE_COMBINATION','YELLOW_DEDICATED_INSTANCE'):
            authority=NS(source=source,authorize_operator=lambda *a:True);b=NS(product=None)
            with patch.object(pb,'authority_for',return_value=authority),patch.object(pb,'current_binding',side_effect=[None,b]),patch.object(pb,'validate_blue_native_worker',return_value={'source':source,'qualification_granted':False}) as check,patch.object(pb,'install_continuous_binding',return_value={'qualification_granted':False}) as install:
                pb.product_binding_rpc(w,'attach',payload)
                check.assert_called_once_with(w,source)
                self.assertIsNone(install.call_args.kwargs['native_library'])
                self.assertFalse(b.alias_runtime_identity['qualification_granted'])
            with patch.object(pb,'authority_for',return_value=authority),patch.object(pb,'current_binding',return_value=None),patch.object(pb,'validate_blue_native_worker',side_effect=RuntimeError('UNPROVEN_CANN')),patch.object(pb,'install_continuous_binding') as install:
                with self.assertRaisesRegex(RuntimeError,'UNPROVEN_CANN'):pb.product_binding_rpc(w,'attach',payload)
                install.assert_not_called()
        w.model_runner.vllm_config.parallel_config.tensor_parallel_size=1
        with patch.object(pb,'authority_for',return_value=authority),patch.object(pb,'current_binding',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'ALIAS15_TP2_ACCEPTANCE_ONLY'):pb.product_binding_rpc(w,'attach',payload)

class Alias15A3ReviewTests(Alias15ReviewTests):
    def test_a3_runtime_identity_required_and_blue_cannot_relabel(self):
        from tools.review_alias15_product import review
        d=self.fixture()
        with self.assertRaisesRegex(ValueError,'INDEPENDENT_MECHANISM_PROOF'):review(d,require_a3=True)
        identity=dict(source='YELLOW_DEDICATED_INSTANCE',tp=2,soc={'family':'A3','observed_name':'Ascend910_9382'},toolkit_version='9.1.0',cann_observation_source='VERIFIED_MAPPED_ELF_LINKED_TO_INSTALLED_VERSION_METADATA',loaded_runtime=[dict(name=n,version='9.1.0',sha256='a'*64,metadata_sha256='b'*64) for n in ('libascendcl.so','libruntime.so')])
        for scope in d['scopes']:
            for request in scope['requests'][:3]:
                for rank in request['result']['ranks']:rank['alias_detach_effect']['runtime_identity']=identity
        review(d,require_a3=True)
        identity['source']='BLUE_COMBINATION'
        with self.assertRaisesRegex(ValueError,'INDEPENDENT_MECHANISM_PROOF'):review(d,require_a3=True)
