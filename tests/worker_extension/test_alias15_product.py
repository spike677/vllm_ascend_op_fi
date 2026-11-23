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
