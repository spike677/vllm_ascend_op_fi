import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from fault_injection.engines.region_lease import RegionLeases
from demoncat.runtime.vllm_ascend.worker_extension.external_sampling import _NoiseConsumer, ExponentialNoiseSite
import test_external_provider as fixture
from demoncat.runtime.vllm_ascend.worker_extension.external_provider import ExternalSite


class SamplingConsumerTests(unittest.TestCase):
    def test_real_division_reads_modified_noise_then_bytes_restore(self):
        for fault in (False,True):
            leases=RegionLeases('owner',1,stream=lambda:1,complete=lambda s:None)
            site=NS(active=True,runner=NS(input_batch=NS(req_ids=['actual-request'])),events=[],
                    forward_calls=[{}],region_leases=leases,pending={},fault=fault)
            probs=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float32)
            q=torch.tensor([[1.,2.,3.,4.]]);before=q.clone();expected=probs/(q*torch.tensor([[1,2,1,2]]) if fault else q)
            with _NoiseConsumer(site,probs):value=probs.div_(q)
            self.assertTrue(torch.equal(value,expected));self.assertTrue(torch.equal(q,before))
            self.assertEqual(leases.count(),0);self.assertFalse(site.pending)
            event=site.events[0];self.assertTrue(event['consumed'] and event['restored'])
            self.assertEqual(event['changed'],2 if fault else 0)
            self.assertEqual(event['request_ids'],['actual-request'])

    def test_unrelated_division_is_not_intercepted(self):
        x=torch.ones(1,4);site=NS(events=[])
        with _NoiseConsumer(site,x):torch.ones(1,4).div_(torch.ones(1,4))
        self.assertEqual(site.events,[])

    def test_multi_request_noise_refused(self):
        x=torch.ones(2,4);site=NS(active=True)
        with self.assertRaisesRegex(RuntimeError,'MAPPING'):
            with _NoiseConsumer(site,x):x.div_(torch.ones(2,4))

    def test_close_refuses_pending_region(self):
        site=object.__new__(ExponentialNoiseSite);site.pending={1:object()}
        with self.assertRaisesRegex(RuntimeError,'RESTORE_PENDING'):site.close()


class SamplingRegistrationTests(unittest.TestCase):
    setUp=fixture.ExternalProviderTests.setUp
    install=fixture.ExternalProviderTests.install
    def test_persistent_region_classes_explicitly_refused(self):
        for region in ('persistent_weight','kv_cache','raw_address'):
            spec=ExternalSite(self.spec.mechanism,self.spec.factory,self.spec.source_sha256,self.actual,region_kind=region)
            with self.assertRaisesRegex(RuntimeError,'RESTORE_UNPROVEN'):self.install(spec)

    def test_stage_is_explicit_and_unknown_stage_refused(self):
        spec=ExternalSite(self.spec.mechanism,self.spec.factory,self.spec.source_sha256,self.actual,stage='sampler')
        self.install(spec)
        from demoncat.runtime.vllm_ascend.worker_extension.external_provider import external_site
        self.assertEqual(external_site(self.worker,self.spec.mechanism).stage,'sampler')


if __name__=='__main__':unittest.main()
