import unittest
from copy import deepcopy
from unittest.mock import Mock
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from test_product_review import receipt


class ProductHostSafetyTests(unittest.TestCase):
    def client(self):
        client=ProductClient(Mock(),FaultConfig(18,'api_half'),profile='norm',operator='trusted',
                             run_canary=Mock(),drain=Mock())
        client.members=[dict(rank=0),dict(rank=1)]
        client.clients=[Mock(),Mock()]
        return client

    def test_binding_rank_order_is_not_rpc_response_order(self):
        client=self.client()
        client.executor.collective_rpc.return_value=[dict(rank=1),dict(rank=0)]
        self.assertEqual([x['rank'] for x in client.capabilities()],[0,1])
        for rows in ([dict(rank=0)], [dict(rank=0),dict(rank=0)], [dict(rank=False),dict(rank=1)]):
            client.executor.collective_rpc.return_value=rows
            with self.assertRaisesRegex(RuntimeError,'PRODUCT_BINDING_RANK_SET_CHANGED'):client.capabilities()

    def test_partial_rank_arm_does_not_allow_further_requests_or_claim_recovery(self):
        client=self.client();client.enabled=True
        client.clients[0].advance.return_value={'status':'ACTION_COMPLETED'}
        client.clients[1].advance.return_value={'status':'FAILED'}
        with self.assertRaisesRegex(RuntimeError,'PRODUCT_REQUEST_ARM_FAILED'):client.before_request('actual')
        self.assertFalse(client.enabled)
        self.assertEqual(client.inflight,'actual')
        self.assertEqual(client.before_request('new'),{'status':'FAULT_DISABLED'})
        with self.assertRaisesRegex(RuntimeError,'PRODUCT_DRAIN_DID_NOT_COMPLETE_RESULT_HOOK'):client.disable()
        self.assertIsNone(client.recovery)
        for worker in client.clients:worker.unregister.assert_not_called()

    def test_unstable_canary_retains_cleanup_responsibility_without_enabling(self):
        client=self.client();client._binding=Mock();client._all=Mock()
        client._canary=Mock(return_value=[dict(warmup_valid=False),dict(warmup_valid=True)])
        with self.assertRaisesRegex(RuntimeError,'PRODUCT_CANARY_UNSTABLE_IN_SIX'):client.enable()
        self.assertEqual(client._canary.call_count,6)
        self.assertTrue(client.bound);self.assertFalse(client.enabled)
        client._all.assert_not_called()

    def test_cleanup_or_canary_failure_never_reports_restored(self):
        for corrupt in ('canary','resources'):
            client=self.client();record=receipt()['scopes'][0]['disable']
            client.baseline=[deepcopy(x['observations'][-1]['output']) for x in record['canary']['before']]
            client.canary_evidence=deepcopy(record['canary'])
            if corrupt=='canary':record['canary']['after'][0]['observations'][-1]['output'][0]['hidden']='wrong'
            else:record['cleanup']['unregister'][0]['response']['provider_state']['leases']=1
            client._all=Mock(side_effect=lambda action:record['cleanup'][action])
            client._binding=Mock();client._canary=Mock(return_value=record['canary']['after'])
            with self.assertRaisesRegex(RuntimeError,'PRODUCT_RECOVERY_NOT_PROVEN'):client.disable()
            self.assertFalse(client.recovery['restored'])
            with self.assertRaisesRegex(RuntimeError,'PRODUCT_CONFIRMED_RECOVERY_REQUIRED'):client.detach()
