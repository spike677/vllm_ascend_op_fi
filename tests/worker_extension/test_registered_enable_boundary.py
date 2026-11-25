import unittest
from unittest.mock import Mock
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
from demoncat.runtime.vllm_ascend.worker_extension.product_errors import ProductContractRefused, product_control_reply


class RegisteredEnableBoundaryTests(unittest.TestCase):
    def test_census_then_unregistered_enable_refuses_and_cleans_all_ranks(self):
        client=ProductClient(Mock(),FaultConfig(22,'wrong_row'),profile='22.wrong_row',
                             operator='trusted',run_canary=Mock(),drain=Mock())
        client.members=[dict(rank=i) for i in range(2)]
        client.clients=[Mock() for _ in range(2)]
        for row in client.clients:
            row.snapshot.return_value=dict(attachment=None,ownership_retained=False)
        actions=[]
        def binding(action,**kw):
            actions.append(action)
            if action=='capabilities':
                return [dict(rank=i,status='PRODUCT_REGISTRATION_REFUSED') for i in range(2)]
            if action=='attach':
                raise ProductContractRefused('PRODUCT_BINDING_ATTACH_FAILED',[
                    dict(rank=i,status='PRODUCT_BINDING_REFUSED',reason='PRODUCT_COMBINATION_NOT_REGISTERED') for i in range(2)])
            if action=='rollback':
                return [dict(rank=i,status='PRODUCT_BINDING_ROLLED_BACK',resources_zero=True) for i in range(2)]
            self.fail(action)
        client._binding=binding
        client.capabilities()
        result=product_control_reply(client.enable)
        self.assertEqual(result['exit_code'],4)
        self.assertIn('PRODUCT_COMBINATION_NOT_REGISTERED',result['refusal_codes'])
        self.assertEqual(actions,['capabilities','attach','rollback'])
        self.assertFalse(client.bound)
        self.assertTrue(client.recovery['resources_zero'])
        client.run_canary.assert_not_called()
