import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from demoncat.runtime.vllm_ascend.worker_extension.product_errors import ProductContractRefused,product_control_reply
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension import product_binding as pb
from demoncat.runtime.vllm_ascend.worker_extension.extension import DemoncatWorkerExtension


class ControlRefusalTests(unittest.TestCase):
    def client(self):
        c=ProductClient(Mock(),FaultConfig(18,'api_half'),profile='norm',operator='host',run_canary=Mock(),drain=Mock())
        c.members=[{'rank':0},{'rank':1}];c.clients=[Mock(),Mock()]
        return c

    def test_all_rank_refusal_is_typed_and_cli_preserves_codes(self):
        c=self.client()
        for i,child in enumerate(c.clients):
            child.prepare.return_value=dict(status='PROVIDER_UNAVAILABLE',worker={'rank':i},response={'provider_code':'ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT','error_type':'RuntimeError'})
        with self.assertRaises(ProductContractRefused):c._all('prepare')
        reply=product_control_reply(lambda:c._all('prepare'))
        self.assertEqual(reply['exit_code'],4)
        self.assertEqual(reply['refusal_codes'],['ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT'])
        self.assertEqual([r['rank'] for r in reply['refusals']],[0,1])

    def test_mixed_partial_and_internal_failure_are_not_all_rank_refusal(self):
        c=self.client()
        c.clients[0].prepare.return_value={'status':'ACTION_COMPLETED'}
        c.clients[1].prepare.return_value={'status':'REJECTED','reason':'contract'}
        self.assertEqual(product_control_reply(lambda:c._all('prepare'))['exit_code'],5)
        for child in c.clients:child.prepare.return_value={'status':'PROVIDER_UNAVAILABLE','response':{'error_type':'KeyError'}}
        self.assertEqual(product_control_reply(lambda:c._all('prepare'))['exit_code'],5)

    def test_binding_all_refused_preserves_reason_partial_is_error(self):
        c=self.client();rows=[dict(rank=i,status='PRODUCT_BINDING_REFUSED',reason='PRODUCT_PROFILE_UNKNOWN') for i in range(2)]
        c.executor.collective_rpc.return_value=rows
        self.assertEqual(product_control_reply(lambda:c._binding('capabilities'))['refusal_codes'],['PRODUCT_PROFILE_UNKNOWN'])
        rows[0]={'rank':0,'status':'PRODUCT_CANDIDATE'}
        self.assertEqual(product_control_reply(lambda:c._binding('capabilities'))['exit_code'],5)

    def test_canary_control_rpc_returns_latched_refusal(self):
        w=NS(rank=0,model_runner=NS(model=object()),_demoncat_product_profiles={'batch':{'mechanism':'quant.batch_shared_scale'}},demoncat_worker_identity=lambda:{'worker':{'rank':0}})
        b=NS(product={},warmup_request=NS(control_refusal='PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1'))
        with patch.object(pb,'authority_for',return_value=NS(authorize_operator=lambda *a:True)),patch.object(pb,'current_binding',return_value=b):
            reply=DemoncatWorkerExtension.demoncat_product_binding(w,'canary',{'operator':'host','profile':'batch'})
        self.assertEqual(reply['exit_code'],4)
        self.assertEqual(reply['reason'],'PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1')

    def test_identity_control_failure_is_structured(self):
        w=NS(rank=1,demoncat_worker_identity=Mock(side_effect=ValueError('IDENTITY_UNAVAILABLE')))
        reply=DemoncatWorkerExtension.demoncat_product_binding(w,'capabilities',{})
        self.assertEqual((reply['rank'],reply['exit_code']),(1,4))

    def test_runtime_control_exception_does_not_escape(self):
        from demoncat.runtime.vllm_ascend.worker_extension import extension as ext
        for error,code in [(ValueError('bad'),4),(KeyError('bug'),5)]:
            with patch.object(ext.WorkerFaultRuntime,'invoke',side_effect=error):
                row=DemoncatWorkerExtension.demoncat_capabilities(NS())
            self.assertEqual(row['exit_code'],code)

    def test_named_validation_reason_preserved_as_code(self):
        from demoncat.runtime.vllm_ascend.worker_extension.product_errors import product_error_reply
        self.assertEqual(product_error_reply(ValueError('PRODUCT_NEW_ATTACHMENT_REQUIRED'))['code'],'PRODUCT_NEW_ATTACHMENT_REQUIRED')
        self.assertEqual(product_control_reply(lambda:FaultConfig(99,'unknown'))['code'],'EXPLICIT_MECHANISM_VARIANT_REQUIRED')
