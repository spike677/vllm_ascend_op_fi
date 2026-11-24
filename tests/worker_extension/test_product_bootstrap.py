import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension import product_bootstrap as b
from demoncat.runtime.vllm_ascend.worker_extension import product_cli_entry as cli

class BootstrapTests(unittest.TestCase):
    def make(self):
        worker=NS(model_runner=NS(model=object()))
        self.allowed=True
        self.auth=lambda principal,operation: principal=='trusted' and self.allowed
        b.configure_product_bootstrap(worker,source='BLUE_COMBINATION',authorize_operator=self.auth,
            paths=['/deployment/p.json'],canary_tokens={'profile':[1,2]})
        return worker
    def test_all_actions_and_callback(self):
        worker=self.make()
        for action,fields,fn in [('configure',{},'configure_product_deployments'),
            ('permit',dict(mechanisms=['18.api_half'],attempt='owner-a',source_manifest_sha256='a'*64),'install_deployment_acceptance_permit'),
            ('remove_permit',{},'remove_deployment_acceptance_permit')]:
            with patch.object(b,fn) as call:
                b.product_bootstrap_rpc(worker,action,dict(operator='trusted',**fields));call.assert_called_once()
        self.allowed=False
        with self.assertRaises(PermissionError):b.product_bootstrap_rpc(worker,'configure',dict(operator='trusted'))
    def test_unconfigured_or_unknown_operator_rejected(self):
        with self.assertRaises(PermissionError):b.product_bootstrap_rpc(NS(),'configure',dict(operator='trusted'))
        with self.assertRaises(PermissionError):b.product_bootstrap_rpc(self.make(),'configure',dict(operator='user'))
    def test_payload_cannot_replace_trust_paths_or_tokens(self):
        worker=self.make()
        for key in ('paths','canary_tokens','authorize_operator','source','native_library'):
            with self.assertRaises(ValueError):b.product_bootstrap_rpc(worker,'configure',dict(operator='trusted',**{key:'untrusted'}))
    def test_worker_model_replaced_rejects_authority(self):
        worker=self.make();worker.model_runner.model=object()
        with self.assertRaises(PermissionError):b.product_bootstrap_rpc(worker,'configure',dict(operator='trusted'))
    def test_local_initializer_cannot_rebind(self):
        worker=self.make()
        with self.assertRaises(RuntimeError):b.configure_product_bootstrap(worker,source='BLUE_COMBINATION',authorize_operator=self.auth,paths=['/other'],canary_tokens={})
    def test_console_dispatch(self):
        with patch.object(cli,'invoke',return_value=dict(schema=cli.SCHEMA,exit_code=4)) as call:
            self.assertEqual(cli.main(['fault','enable','--socket','/trusted.sock','--fault-id','99']),4)
            call.assert_called_once_with('/trusted.sock','enable','99')
    def test_legacy_catalog_labels_machine_readable_non_authority(self):
        from fault_injection.capabilities import describe_mechanism
        row=describe_mechanism('41.tail_column')
        self.assertEqual(row['catalog_scope'],'LEGACY_RESEARCH_NOT_PRODUCT')
        self.assertFalse(row['product_eligibility_authoritative'])

    def test_legacy_catalog_not_imported_by_product_path(self):
        from pathlib import Path
        root=Path('src/demoncat/runtime/vllm_ascend/worker_extension')
        for p in root.glob('product*.py'):
            self.assertNotIn('fault_injection.capabilities',p.read_text(encoding='utf-8'))

class SessionTests(unittest.TestCase):
    def test_dynamic_candidate_lifecycle_and_negative(self):
        from demoncat.runtime.vllm_ascend.worker_extension.product_cli import TrustedProductSession
        from unittest.mock import Mock
        row=dict(status='PRODUCT_CANDIDATE',mechanism='15.alias_detach',mode='PRODUCT_ACCEPTANCE_ONLY',product_qualified=False,range={},named_variant='alias_detach')
        c=Mock();c.capabilities.return_value=[dict(row,rank=0),dict(row,rank=1)]
        c.enable.return_value={};c.disable.return_value={};c.detach.return_value={}
        session=TrustedProductSession({'profile':c},authorize_operator=lambda p,a:p=='trusted')
        req=lambda action,**kw:dict(schema=cli.SCHEMA,action=action,**kw)
        self.assertEqual(session.dispatch(req('list'),principal='user')['exit_code'],4)
        self.assertEqual(session.dispatch(req('enable',mechanism='99'),principal='trusted')['exit_code'],4)
        self.assertEqual(session.dispatch(req('enable',mechanism='15.alias_detach'),principal='trusted')['exit_code'],0)
        c.attach.assert_not_called()
        self.assertEqual(session.dispatch(req('disable'),principal='trusted')['exit_code'],0)
        c.disable.assert_called_once();c.detach.assert_called_once();self.assertIsNone(session.active)
    def test_failed_enable_keeps_owner_for_recovery(self):
        from demoncat.runtime.vllm_ascend.worker_extension.product_cli import TrustedProductSession
        from unittest.mock import Mock
        c=Mock();c.capabilities.return_value=[dict(status='PRODUCT_CANDIDATE',mechanism='18.api_half')];c.enable.side_effect=RuntimeError('failure')
        s=TrustedProductSession({'p':c},authorize_operator=lambda p,a:True)
        r=s.dispatch(dict(schema=cli.SCHEMA,action='enable',mechanism='18.api_half'),principal='trusted')
        self.assertEqual(r['exit_code'],5);self.assertTrue(r['recovery_required']);self.assertIs(s.active,c)
    def test_typed_refusal_codes_reach_console_body(self):
        from demoncat.runtime.vllm_ascend.worker_extension.product_cli import TrustedProductSession
        from demoncat.runtime.vllm_ascend.worker_extension.product_errors import ProductContractRefused
        from unittest.mock import Mock
        c=Mock();c.capabilities.return_value=[dict(status='PRODUCT_CANDIDATE',mechanism='18.api_half')]
        c.enable.side_effect=ProductContractRefused('PRODUCT_ALL_RANK_PREPARE_FAILED',[
            dict(rank=i,status='PROVIDER_UNAVAILABLE',reason='NO_PERMIT') for i in range(2)])
        s=TrustedProductSession({'p':c},authorize_operator=lambda p,a:True)
        r=s.dispatch(dict(schema=cli.SCHEMA,action='enable',mechanism='18.api_half'),principal='trusted')
        self.assertEqual(r['exit_code'],4);self.assertEqual(r['refusal_codes'],['NO_PERMIT'])
        self.assertEqual([x['rank'] for x in r['refusals']],[0,1])
if __name__=='__main__':unittest.main()
