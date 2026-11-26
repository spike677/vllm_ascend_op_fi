import os,threading,unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
from demoncat.runtime.vllm_ascend.worker_extension.product_errors import product_control_reply,ProductContractRefused
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig


class AttachRefusalNoopTests(unittest.TestCase):
    def client(self,size=2):
        c=ProductClient(Mock(),FaultConfig(18,'api_half'),profile='norm',operator='trusted',run_canary=Mock(),drain=Mock())
        c.members=[dict(rank=i) for i in range(size)];c.clients=[Mock() for _ in c.members]
        for child in c.clients:child.snapshot.return_value=dict(attachment=None,ownership_retained=False)
        def rpc(method,timeout=None,args=()):
            action=args[0]
            return [dict(rank=i,status='PRODUCT_BINDING_REFUSED',reason='CONTINUOUS_BINDING_ALREADY_INSTALLED' if action=='attach' else 'PRODUCT_ROLLBACK_OWNER_MISMATCH') for i in range(size)]
        c.executor.collective_rpc.side_effect=rpc
        return c

    def test_all_rank_existing_binding_preserves_exit4_and_never_touches_foreign_owner(self):
        for size in (1,2):
            c=self.client(size);reply=product_control_reply(c.enable)
            self.assertEqual(reply['exit_code'],4,reply)
            self.assertEqual(reply['code'],'PRODUCT_BINDING_ATTACH_FAILED')
            self.assertEqual(reply['refusal_codes'],['CONTINUOUS_BINDING_ALREADY_INSTALLED'])
            self.assertFalse(c.bound);self.assertFalse(c._enable_failure_pending)
            self.assertFalse(c.recovery['restored']);self.assertIsNone(c.recovery['global_resources_zero'])
            c.disable();c.disable();self.assertEqual(c.detach()['status'],'PRODUCT_DETACHED')
            self.assertEqual([x.kwargs['args'][0] for x in c.executor.collective_rpc.call_args_list],['attach'])
            c.drain.assert_not_called();c.run_canary.assert_not_called()
            for child in c.clients:child.detach.assert_not_called();child.prepare.assert_not_called()

    def test_real_worker_binding_guard_keeps_existing_binding_and_leases(self):
        from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import ContinuousBinding,ATTRIBUTE
        from demoncat.runtime.vllm_ascend.worker_extension.extension import DemoncatWorkerExtension
        c=self.client();workers=[];originals=[]
        for rank in range(2):
            model=NS(training=False);runner=NS(model=model,vllm_config=NS(cache_config=NS(enable_prefix_caching=False),kv_transfer_config=None,parallel_config=NS(tensor_parallel_size=2)))
            w=NS(rank=rank,model_runner=runner,demoncat_worker_identity=lambda rank=rank:dict(worker=dict(rank=rank)),
                 _demoncat_product_profiles={'norm':dict(mechanism='18.api_half',mode='PRODUCT_ACCEPTANCE_ONLY',canary_prompts=[[1,2]])})
            b=ContinuousBinding(w,model,'18.api_half',((1,2),),dict(owner='original-owner',members=c.members),dict(rank=rank),'cpu',os.getpid(),threading.get_ident())
            b.leases.add('original-lease');b.warmup_request=Mock();setattr(w,ATTRIBUTE,{'18.api_half':b})
            workers.append(w);originals.append(b)
        def rpc(method,timeout=None,args=()):
            return list(reversed([DemoncatWorkerExtension.demoncat_product_binding(w,*args) for w in workers]))
        c.executor.collective_rpc.side_effect=rpc
        with patch('demoncat.runtime.vllm_ascend.worker_extension.product_binding.authority_for',return_value=NS(authorize_operator=lambda *a:True)),patch('demoncat.runtime.vllm_ascend.worker_extension.continuous_model.validate_continuous_model',return_value='qwen35'):
            reply=product_control_reply(c.enable)
        self.assertEqual(reply['exit_code'],4,reply)
        c.disable();c.detach()
        for w,b in zip(workers,originals):
            self.assertIs(getattr(w,ATTRIBUTE)['18.api_half'],b)
            self.assertEqual(b.leases,{'original-lease'});b.warmup_request.close.assert_not_called()

    def test_partial_attach_does_not_skip_recovery(self):
        c=self.client();old=c.executor.collective_rpc.side_effect
        def rpc(*a,**kw):
            rows=old(*a,**kw)
            if kw['args'][0]=='attach':rows[0]=dict(rank=0,status='CONTINUOUS_BOUND_NOT_QUALIFIED')
            return rows
        c.executor.collective_rpc.side_effect=rpc
        self.assertEqual(product_control_reply(c.enable)['exit_code'],5)
        self.assertTrue(c.bound);self.assertTrue(c._enable_failure_pending)
        self.assertIn('rollback',[x.kwargs['args'][0] for x in c.executor.collective_rpc.call_args_list])

    def test_unknown_or_incomplete_rank_responses_do_not_skip_recovery(self):
        for kind in ('missing','duplicate','transport','other_refusal','internal_error'):
            c=self.client();old=c.executor.collective_rpc.side_effect
            def rpc(*a,**kw):
                rows=old(*a,**kw)
                if kw['args'][0]=='attach':
                    if kind=='missing':return rows[:1]
                    if kind=='duplicate':return [rows[0],rows[0]]
                    if kind=='transport':raise TimeoutError('lost reply')
                    if kind=='other_refusal':
                        for r in rows:r['reason']='UNKNOWN_POST_INSTALL_REFUSAL'
                    if kind=='internal_error':rows[0]['status']='PRODUCT_BINDING_ERROR'
                return rows
            c.executor.collective_rpc.side_effect=rpc
            self.assertEqual(product_control_reply(c.enable)['exit_code'],5,kind)
            self.assertTrue(c.bound,kind)

    def test_runtime_attachment_or_unknown_ownership_requires_cleanup(self):
        for state in (dict(attachment={'ticket':1},ownership_retained=True),dict(attachment=None),dict(attachment=None,ownership_retained=None)):
            c=self.client();c.clients[0].snapshot.return_value=state
            c.clients[0].detach.return_value=dict(status='CLEANUP_FAILED')
            self.assertEqual(product_control_reply(c.enable)['exit_code'],5)
            self.assertTrue(c.bound)

    def test_missing_permit_after_successful_attach_still_rolls_back(self):
        c=self.client();actions=[]
        def rpc(method,timeout=None,args=()):
            actions.append(args[0])
            if args[0]=='attach':return [dict(rank=i,status='CONTINUOUS_BOUND_NOT_QUALIFIED') for i in range(2)]
            if args[0]=='rollback':return [dict(rank=i,status='PRODUCT_BINDING_ROLLED_BACK',resources_zero=True) for i in range(2)]
            self.fail(args[0])
        c.executor.collective_rpc.side_effect=rpc
        c._canary=Mock(return_value=[dict(warmup_valid=True,observations=[dict(output=['stable'])]) for _ in range(2)])
        c._all=Mock(side_effect=ProductContractRefused('PRODUCT_ALL_RANK_PREPARE_FAILED',[
            dict(worker=dict(rank=i),status='PROVIDER_UNAVAILABLE',response=dict(provider_code='BLUE_ACCEPTANCE_PERMIT_NOT_BOUND')) for i in range(2)]))
        reply=product_control_reply(c.enable)
        self.assertEqual(reply['exit_code'],4);self.assertEqual(actions,['attach','rollback'])
        self.assertFalse(c.bound);self.assertTrue(c.recovery['resources_zero'])
        self.assertFalse(c.recovery['restored']);c.disable();c.detach()

    def test_later_failure_with_attach_like_code_still_rolls_back(self):
        c=self.client();actions=[]
        def rpc(method,timeout=None,args=()):
            actions.append(args[0])
            if args[0]=='attach':return [dict(rank=i,status='CONTINUOUS_BOUND_NOT_QUALIFIED') for i in range(2)]
            return [dict(rank=i,status='PRODUCT_BINDING_ROLLED_BACK',resources_zero=True) for i in range(2)]
        c.executor.collective_rpc.side_effect=rpc
        c._canary=Mock(side_effect=ProductContractRefused('PRODUCT_BINDING_ATTACH_FAILED',[
            dict(rank=i,status='PRODUCT_BINDING_REFUSED',reason='CONTINUOUS_BINDING_ALREADY_INSTALLED') for i in range(2)]))
        self.assertEqual(product_control_reply(c.enable)['exit_code'],4)
        self.assertEqual(actions,['attach','rollback'])
        self.assertNotIn('global_resources_zero',c.recovery)
