"""TP2 protocol/Native ownership CPU doubles; no numerical/NPU qualification."""
import unittest
from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace as NS
import test_quant_projection_runtime as fixture
from demoncat.runtime.vllm_ascend.worker_extension import quant_tp2 as tp
from demoncat.runtime.vllm_ascend.worker_extension.providers.quant import QuantProvider

class TP2QuantTests(unittest.TestCase):
    def make(self):
        f=fixture.OrdinaryQuantTests();f.setUp();self.addCleanup(f.doCleanups)
        f.session.close(abort=True)
        for _,module,_,_ in f.session.candidates:module.tp_size=2
        f.model.processor.use_all_gather=True
        identity=deepcopy(f.fixture.identity)
        c=NS(identity=identity,contract=dict(owner='ordinary-owner',group_generation=1,members_sha256='c'*64),tp_group=object(),decision=None)
        self.enterContext(patch.object(tp.TP2QuantSession,'_require_runner'))
        self.enterContext(patch.object(tp.TP2QuantSession,'_snapshot_decode',lambda s,a,c:fixture.ordinary.serving.snapshot_first_decode(s.runner,s.model,a,c,s.expected)))
        f.session=tp.TP2QuantSession(f.worker,f.model,root='CPU_ONLY',build_sha256='b'*64,owner='ordinary-owner',generation=4,worker_identity=identity,group_context=c)
        self.addCleanup(lambda:setattr(c,'decision',None))
        f.provider=QuantProvider(f.worker,f.model,identity)
        self.enterContext(patch('fault_injection.providers.resolve_runtime',return_value=tp.TP2QuantBinding(lambda *a,**k:f.session)))
        for action in ('prepare','register','check','enable'):f.dispatch(action)
        return f,c
    def decision(self,f,c,rank):
        proof=f.session.group_proof()
        return dict(group=deepcopy(c.contract),selected_rank=rank,normal_request_id=proof['normal_request_id'],request_fingerprint=proof['request_fingerprint'],roles={0:'injector' if rank==0 else 'guard',1:'injector' if rank==1 else 'guard'})
    def test_injector_and_guard_native_modes_and_off(self):
        # Distinct fixture lifetimes avoid two process-global API owners.
        for selected in (0,1):
            with self.subTest(selected=selected):
                f,c=self.make()
                for _ in range(4):f.phase()
                normal_ids=list(f.session.group_proof()['normal_request_id'])
                self.assertEqual(normal_ids,['ordinary-0-'+str(i) for i in range(8)])
                f.session.commit_role(self.decision(f,c,selected));f.phase()
                state=f.provider.query(owner='ordinary-owner',generation=4);self.assertEqual(state['multi_rank']['normal_request_id'],normal_ids)
                self.assertEqual(f.binding.selected_modes,[1,2,4,3] if selected==0 else [1,2,4])
                if selected==1:
                    self.assertTrue(state['serving']['guard_boundary_unchanged'])
                    self.assertIsNone(state['serving']['fault_effect'])
                f.phase();f.dispatch('finish');self.assertTrue(f.session.recovered)
                self.assertFalse(f.session.installed or f.binding.config)
                self.doCleanups()
    def test_role_requires_actual_calibration_and_matching_fingerprint(self):
        f,c=self.make()
        with self.assertRaisesRegex(RuntimeError,'AFTER_CURRENT_CALIBRATION'):f.session.commit_role({})
        for _ in range(4):f.phase()
        decision=self.decision(f,c,0);decision['request_fingerprint']='f'*64
        with self.assertRaisesRegex(RuntimeError,'CURRENT_NORMAL'):f.session.commit_role(decision)
        self.assertIsNone(c.decision)
    def test_installer_binds_public_owner_to_distinct_runtime_native_owner(self):
        model=object();identity={'rank':0};lease=NS(owner='worker-group-a',native_owner='private-native-a',generation=7)
        worker=NS(model_runner=NS(model=model),_demoncat_worker_fault_runtime=NS(_leases={'quant.batch_shared_scale':lease}))
        context=NS(identity=identity,contract={'owner':'worker-group-a'})
        with patch.object(tp,'validate_tp2_group',return_value=context), patch.dict(tp.os.environ,{'VLLM_ENABLE_V1_MULTIPROCESSING':'0','TASK_QUEUE_ENABLE':'0'}), patch('fault_injection.providers.register_runtime'), patch.object(tp,'TP2QuantSession') as ctor:
            tp.install_quant_tp2_binding(worker,root='CPU',build_sha256='b'*64,group_contract={})
            factory=worker._dcat_quant_binding(model).make_session
            factory(worker,model,owner='private-native-a',generation=7,worker_identity=identity)
            self.assertEqual(ctor.call_count,1)
            for owner,generation in [('worker-group-a',7),('private-native-a',8)]:
                with self.assertRaisesRegex(RuntimeError,'GROUP_OWNER_REQUIRED'):
                    factory(worker,model,owner=owner,generation=generation,worker_identity=identity)
            lease.owner='foreign-public-owner'
            with self.assertRaisesRegex(RuntimeError,'GROUP_OWNER_REQUIRED'):
                factory(worker,model,owner='private-native-a',generation=7,worker_identity=identity)
            self.assertEqual(ctor.call_count,1)

    def test_guard_must_finish_real_off_after_group_recover(self):
        f,c=self.make()
        for _ in range(4):f.phase()
        f.session.commit_role(self.decision(f,c,1));f.phase()
        f.dispatch('recover');self.assertTrue(f.session.recovery_only and f.session.installed)
        with self.assertRaisesRegex(RuntimeError,'OFF_NOT_VERIFIED'):f.dispatch('finish')
        f.phase();f.dispatch('finish');self.assertTrue(f.session.recovered)

if __name__=='__main__':unittest.main()
