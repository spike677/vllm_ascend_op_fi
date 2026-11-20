"""Actual CPU cache writes/consumers, explicit CPU group, no NPU claims."""
import unittest
from unittest.mock import patch
from dataclasses import replace
from types import SimpleNamespace as NS
import test_kv41_task as fixture
from demoncat.runtime.vllm_ascend.worker_extension import postwrite41,batch_context
from demoncat.runtime.vllm_ascend.worker_extension.rms18_tp2 import TP2GroupContext
from demoncat.runtime.vllm_ascend.worker_extension.kv41_tp2 import TP2KV41TaskProvider

class KV41TP2Tests(unittest.TestCase):
    def make(self):
        f=fixture.KV41TaskTests();f.setUp();self.addCleanup(f.doCleanups)
        old=f.client.host._binding
        def bind(identity,config):
            result=old(identity,config)
            c=TP2GroupContext(dict(owner=f.client.owner,members_sha256='c'*64),identity,NS())
            f.worker._demoncat_kv41_serving_binding=replace(f.worker._demoncat_kv41_serving_binding,group_context=c)
            return result
        f.client.host._binding=bind
        def snapshot(p,actual,context,expected):
            v=batch_context.snapshot_first_decode_rows(p.runner,p.model,actual,context,expected,expected_max_seqs=1)
            return dict(v,logical_ids=[0],input_ids=[7])
        self.enterContext(patch.object(postwrite41.Postwrite41WorkerProvider,'snapshot_decode',snapshot))
        f.start();self.assertIs(type(f.provider),TP2KV41TaskProvider);return f
    def decision(self,p,rank):
        proof=p.group_proof()
        return dict(group=p.binding.group_context.contract,selected_rank=rank,normal_request_id=proof['normal_request_id'],request_fingerprint=proof['request_fingerprint'],roles={i:'injector' if i==rank else 'guard' for i in (0,1)})
    def execute(self,rank):
        f=self.make();f.through(3);p=f.provider
        p.commit_role(self.decision(p,rank));before=len(f.transport.opened);f.enqueue(3)
        state=f.ok(f.client.query());fault=state['evidence']['actual_requests'][-1]
        self.assertEqual(fault['new_fault_hits'],int(rank==0))
        if rank==1:
            self.assertEqual(len(f.transport.opened),before)
            self.assertTrue(fault['effect']['unchanged']);self.assertEqual(fault['effect']['native_writes'],0)
        else:self.assertGreater(fault['effect']['changed_elements'],0)
        f.ok(f.client.recover());f.enqueue(4);f.enqueue(5)
        state=f.ok(f.client.query());self.assertTrue(state['complete'])
        self.assertTrue(state['evidence']['matched_off_and_health_confirmed'])
        self.assertEqual((state['leases'],state['owned_hooks']),(0,0))
        self.assertEqual(p.rows[0]['model_output_sha256'],p.rows[-1]['model_output_sha256'])
        f.ok(f.client.finish());f.ok(f.client.unregister());self.assertEqual(f.client.detach()['status'],'DETACHED')
    def test_guard_has_no_native_write_and_all_phases_recover(self):self.execute(1)
    def test_injector_actual_native_effect_and_recovery(self):self.execute(0)
    def test_roles_cannot_precede_calibration_or_use_wrong_request(self):
        f=self.make()
        with self.assertRaisesRegex(RuntimeError,'AFTER_REAL_CALIBRATION'):f.provider.commit_role({})
        f.through(3);d=self.decision(f.provider,0);d['normal_request_id']='foreign'
        with self.assertRaisesRegex(RuntimeError,'CURRENT_NORMAL'):f.provider.commit_role(d)
        self.assertIsNone(f.provider.binding.group_context.decision)

if __name__=='__main__':unittest.main()
