"""CPU lifecycle contracts only; no distributed propagation or NPU claim."""
from dataclasses import replace
from copy import deepcopy
from unittest.mock import patch
import unittest
import torch
from test_rms18_task import Rms18TaskTests
from demoncat.runtime.vllm_ascend.worker_extension import rms18_tp2 as tp

class TP2Rms18Tests(Rms18TaskTests):
    def setUp(self):
        super().setUp()
        self.model.processor.use_all_gather=True
        self.enterContext(patch.object(tp.TP2Rms18Request,'_require_runner',return_value={'status':'CPU_FIXTURE'}))
        self.enterContext(patch.object(tp,'snapshot_tp2_fresh_prefill',side_effect=self.snapshot))

    def prepare_rank(self,rank):
        # Existing public host/registry still enforce owner and CPU-only binding.
        self.worker.rank=rank
        self.ok(self.client.prepare())
        old=self.worker._demoncat_worker_fault_runtime._leases['18.api_half'].provider
        identity=old.identity
        members=[dict(identity,rank=r) for r in range(2)]
        members[rank]=identity
        c=dict(owner='worker-group-cpu',group_generation=1,members=members,
            members_sha256=tp.digest(members),tp=2,mechanism='18.api_half',injection_policy='ONE_AUTOMATIC_RANK')
        binding=replace(old.binding,group_context=tp.TP2GroupContext(c,identity,object()))
        self.worker._demoncat_natural_tensor_serving_bindings['18.api_half']=binding
        provider=tp.TP2Rms18Provider(self.worker,self.model,identity)
        provider.bind(*old._ticket)
        self.worker._demoncat_worker_fault_runtime._leases['18.api_half'].provider=provider
        self.provider=provider
        for action in ('register','check','enable'):self.ok(getattr(self.client,action)())
        return c

    def decision(self,c):
        p=self.provider.group_proof()
        return dict(group=c,selected_rank=0,roles={0:'injector',1:'guard'},
            normal_request_id=p['normal_request_id'],request_fingerprint=p['request_fingerprint'])

    def test_tp2_injector_and_guard_each_restore(self):
        # Separate CPU fixtures are not an emulation of a TP collective.
        c=self.prepare_rank(0)
        normal=self.enqueue('normal')
        self.provider.commit_role(self.decision(c))
        fault=self.enqueue('fault')
        self.assertFalse(torch.equal(normal,fault))
        self.assertTrue(self.provider.fault_applied)
        self.assertTrue(torch.equal(normal,self.enqueue('off')))
        self.assertTrue(torch.equal(normal,self.enqueue('health')))
        state=self.finish()
        self.assertEqual(state['multi_rank']['role'],'injector')
        self.assertEqual((state['owned_hooks'],state['leases'],state['inflight']),(0,0,0))

    def test_tp2_guard_never_admits_write(self):
        c=self.prepare_rank(1); normal=self.enqueue('normal')
        self.provider.commit_role(self.decision(c))
        with patch.object(self.provider,'admit',side_effect=AssertionError('guard must not admit')):
            fault=self.enqueue('fault')
        self.assertTrue(torch.equal(normal,fault));self.assertFalse(self.provider.fault_applied)
        self.assertTrue(self.provider.rows[-1]['guard_boundary_unchanged'])
        self.enqueue('off');self.enqueue('health')
        state=self.finish();self.assertEqual(state['multi_rank']['role'],'guard')
        self.assertTrue(state['serving']['recovered'])

    def test_tp2_role_seal_rejects_wrong_request_and_repeat(self):
        c=self.prepare_rank(0);self.enqueue('normal');decision=self.decision(c)
        wrong=deepcopy(decision);wrong['normal_request_id']='foreign'
        with self.assertRaisesRegex(Exception,'NOT_BOUND'):self.provider.commit_role(wrong)
        self.assertIsNone(self.provider.binding.group_context.decision)
        self.provider.commit_role(decision)
        with self.assertRaisesRegex(Exception,'ONCE'):self.provider.commit_role(decision)

    def test_peer_recovery_after_local_complete_rechecks_off_health(self):
        c=self.prepare_rank(1);self.enqueue('normal');self.provider.commit_role(self.decision(c));self.enqueue('fault');self.enqueue('off');self.enqueue('health')
        self.assertTrue(self.provider.complete);self.ok(self.client.recover());self.assertEqual(self.provider.phase,'off');self.assertFalse(self.provider.complete)
        self.enqueue('recovery-off');self.enqueue('recovery-health');state=self.finish();self.assertTrue(state['serving']['recovered'])

    def test_recover_after_fault_finishes_host_plan(self):
        c=self.prepare_rank(0);self.enqueue('normal');self.provider.commit_role(self.decision(c));self.enqueue('fault')
        self.ok(self.client.recover());self.enqueue('recovery-off');self.enqueue('recovery-health')
        reply=self.client.query();self.assertEqual(reply['status'],'QUERY_OK')
        state=reply['response']['provider_state']
        self.assertTrue(state['serving']['complete'])
        self.assertEqual(state['serving']['next_phase'],'complete')
        self.finish()

    def test_tp2_fault_without_agreement_refuses(self):
        self.prepare_rank(0);self.enqueue('normal')
        with self.assertRaisesRegex(Exception,'ROLE_DECISION_REQUIRED'):
            _=self.provider.injection_arm
        self.assertFalse(self.provider.fault_applied)

if __name__=='__main__':unittest.main()
