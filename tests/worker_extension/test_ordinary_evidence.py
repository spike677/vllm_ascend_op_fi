import unittest
from demoncat.runtime.vllm_ascend.worker_extension.ordinary_evidence import (
    bound_state,encoded,MAX_EVIDENCE_BYTES,MAX_PROVIDER_STATE_BYTES)

class OrdinaryEvidenceTests(unittest.TestCase):
    def state(self,payload):
        return dict(ordinary_task=True,registered=False,enabled=False,inflight=0,leases=0,
            native_loaded=False,native_registered=False,owned_hooks=0,hook_owned=False,
            replay_pending=False,replay_failed=False,evidence=payload,serving=payload)
    def test_small_evidence_bytes_unchanged(self):
        original=self.state({'actual_requests':[{'phase':'fault','value':0.1}]})
        bounded=bound_state(original)
        self.assertEqual(encoded(original['evidence']),encoded(bounded['evidence']))
        self.assertTrue(bounded['evidence_complete']);self.assertLess(len(encoded(bounded)),MAX_PROVIDER_STATE_BYTES)
    def test_evidence_overflow_is_not_success_and_preserves_cleanup_state(self):
        original=self.state({'complete':True,'huge':'x'*MAX_EVIDENCE_BYTES})
        bounded=bound_state(original)
        self.assertTrue(bounded['evidence_limit_error']);self.assertFalse(bounded['evidence']['complete'])
        self.assertFalse(bounded['evidence']['matched_off_and_health_confirmed'])
        self.assertTrue(original['evidence']['complete'])
        self.assertEqual(bounded['leases'],0);self.assertFalse(bounded['registered'])
        self.assertLess(len(encoded(bounded)),MAX_EVIDENCE_BYTES)
    def test_non_evidence_state_overflow_is_also_bounded(self):
        original=self.state({'complete':True});original['other']='x'*MAX_PROVIDER_STATE_BYTES
        self.assertFalse(bound_state(original)['evidence_complete'])
    def test_multibyte_wire_size_and_no_research_contract_rewrite(self):
        state=self.state({'huge':'数'*MAX_EVIDENCE_BYTES})
        self.assertFalse(bound_state(state)['evidence_complete'])
        state.pop('ordinary_task');self.assertIs(bound_state(state),state)
        self.assertFalse(bound_state(state,ordinary=True)['evidence_complete'])

import test_extension as extension_fixture
class OrdinaryEvidenceRpcTests(unittest.TestCase):
    setUp=extension_fixture.WorkerExtensionTest.setUp
    _attach=extension_fixture.WorkerExtensionTest._attach
    _prepare=extension_fixture.WorkerExtensionTest._prepare
    _check=extension_fixture.WorkerExtensionTest._check
    def test_overflow_blocks_next_execute_but_real_cleanup_can_release(self):
        self.worker.load_model();attached=self._attach();ticket=self._check(attached)
        self.provider.ordinary_only=True
        self.provider.state['evidence']={'large':'x'*MAX_EVIDENCE_BYTES}
        reply=self.worker.demoncat_fault_query(ticket)
        self.assertTrue(reply['recovery_required'])
        self.assertFalse(reply['provider_state']['evidence_complete'])
        self.assertLess(len(encoded(reply)),MAX_PROVIDER_STATE_BYTES)
        self.assertEqual(self.worker.demoncat_fault_rpc('execute',ticket)['reason'],'RECOVERY_REQUIRED')
        for action in ('abort','clean','unregister'):
            self.assertEqual(self.worker.demoncat_fault_rpc(action,ticket)['status'],'ACTION_COMPLETED')
        self.assertEqual(self.worker.demoncat_detach(attached)['status'],'DETACHED')

if __name__=='__main__':unittest.main()
