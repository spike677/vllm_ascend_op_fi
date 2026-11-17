"""Python ownership cannot stand in for Native ABI/state evidence."""
import copy
from types import SimpleNamespace
import unittest

from demoncat.runtime.vllm_ascend.worker_extension.runtime import WorkerFaultRuntime


class BoundaryStateTests(unittest.TestCase):
    def lease(self, state, boundary='python_hook'):
        provider = SimpleNamespace(execution_boundary=boundary,
                                   query=lambda **_: copy.deepcopy(state))
        return SimpleNamespace(provider=provider, native_owner='actual-owner', generation=2,
                               replay_pending=False, recovery_required=False)

    def state(self):
        return dict(execution_boundary='python_hook', state_source='OWNED_PYTHON_HOOKS',
                    hook_state_available=True, hook_owned=False, owner='actual-owner', generation=2,
                    registered=False, enabled=False, inflight=0, leases=0,
                    native_loaded=False, native_registered=False)

    def test_python_ownership_is_not_labeled_native(self):
        state = WorkerFaultRuntime._fresh(self.lease(self.state()))
        self.assertTrue(WorkerFaultRuntime._inert(state))
        result = WorkerFaultRuntime._state_reply(state)
        self.assertIsNone(result['native'])
        self.assertEqual(result['provider_state']['execution_boundary'], 'python_hook')
        state['hook_owned'] = True
        self.assertFalse(WorkerFaultRuntime._inert(state))

    def test_python_receipt_cannot_weaken_native_admission(self):
        self.assertIsNone(WorkerFaultRuntime._fresh(self.lease(self.state(), 'native')))
        native = dict(native_loaded=True, native_registered=False, registered=False,
                      enabled=False, inflight=0, leases=0)
        self.assertIsNotNone(WorkerFaultRuntime._fresh(self.lease(native, 'native')))
        self.assertIsNone(WorkerFaultRuntime._fresh(self.lease(native)))

    def test_missing_stale_or_false_ownership_refuses(self):
        for key, value in [('owner', 'other'), ('generation', 1), ('generation', True),
                           ('hook_state_available', False), ('hook_owned', 0),
                           ('native_loaded', True), ('native_registered', True),
                           ('execution_boundary', 'native'), ('inflight', True),
                           ('leases', -1), ('enabled', 0)]:
            with self.subTest(key=key, value=value):
                state = self.state(); state[key] = value
                self.assertIsNone(WorkerFaultRuntime._fresh(self.lease(state)))
        for key in ('owner', 'generation', 'hook_owned', 'hook_state_available', 'state_source'):
            state = self.state(); del state[key]
            self.assertIsNone(WorkerFaultRuntime._fresh(self.lease(state)))

    def test_inflight_and_registered_hooks_prevent_release(self):
        for update in ({'inflight': 1}, {'leases': 1}, {'enabled': True},
                       {'registered': True}, {'hook_owned': True}):
            self.assertFalse(WorkerFaultRuntime._inert(dict(self.state(), **update)))
        self.assertIsNone(WorkerFaultRuntime._fresh(self.lease(self.state(), 'unknown')))


if __name__ == '__main__':
    unittest.main()
