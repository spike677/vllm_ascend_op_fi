"""CPU fake-provider tests; these do not prove real CANN owner isolation."""
from copy import deepcopy
import unittest

from native_probe import verify_active_owner_isolation
from demoncat_fault39.provider import NativeError
from demoncat_fault39.schema import Fault39Config


class FakeProvider:
    def __init__(self):
        self.state = dict(owner='original', generation=7, enabled=True, mode=4,
                          inflight=0, leases=0, hits=0, status='ARMED')
        self.calls = []
        self.corrupt_on_rejection = False

    def snapshot(self):
        return deepcopy(self.state)

    def configure(self, owner, config, rank, device, **kwargs):
        self.calls.append(('configure', owner, kwargs))
        if self.corrupt_on_rejection:
            self.state['enabled'] = False
        if owner != self.state['owner']:
            raise NativeError(-4, 'configure')
        raise AssertionError('Test must never configure the active owner')

    def disable(self, owner, generation, timeout):
        self.calls.append(('disable', owner, generation))
        if owner != self.state['owner']:
            raise NativeError(-4, 'disable')
        if generation != self.state['generation']:
            raise NativeError(-5, 'disable')
        raise AssertionError('Test must never disable the active generation')

    def clean(self, owner, generation, timeout):
        self.calls.append(('clean', owner, generation))
        if owner != self.state['owner']:
            raise NativeError(-4, 'clean')
        raise AssertionError('Test must never release the active owner')


class ActiveOwnerIsolationTests(unittest.TestCase):
    def check(self, provider):
        return verify_active_owner_isolation(provider, 'original', 7, Fault39Config(), 0, 0, 'actual-signature')

    def test_expected_rejections_preserve_owner_until_actual_run(self):
        provider = FakeProvider()
        before = provider.snapshot()
        report = self.check(provider)
        self.assertEqual(provider.snapshot(), before)
        self.assertEqual([operation['native_code'] for operation in report['operations']], [-4, -4, -4, -5])
        self.assertEqual(report['status'], 'OWNER_REJECTIONS_VERIFIED_PENDING_RUN')
        self.assertFalse(report['owner_execution_verified'])
        self.assertEqual(provider.calls[0][2]['mode'], 'observer')
        self.assertEqual(provider.calls[0][2]['target_signature'], 'actual-signature')

    def test_rejection_cannot_hide_changed_owned_state(self):
        provider = FakeProvider(); provider.corrupt_on_rejection = True
        with self.assertRaisesRegex(RuntimeError, 'REJECTION_CHANGED_STATE'):
            self.check(provider)

    def test_wrong_native_error_does_not_pass_owner_check(self):
        provider = FakeProvider()
        def wrong(*args, **kwargs):
            raise NativeError(-3, 'configure')
        provider.configure = wrong
        with self.assertRaisesRegex(RuntimeError, 'WRONG_REJECTION'):
            self.check(provider)


if __name__ == '__main__':
    unittest.main()
