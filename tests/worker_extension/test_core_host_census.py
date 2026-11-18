"""Host RPC ownership contracts; fixture readback is CPU-only."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import call_census
from worker_extension import test_core_host as host_fixtures


class Readback:
    def __init__(self, worker, identity, **settings):
        self.worker = deepcopy(identity)
        self.stopped = self.clean = self.fail_cleanup = False

    def start(self):
        return self.query(None, None)

    def query(self, worker, identity):
        return dict(worker=self.worker, qualification_granted=False,
                    releasable=self.clean, identity_valid=True, stop_requested=self.stopped,
                    resource_cleanup_confirmed=self.clean, cleanup_complete=self.clean,
                    inflight=False, observer_hooks=0 if self.clean else 2,
                    foreign_hooks=False, metadata_scope_active=not self.clean,
                    observation_status='FAILED', request_binding_valid=False)

    def stop(self, worker, identity):
        self.stopped = True
        return self.query(worker, identity)

    def cleanup(self, worker, identity):
        self.stop(worker, identity)
        if self.fail_cleanup:
            raise RuntimeError('actual owned fixture hook remains')
        self.clean = True
        return self.query(worker, identity)


class HostCensusTests(unittest.TestCase):
    setUp = host_fixtures.CoreHostTests.setUp
    _resolve = host_fixtures.CoreHostTests._resolve
    client = host_fixtures.CoreHostTests.client

    def start(self):
        self.capture = None
        def factory(*args, **kwargs):
            self.capture = Readback(*args, **kwargs)
            return self.capture
        replacement = patch.object(call_census, 'WorkerCallCensus', side_effect=factory)
        replacement.start()
        self.addCleanup(replacement.stop)
        client = self.client()
        reply = client.census_start(phase='prefill', expected={
            'internal_request_id': 'scheduled-request', 'prompt_token_ids': [1, 2]})
        self.assertEqual(reply['status'], 'CENSUS_SNAPSHOT', reply)
        self.assertIsNotNone(reply['census_ticket'])
        return client

    def test_observation_failure_can_release_verified_zero_resources(self):
        client = self.start()
        self.assertEqual(client.prepare()['response']['reason'], 'CENSUS_CLEANUP_REQUIRED')
        clean = client.census_cleanup()
        self.assertEqual(clean['status'], 'CENSUS_RELEASED')
        self.assertEqual(clean['response']['census']['observation_status'], 'FAILED')
        self.assertIsNone(clean['census_ticket'])
        self.assertEqual(client.prepare()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_failed_cleanup_retains_owner_and_explicit_detach_retries(self):
        client = self.start()
        self.capture.fail_cleanup = True
        failed = client.census_cleanup()
        self.assertEqual(failed['status'], 'CENSUS_RECOVERY_REQUIRED')
        self.assertTrue(failed['ownership_retained'])
        self.assertEqual(client.detach()['status'], 'RECOVERY_REQUIRED')
        self.capture.fail_cleanup = False
        clean = client.detach()
        self.assertEqual(clean['status'], 'DETACHED', clean)
        self.assertFalse(clean['ownership_retained'])

    def test_lost_start_reply_is_recovered_only_by_same_attachment(self):
        client = self.client()
        with patch.object(call_census, 'WorkerCallCensus', Readback):
            def lose(method, args, replies):
                if method == 'demoncat_census' and args[0] == 'start':
                    self.executor.after = None
                    raise TimeoutError('lost actual start reply')
                return replies
            self.executor.after = lose
            reply = client.census_start(phase='prefill', expected={})
            self.assertEqual(reply['status'], 'RPC_UNCERTAIN')
            self.assertIsNone(reply['census_ticket'])
            clean = client.detach()
            self.assertEqual(clean['status'], 'DETACHED', clean)
            self.assertFalse(clean['ownership_retained'])

    def test_forged_census_owner_reply_keeps_attachment_for_real_cleanup(self):
        client = self.start()
        def forge(method, args, replies):
            if method == 'demoncat_census':
                replies[0]['ticket']['owner'] = 'other'
                self.executor.after = None
            return replies
        self.executor.after = forge
        self.assertEqual(client.census_query()['status'], 'RPC_UNCERTAIN')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_malformed_start_settings_cannot_request_native_factory(self):
        client = self.client()
        with self.assertRaises(TypeError):
            client.census_start(phase='prefill', expected={}, native_observer_factory='remote code')
        self.assertEqual(self.executor.calls, [])
