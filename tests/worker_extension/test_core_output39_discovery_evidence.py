"""Real Host/Runtime failed discovery and cleanup; CPU Native double only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments.postwrite_evidence import digest
import test_core_output39_group as fixture


class Output39DiscoveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreOutput39GroupTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.model.unused = torch.nn.Linear(4, 4, dtype=torch.bfloat16).eval()
        self.case.start()
        for index in range(5):
            self.case.request(index)
        self.case.enqueue(5)
        self.provider = self.case.provider

    def failed_collect(self):
        result = self.case.client.collect_replay()
        self.assertEqual(result['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(result['response']['error_type'], 'PostwriteRefused')
        self.assertEqual(self.provider.group_failure, 'POSTWRITE_INDEPENDENT_DISCOVERY_INCOMPLETE')
        self.assertTrue(result['ownership_retained'])
        return result

    def readback(self):
        result = self.case.client.query()
        self.assertEqual(result['status'], 'QUERY_OK')
        return result['response']['provider_state']

    def original(self):
        evidence = self.readback()['evidence']
        record = evidence['group_originals'][-1]
        self.assertEqual(record['action'], 'collect_failed')
        raw = Path(record['path']).read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (record['bytes'], record['sha256']))
        value = json.loads(raw)
        self.assertEqual(record['group_discovery_sha256'], digest(value['group']['discovery']))
        self.assertEqual(evidence['group_discovery']['diagnostic_sha256'], record['group_discovery_sha256'])
        self.assertEqual(evidence['group_discovery']['originals'], [record])
        return evidence, record, raw, value

    def test_failed_discovery_complete_original_hash_survives_public_cleanup(self):
        before = (self.case.model.forward_calls, len(self.case.transport.opened))
        self.failed_collect()
        evidence, record, raw, value = self.original()
        self.assertEqual(before, (self.case.model.forward_calls, len(self.case.transport.opened)))
        self.assertEqual(before, (6, 2))  # The prior C1 lossless and F only.
        group = value['group']
        self.assertEqual(group['discovery'], json.loads(json.dumps(self.provider.group.discovery_diagnostics)))
        self.assertEqual(group['independent_discovery'], json.loads(json.dumps(self.provider.group.discovery_reports)))
        self.assertEqual(group['actual_parameter_consumption'], self.provider.group.parameter_consumption)
        self.assertEqual(len(value['actual_requests']), 6)
        self.assertIsNone(group['coverage'])
        self.assertFalse(value['qualification_granted'] or group['qualification_granted'])
        self.assertEqual(value['failure'], 'POSTWRITE_INDEPENDENT_DISCOVERY_INCOMPLETE')
        self.assertEqual(value['collection_failure'], value['first_collection_failure'])
        self.assertEqual(evidence['first_collection_failure']['reason'], value['failure'])
        self.assertFalse(evidence['c2'] or evidence['c3'] or evidence['collection_failure_capture_pending'])
        unresolved = evidence['group_discovery']['unresolved_candidates']
        self.assertEqual([row['candidate_id'] for row in unresolved], ['unobserved-module:'+str(id(self.case.model.unused))])
        self.assertEqual(unresolved[0]['eligibility']['status'], 'NOT_OBSERVED')
        unused, = [row for row in group['discovery']['structural_noncall_inventory'] if row['path'] == 'unused']
        self.assertFalse(unused['structural_noncall'])
        self.assertEqual(unused['reason'], 'NO_ACTUAL_BOUNDARY_IN_HEALTHY_DISCOVERY')
        with patch.object(self.provider, '_capture', side_effect=AssertionError('query wrote originals')):
            self.readback()
        state = self.case.ok(self.case.client.unregister())
        self.assertEqual((state['leases'], state['owned_hooks'], state['inflight']), (0, 0, 0))
        self.assertFalse(self.provider.binding.leases)
        self.assertEqual(state['evidence']['group_originals'][-1], record)
        self.assertEqual(Path(record['path']).read_bytes(), raw)
        self.assertEqual(state['evidence']['group_discovery'], evidence['group_discovery'])
        self.assertEqual(state['evidence']['group_failure'], value['failure'])
        self.assertFalse(state['evidence']['c2'] or state['evidence']['c3'])
        self.assertEqual(self.case.client.detach()['status'], 'DETACHED')
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(self.case.forward(self.provider.session)).all())

    def test_query_returns_detached_discovery_metadata_and_references(self):
        self.failed_collect()
        evidence, record, raw, value = self.original()
        expected = deepcopy(evidence['group_discovery'])
        evidence['group_discovery']['unresolved_candidates'][0]['eligibility']['status'] = 'UNSUPPORTED'
        evidence['group_discovery']['structural_noncall_inventory'][0]['reason'] = 'FORGED'
        evidence['group_discovery']['originals'][0]['sha256'] = '0'*64
        self.assertEqual(self.readback()['evidence']['group_discovery'], expected)
        self.assertEqual(Path(record['path']).read_bytes(), raw)
        self.case.ok(self.case.client.unregister())

    def test_failed_artifact_write_retains_first_refusal_and_cleanup_retries(self):
        original_capture = self.provider._capture
        def unavailable(action, **kwargs):
            if action == 'collect_failed':
                raise OSError('CPU_ARTIFACT_TEMPORARILY_UNAVAILABLE')
            return original_capture(action, **kwargs)
        with patch.object(self.provider, '_capture', side_effect=unavailable):
            result = self.failed_collect()
            self.assertEqual(result['response']['error_type'], 'PostwriteRefused')
            state = self.readback()
            self.assertTrue(state['evidence']['collection_failure_capture_pending'])
            self.assertFalse(state['evidence']['group_discovery']['originals'])
            self.assertIn('CPU_ARTIFACT_TEMPORARILY_UNAVAILABLE', str(state['cleanup_failures']))
            refused = self.case.client.unregister()
            self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(refused['ownership_retained'] and self.provider.binding.leases)
            after_cleanup = self.readback()
            self.assertEqual((after_cleanup['leases'], after_cleanup['owned_hooks'], after_cleanup['inflight']), (0, 0, 0))
            self.assertFalse(self.provider.p._observations)
            self.assertTrue(after_cleanup['evidence']['collection_failure_capture_pending'])
        state = self.case.ok(self.case.client.unregister())
        self.assertFalse(state['evidence']['collection_failure_capture_pending'])
        self.assertEqual((state['leases'], state['owned_hooks'], state['inflight']), (0, 0, 0))
        record = state['evidence']['group_originals'][-1]
        raw = Path(record['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), record['sha256'])
        value = json.loads(raw)
        self.assertEqual(value['first_collection_failure']['reason'], 'POSTWRITE_INDEPENDENT_DISCOVERY_INCOMPLETE')
        self.assertFalse(value['group']['discovery']['candidates']['discovery_complete'])
        self.assertIn('CPU_ARTIFACT_TEMPORARILY_UNAVAILABLE', str(value['cleanup_failures']))
        self.assertFalse(state['evidence']['c2'] or state['evidence']['c3'])
        self.assertEqual((self.case.model.forward_calls, len(self.case.transport.opened)), (6, 2))

    def test_changed_failed_original_refuses_query_without_rewriting_it(self):
        self.failed_collect()
        _, record, raw, _ = self.original()
        path = Path(record['path'])
        path.write_bytes(raw+b' ')
        try:
            result = self.case.client.query()
            self.assertEqual(result['status'], 'NATIVE_STATE_UNAVAILABLE')
            with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_GROUP_ORIGINAL_CHANGED'):
                self.provider.query(owner=self.provider._ticket[0], generation=self.provider._ticket[1])
            self.assertEqual(path.read_bytes(), raw+b' ')
            self.assertTrue(self.case.client.snapshot()['ownership_retained'])
        finally:
            path.write_bytes(raw)
        self.case.ok(self.case.client.unregister())


if __name__ == '__main__':
    unittest.main()
