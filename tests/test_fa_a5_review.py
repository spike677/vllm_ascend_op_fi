import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from experiments.c2c3_closure_20261117 import review_fa_a5_sequence_124 as review


class FaA5GateReviewTests(unittest.TestCase):
    def fixture(self):
        original = dict(card=5, pod_uid='33991897-e3ed-4910-aa2c-835b9fe682a0',
                        checkpoint_sha256='b570d0ea0d0280157a26d48c536a6aacff8cdf4fb55037f3b3882e85c1395414')
        proof = dict(status='NO_LIVE_OWNER_CORROBORATED', post_gate=dict(blockers=[original]),
                     oci_tasks_sha256=review.canonical([]), internal_allocation_clearance_proven=False,
                     driver_initialization_available_proven=False, exclusive_device_reservation=False)
        proof['evidence_sha256'] = review.canonical(proof)
        return dict(cards=[5], idle=True, blockers=[], oci_tasks=[], stale_checkpoint_review=proof,
                    stale_checkpoint_allocations=[dict(original_blocker=original,
                        release_evidence_sha256=proof['evidence_sha256'])])

    def test_exact_historical_scope_only(self):
        result = review.audit_gate(self.fixture())
        self.assertFalse(result['internal_allocation_clearance_proven'])
        self.assertFalse(result['driver_initialization_available_proven'])

    def test_changed_owner_or_hash_is_not_historical_authorization(self):
        for key, value in (('pod_uid', 'another-pod'), ('checkpoint_sha256', 'a' * 64), ('card', 2)):
            gate = self.fixture()
            gate['stale_checkpoint_allocations'][0]['original_blocker'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'AUTHORIZED_OWNER_OR_CHECKPOINT_DIFFERS'):
                review.audit_gate(gate)

    def test_digest_tampering_or_live_blocker_refuses(self):
        gate = self.fixture()
        gate['stale_checkpoint_review']['evidence_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'PROOF_HASH_DIFFERS'):
            review.audit_gate(gate)
        gate = self.fixture(); gate['blockers'] = [dict(reason='OPEN_DEVICE')]
        with self.assertRaisesRegex(ValueError, 'CARD_SCOPE_DIFFERS'):
            review.audit_gate(gate)


class FaA5PayloadReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.baseline = self.root / 'baseline'; self.baseline.mkdir()
        self.fault = self.root / 'fault'; self.fault.mkdir()
        self.payload = {'output': dict(kind='tuple', length=6)}
        data = struct.pack('<ff', 1.0, 0.0)
        for index in range(6):
            name = f'output-{index}'
            (self.baseline / (name + '.bin')).write_bytes(data)
            (self.fault / (name + '.bin')).write_bytes(data)
            self.payload[name] = dict(kind='tensor', shape=[2], dtype='torch.float32',
                                     bytes=8, sha256=hashlib.sha256(data).hexdigest(), finite=True)
        (self.baseline / 'call.json').write_text(json.dumps(dict(payload=self.payload)))

    def test_complete_exact_payload(self):
        rows = review.full_payload(self.fault, self.payload, self.baseline, np)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row['byte_exact_vs_baseline'] for row in rows.values()))

    def test_signed_zero_is_not_byte_exact_recovery(self):
        data = struct.pack('<ff', 1.0, -0.0)
        (self.fault / 'output-0.bin').write_bytes(data)
        self.payload['output-0']['sha256'] = hashlib.sha256(data).hexdigest()
        row = review.full_payload(self.fault, self.payload, self.baseline, np)['output-0']
        self.assertFalse(row['byte_exact_vs_baseline'])
        self.assertEqual(row['changed_elements'], 0)
        self.assertGreater(row['changed_bytes'], 0)

    def test_tampered_bytes_and_changed_shape_rejected(self):
        (self.fault / 'output-0.bin').write_bytes(b'corrupt!')
        with self.assertRaisesRegex(ValueError, 'PAYLOAD_IDENTITY_DIFFERS'):
            review.full_payload(self.fault, self.payload, self.baseline, np)
        self.payload['output-0']['shape'] = [1, 2]
        with self.assertRaisesRegex(ValueError, 'STRUCTURE_DIFFERS'):
            review.full_payload(self.fault, self.payload, self.baseline, np)

    def test_nonfinite_cannot_be_hidden_by_claimed_flag(self):
        data = struct.pack('<ff', float('nan'), 0.0)
        (self.fault / 'output-0.bin').write_bytes(data)
        self.payload['output-0']['sha256'] = hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(ValueError, 'NONFINITE_COMPLETE_PAYLOAD'):
            review.full_payload(self.fault, self.payload, self.baseline, np)

    def test_missing_leaf_is_not_complete_output(self):
        payload = copy.deepcopy(self.payload); del payload['output-5']
        with self.assertRaisesRegex(ValueError, 'COMPLETE_SIX_LEAF_OUTPUT_REQUIRED'):
            review.full_payload(self.fault, payload, self.baseline, np)


if __name__ == '__main__':
    unittest.main()
