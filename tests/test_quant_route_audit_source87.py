"""Synthetic integer relations only; no production tensor payload or framework."""
import copy
import struct
import unittest

from experiments.worker_core_v11_20261118.audit_quant_source87_routes import (
    LAYOUTS, arithmetic, digest, sha,
)


def complete_record():
    counts = [1] * 48 + [0] * 16
    data = dict(topk_ids=list(range(48)), raw_indices=list(range(48)),
                expert_counts=counts, cumulative_counts=list(range(1, 49)) + [48] * 16,
                absolute_indices=list(range(48)))
    record = dict(parent_sequence=7, forward_id='synthetic-forward', snapshots={})
    for name, (dtype, shape, size, code) in LAYOUTS.items():
        raw = struct.pack('<' + code * size, *data[name])
        meta = dict(storage=dict(identity=100, bytes=len(raw)), object_identity=200)
        record['snapshots'][name] = dict(dtype=dtype, shape=shape, values=data[name],
            bytes=len(raw), sha256=sha(raw), metadata_before=copy.deepcopy(meta),
            metadata_after=copy.deepcopy(meta), epoch_before=2, epoch_after=2,
            parent_sequence=7, forward_id='synthetic-forward')
    record['token_route_proof'] = dict(source_to_sorted=list(range(48)),
        sorted_slots=[dict(destination=i, source_token=i // 6, route_slot=i % 6, expert=i)
                      for i in range(48)],
        expert_counts=counts, cumulative_counts=data['cumulative_counts'],
        native_registration_proven=False, routed_output_row_proof=False,
        native_qualified=False, c2=False, c3=False)
    refresh(record)
    return record


def refresh(record):
    """Keep checksums valid so negative fixtures exercise semantic checks."""
    for name, (_, _, size, code) in LAYOUTS.items():
        snap = record['snapshots'][name]
        raw = struct.pack('<' + code * size, *snap['values'])
        snap['sha256'] = sha(raw)
    record['token_route_proof']['evidence_sha256'] = digest(
        {key: value for key, value in record.items() if key != 'token_route_proof'})


class Source87RouteAuditTests(unittest.TestCase):
    def setUp(self):
        self.record = complete_record()

    def refuses(self, reason):
        refresh(self.record)
        with self.assertRaisesRegex(ValueError, reason):
            arithmetic(self.record)

    def test_valid_integer_relation_does_not_grant_qualification(self):
        self.assertEqual(set(arithmetic(self.record)), set(LAYOUTS))
        self.assertFalse(self.record['token_route_proof']['native_qualified'])

    def test_negative_index_not_hidden_by_absolute_value(self):
        self.record['snapshots']['raw_indices']['values'][-1] = -47
        self.refuses('NONNEGATIVE_INVERSE_PERMUTATION')

    def test_counts_must_match_actual_topk_histogram(self):
        self.record['snapshots']['expert_counts']['values'][0] = 2
        self.refuses('EXPERT_COUNT_MISMATCH')

    def test_cumulative_groups_must_match_counts(self):
        self.record['snapshots']['cumulative_counts']['values'][0] = 2
        self.refuses('CUMULATIVE_COUNT_MISMATCH')

    def test_same_token_duplicate_expert_is_rejected(self):
        self.record['snapshots']['topk_ids']['values'][1] = 0
        self.refuses('EXPERT_RANGE_OR_UNIQUENESS')

    def test_nested_snapshot_metadata_change_is_rejected(self):
        self.record['snapshots']['topk_ids']['metadata_after']['storage']['identity'] += 1
        self.refuses('CURRENT_SNAPSHOT_IDENTITY')

    def test_observed_write_epoch_change_is_rejected(self):
        self.record['snapshots']['topk_ids']['epoch_after'] += 1
        self.refuses('CURRENT_SNAPSHOT_IDENTITY')

    def test_relabelled_request_slot_is_rejected(self):
        self.record['token_route_proof']['sorted_slots'][0]['source_token'] = 1
        self.refuses('REPORTED_ARITHMETIC_PROOF_MISMATCH')

    def test_integer_result_cannot_grant_native_authority(self):
        self.record['token_route_proof']['native_qualified'] = True
        self.refuses('ARITHMETIC_IS_NOT_ADMISSION_AUTHORITY')


if __name__ == '__main__':
    unittest.main()
