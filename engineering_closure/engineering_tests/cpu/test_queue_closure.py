import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from engineering_closure.queue import load_batches, plan_queue, prepare_cases


DIRECTORY = Path(__file__).resolve().parents[2] / 'npu_execution'


class QueueClosureTests(unittest.TestCase):
    def test_all_mechanisms_remain_unscheduled_without_evidence(self):
        report = plan_queue(DIRECTORY)
        self.assertEqual(len(report['jobs']), 17)
        self.assertEqual(report['queued_npu_processes'], 0)
        self.assertTrue(all(job['status'] in ('PENDING_PREREQUISITES', 'IMPLEMENTATION_PENDING')
                            for job in report['jobs']))

    def test_missing_execution_path_never_becomes_ready(self):
        job = next(row for row in load_batches(DIRECTORY) if row['id'] == 'fia28-replay')
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'review.json'
            artifact.write_bytes(b'{}')
            receipts = [dict(job=job['id'], gate=gate, artifact=artifact.name,
                sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), reviewer='fixture',
                status='PASS', scope=job['scope']) for gate in job['requires']]
            result = plan_queue(DIRECTORY, receipts=receipts, evidence_root=directory)
            row = next(row for row in result['jobs'] if row['id'] == job['id'])
            self.assertEqual(row['status'], 'IMPLEMENTATION_PENDING')

    def test_gate_receipts_require_exact_scope_and_original_bytes(self):
        job = load_batches(DIRECTORY)[0]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'receipt.json'
            artifact.write_bytes(b'{"scope":"cpu fixture only"}')
            receipts = [dict(job=job['id'], gate=gate, artifact=artifact.name,
                sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                reviewer='CPU test', status='PASS', scope=job['scope']) for gate in job['requires']]
            report = plan_queue(DIRECTORY, receipts=receipts, evidence_root=directory)
            self.assertEqual(report['jobs'][0]['status'], 'READY_TO_PREPARE')
            changed = copy.deepcopy(receipts)
            changed[0]['scope']['cann'] = '9.1'
            self.assertEqual(plan_queue(DIRECTORY, receipts=changed, evidence_root=directory)
                             ['jobs'][0]['status'], 'PENDING_PREREQUISITES')
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'ARTIFACT'):
                plan_queue(DIRECTORY, receipts=receipts, evidence_root=directory)

    def test_existing_case_factory_preserves_complete_request_capture(self):
        cases = prepare_cases('worker18-statistics', {'source': '/reviewed/source'})
        self.assertEqual(len(cases), 3)
        self.assertEqual([case['id'] for case in cases],
            ['natural-worker18-stats-b', 'natural-worker18-stats-repeat', 'natural-worker18-stats-f'])
        self.assertTrue(all(case['review']['physical_cards'] == [2] for case in cases))

    def test_kv_cases_require_actual_build_and_keep_baseline_before_fault(self):
        options = dict(source='/reviewed/source', native='/reviewed/native',
            sha256='a'*64, build_fingerprint='b'*64, model='/models/qwen')
        cases = prepare_cases('41.tail_column', options)
        self.assertEqual([case['id'] for case in cases], ['postwrite41-c2-baseline', 'postwrite41-c2-fault'])
        self.assertIn('--expected-targets 28', cases[0]['command'])
        self.assertIn('--try-c3', cases[1]['command'])
        options['sha256'] = 'unknown'
        with self.assertRaises(ValueError):
            prepare_cases('41.tail_column', options)


if __name__ == '__main__':
    unittest.main()
