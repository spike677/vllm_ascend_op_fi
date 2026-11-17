"""Failure-sensitive contracts; CPU receipts never qualify HCCL or injection."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from fault_injection.collective25 import ReduceScatterSession
from experiments.c2c3_closure_20261117.collective25_closure_probe import RankReceipts, case_specs, make_source
from experiments.c2c3_closure_20261117.collective25_supervisor import assess


class Collective25Closure(unittest.TestCase):
    def test_reviewed_smoke_has_valid_hccl_timeout_and_frozen_source(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'experiments/c2c3_closure_20261117/collective25_queue_smoke.json'
        for case in json.loads(path.read_text()):
            # CANN 8.5.1 rejected 60 with EI0001 on the actual first attempt.
            self.assertTrue(120 <= int(case['env']['HCCL_CONNECT_TIMEOUT']) <= 7200)
            self.assertEqual(case['review']['expected_hits'], 0)
            self.assertEqual(case['review']['physical_cards'], [2, 5])
            for relative, expected in case['review']['build_fingerprint']['files'].items():
                self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), expected)

    def test_release_requires_close_and_drops_local_group_observations(self):
        session = ReduceScatterSession(torch.nn.Identity(), simulation=True)
        session.observations.append(dict(group=object()))
        with self.assertRaisesRegex(RuntimeError, 'Close'):
            session.release_calibration()
        session.close()
        receipt = session.release_calibration()
        self.assertEqual(receipt['observations_released'], 1)
        self.assertEqual(session.observations, [])
        self.assertFalse(receipt['group_drain_claimed'])

    def test_missing_rank_times_out_and_same_checkpoint_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            receipts = RankReceipts(folder, rank=0, world=2, timeout=.01)
            with self.assertRaisesRegex(TimeoutError, 'RANK_CHECKPOINT_TIMEOUT'):
                receipts.exchange('drain', {})
            with self.assertRaisesRegex(RuntimeError, 'RANK_CHECKPOINT_REUSE'):
                receipts.exchange('drain', {})

    def test_inputs_are_finite_deterministic_and_original_smoke_is_preserved(self):
        for spec in case_specs('bounded'):
            for rank in range(2):
                first = make_source(torch, spec, rank, 2, 'cpu')
                self.assertTrue(torch.isfinite(first).all())
                self.assertTrue(torch.equal(first, make_source(torch, spec, rank, 2, 'cpu')))
        spec = case_specs('smoke')[0]
        self.assertTrue(torch.equal(make_source(torch, spec, 1, 2, 'cpu'),
                                    torch.arange(256) / 32 + 2 / 8))

    def test_shutdown_requires_both_native_callbacks_and_real_launcher_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for rank in range(2):
                (folder / f'rank{rank}.json').write_text(json.dumps(dict(
                    pid=rank + 100, status='NO_REPLAY_SAMPLE', cases=[dict(off_exact=True)],
                    group_drain=dict(status='CROSS_RANK_DRAIN_RETURNED'))))
                stages = ['destroy_process_group_returned', 'main_returned',
                          'native_shutdown_returned', 'python_atexit_tail']
                (folder / f'rank{rank}.lifecycle.jsonl').write_text(''.join(
                    json.dumps(dict(stage=stage)) + '\n' for stage in stages) +
                    json.dumps(dict(stage='native_shutdown_boundary', synchronize_success=True)) + '\n')
            passed = assess(folder, 0, False)
            self.assertTrue(passed['lifecycle_pass'])
            self.assertFalse(passed['c2_pass'] or passed['c3_pass'] or passed['actual_fault_recovery_proven'])
            self.assertFalse(assess(folder, 1, False)['lifecycle_pass'])
            self.assertFalse(assess(folder, 0, True)['lifecycle_pass'])
            lifecycle = folder / 'rank1.lifecycle.jsonl'
            original = lifecycle.read_text()
            lifecycle.write_text(original.replace('"synchronize_success": true', '"synchronize_success": false'))
            self.assertFalse(assess(folder, 0, False)['lifecycle_pass'])
            (folder / 'rank1.lifecycle.jsonl').write_text(json.dumps(dict(stage='main_returned')) + '\n')
            self.assertFalse(assess(folder, 0, False)['lifecycle_pass'])


if __name__ == '__main__':
    unittest.main()
