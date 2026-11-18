"""Timing originals are cost evidence only, never qualification authority."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.c2c3_closure_20261117 import providers26_isolated_budget as budget


def cost_fixture(case, folder):
    phases = ['inventory_1', 'inventory_2', *budget.QUALIFICATION]+['group_fault', 'group_off']*29+['health']
    result = dict(stage='c2_isolated_once', mechanism='26.fia_destination', status='NOT_QUALIFICATION',
        c2=False, c3=False, qualification_granted=False, isolated_plan=dict(full_native_writer_count=30),
        requests=[dict(phase=phase, complete_cycle_seconds=(1 if index < 5 else 25 if index == 63 else 12))
                  for index, phase in enumerate(phases)])
    controller = dict(status='NOT_QUALIFICATION', cases=[dict(id='providers26-core-fia-smol-c2_isolated_once',
        started_utc='2026-11-18T11:40:00+00:00', ended_utc='2026-11-18T11:55:00+00:00')])
    paths = []
    for name, document, constant in (('result', result, 'RESULT_SHA256'), ('controller', controller, 'CONTROLLER_SHA256')):
        path = folder/(name+'.json')
        raw = json.dumps(document).encode()
        path.write_bytes(raw)
        case.enterContext(patch.object(budget, constant, hashlib.sha256(raw).hexdigest()))
        paths.append(path)
    return budget.load_cost_reference(*paths), paths


class IsolatedTransitionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.reference, self.paths = cost_fixture(self, self.folder)
        self.plan = dict(full_native_writer_count=30, c2_group_requests=64, c3_group_requests=9,
            combined_group_requests=73, target_order=['target-'+str(index) for index in range(29)])

    def rows(self, run='c2', scale=1):
        return [dict(run=run, phase=row['phase'], complete_cycle_seconds=row['seconds']*scale)
                for row in self.reference['cycles'][:5]]

    def test_actual_late_cycles_and_both_raw_shas_bind_cost_without_success_flags(self):
        value = budget.transition_budget(self.rows(), self.plan, self.reference, elapsed_seconds=114)
        self.assertFalse(value['refusal_reasons'])
        self.assertEqual(value['observed_late_cycle_seconds'], 25)
        self.assertGreater(value['forecast_remaining_seconds'], 1200)
        self.assertLess(value['generation_forecast_seconds']['c2'], 1200)
        self.assertLess(value['generation_forecast_seconds']['c3'], 1200)
        self.assertLess(value['projected_outer_seconds'], 1800)
        self.assertFalse(self.reference['qualification_granted'] or self.reference['used_success_flags'])
        self.assertEqual(value['outside_probe_startup_exit_disk_margin_seconds'], 120)
        self.assertEqual(len(value['cost_originals']), 2)

    def test_missing_corrupt_or_modified_original_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'COST_ORIGINAL_REQUIRED'):
            budget.load_cost_reference(None, self.paths[1])
        self.paths[0].write_bytes(self.paths[0].read_bytes()+b' ')
        with self.assertRaisesRegex(RuntimeError, 'COST_ORIGINAL_SHA_CHANGED'):
            budget.load_cost_reference(*self.paths)

    def test_source_cycle_or_controller_time_inconsistency_is_not_a_cost_reference(self):
        for name in ('result', 'controller'):
            with self.subTest(name=name):
                reference, paths = cost_fixture(self, self.folder)
                target = paths[0 if name == 'result' else 1]
                doc = json.loads(target.read_text())
                if name == 'result':
                    doc['requests'].pop()
                else:
                    doc['cases'][0]['ended_utc'] = '2026-11-18T11:40:01+00:00'
                raw = json.dumps(doc).encode()
                target.write_bytes(raw)
                with patch.object(budget, 'RESULT_SHA256' if name == 'result' else 'CONTROLLER_SHA256', hashlib.sha256(raw).hexdigest()):
                    with self.assertRaisesRegex(RuntimeError, 'COST_REQUESTS|COST_CLOCK'):
                        budget.load_cost_reference(*paths)

    def test_returned_reference_mutation_cannot_lower_the_cost(self):
        changed = deepcopy(self.reference)
        changed['cycles'][-1]['seconds'] = .01
        with self.assertRaisesRegex(RuntimeError, 'UNCHANGED_DERIVATION_REQUIRED'):
            budget.transition_budget(self.rows(), self.plan, changed, elapsed_seconds=114)

    def test_continuous_outer_clock_cannot_reset_at_new_generation(self):
        rows = self.rows()+self.rows('c3')
        value = budget.transition_budget(rows, self.plan, self.reference, elapsed_seconds=1550, run='c3')
        self.assertEqual(value['c3_remaining_requests'], 4)
        self.assertIsNone(value['generation_forecast_seconds']['c2'])
        self.assertIn('ISOLATED_C3_ROOT_CASE_1800_SECOND_BUDGET_UNPROVEN', value['refusal_reasons'])

    def test_c3_current_late_cost_is_compared_with_actual_late_history_not_early_subsecond_cost(self):
        rows = self.rows()+self.rows('c3', scale=25)
        value = budget.transition_budget(rows, self.plan, self.reference, elapsed_seconds=1000, run='c3')
        self.assertFalse(value['refusal_reasons'])
        self.assertEqual(value['qualification_cost_baseline'], 'SOURCE82_LATE_COMPLETE_CYCLE')
        self.assertEqual(value['qualification_cost_baseline_seconds'], [25]*3)
        self.assertEqual(value['observed_late_cycle_seconds'], 25)
        self.assertEqual(value['actual_cycle_slowdown'], 1)

    def test_actual_slowdown_or_complete_large_denominator_refuses_without_shortening_plan(self):
        for rows, plan in ((self.rows(scale=2), self.plan), (self.rows(), dict(self.plan, full_native_writer_count=135))):
            value = budget.transition_budget(rows, plan, self.reference, elapsed_seconds=114)
            self.assertEqual(value['status'], 'REFUSED_BEFORE_F')
            self.assertIn('ISOLATED_C3_PER_GENERATION_1200_SECOND_BUDGET_UNPROVEN', value['refusal_reasons'])
        self.assertEqual(len(self.plan['target_order']), 29)

    def test_missing_cycles_reordered_phases_and_postfault_budget_checks_refuse(self):
        for rows in (self.rows()[:-1], list(reversed(self.rows())), self.rows()+[dict(run='c2', phase='group_fault', complete_cycle_seconds=1)]):
            with self.assertRaisesRegex(RuntimeError, 'FIVE_COMPLETE_CURRENT_GENERATION'):
                budget.transition_budget(rows, self.plan, self.reference, elapsed_seconds=114)

    def test_plan_cannot_drop_c3_duplicate_targets_or_change_request_count(self):
        for changed in (dict(self.plan, c3_group_requests=0), dict(self.plan, combined_group_requests=64),
                        dict(self.plan, target_order=['same']*29)):
            with self.assertRaisesRegex(RuntimeError, 'EXACT_COMPLETE_C2_C3_PLAN'):
                budget.transition_budget(self.rows(), changed, self.reference, elapsed_seconds=114)


if __name__ == '__main__':
    unittest.main()
