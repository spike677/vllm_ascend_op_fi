import json
from pathlib import Path
import unittest
from tools.build_library_coverage import render, progress_counts


class LedgerTests(unittest.TestCase):
    def test_progress_is_independent_and_recomputed(self):
        _,data=self.load()
        before=progress_counts(data)
        row=data['rows'][8]
        category=row['category']
        old=row['progress_state']
        row['progress_state']='ADAPTER_READY_UNTESTED'
        after=progress_counts(data)
        self.assertEqual(after[old],before[old]-1)
        self.assertEqual(after['ADAPTER_READY_UNTESTED'],before.get('ADAPTER_READY_UNTESTED',0)+1)
        self.assertEqual(row['category'],category)
        self.assertEqual(sum(after.values()),41)
        render(data)

    def test_missing_progress_refused(self):
        _,data=self.load()
        del data['rows'][0]['progress_state']
        with self.assertRaisesRegex(ValueError,'PROGRESS_STATE'):render(data)

    def load(self):
        p=Path(__file__).resolve().parents[1]/'docs/worker_core_v1/fault_library_coverage_20261124.json'
        return p,json.loads(p.read_bytes())
    def test_canonical_41_rows_and_render_match(self):
        p,data=self.load()
        self.assertEqual(render(data).encode(),p.with_suffix('.md').read_bytes())
        self.assertEqual([r['row'] for r in data['rows']],list(range(1,42)))
    def test_missing_or_duplicate_row_refused(self):
        _,d=self.load();d['rows'][7]=d['rows'][6]
        with self.assertRaisesRegex(ValueError,'41_LIBRARY'):render(d)
    def test_namespace_corrections_are_not_undone(self):
        _,d=self.load();rows={r['row']:r for r in d['rows']}
        self.assertIn('CP',str(rows[8]))
        self.assertIn('15',str(rows[27]))
        self.assertIn('20',str(rows[41]))
        self.assertIn('39',str(rows[39]))
        self.assertIn('ConvTranspose',str(rows[17]))
