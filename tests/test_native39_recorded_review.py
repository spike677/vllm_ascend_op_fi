"""Negative audit-boundary checks; these do not simulate NPU acceptance."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments.c2c3_closure_20261117.review_native39_group_124 import (
    fault_effect_errors, statistics, target_key, metadata_scope_binding, sha)


class Native39RecordedReviewTests(unittest.TestCase):
    def test_cache_reuse_does_not_merge_actual_sites(self):
        delta = [{'key': ['signature', 123, 7], 'count': 1}]
        self.assertNotEqual(target_key(dict(site=[1, 'linear', 0], native_delta=delta)),
                            target_key(dict(site=[2, 'linear', 0], native_delta=delta)))

    def test_ambiguous_native_call_count_refused(self):
        with self.assertRaisesRegex(ValueError, 'ONE_NATIVE_RUN'):
            target_key(dict(site=[1, 'linear', 0], native_delta=[{'key': ['s', 1, 2], 'count': 2}]))

    def test_one_missing_numerical_effect_cannot_pass_all_target_review(self):
        # Original native execution and replacement counts can still be 113.
        # One unchanged target is sufficient to reject the whole effect gate.
        self.assertEqual(fault_effect_errors({'changed_outputs': 112}, 113),
                         ['PER_TARGET_NUMERICAL_EFFECT_INCOMPLETE'])
        self.assertEqual(fault_effect_errors({'changed_outputs': 113}, 113), [])
        self.assertTrue(fault_effect_errors({}, 0))

    def test_missing_capture_is_not_inherited_from_other_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = statistics(root / 'absent', root / 'audit')
            for item in result.values():
                self.assertEqual(item['status'], 'MISSING_RAW_CAPTURE')
                self.assertFalse(item['e5_statistics_comparable'])
                self.assertIsNone(item['first_exceedance'])
                self.assertTrue(item['missing_files'])
            saved = json.loads((root / 'audit/baseline_fault/summary.json').read_text())
            self.assertEqual(saved, result['baseline_fault'])
            with self.assertRaises(FileExistsError):
                statistics(root / 'absent', root / 'audit')

    def test_metadata_source_is_the_fourth_exact_scope_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);installed=root/'actual-installed';source=root/'frozen-source42'
            for folder in (installed,source):
                (folder/'fault_injection').mkdir(parents=True)
                (folder/'fault_injection/metadata_arrays.py').write_bytes(b'actual frozen metadata source\n')
            components=['a'*64,'b'*64,'c'*64]
            metadata=hashlib.sha256(b'actual frozen metadata source\n').hexdigest()
            stages={'c2':dict(scope=dict(implementation_sha256=sha([*components,metadata])))}
            result=metadata_scope_binding(installed,source,*components,stages)
            self.assertTrue(result['checks_passed'],result)
            self.assertEqual(result['implementation_sha256'],metadata)
            stages['c2']['scope']['implementation_sha256']=sha(components)
            result=metadata_scope_binding(installed,source,*components,stages)
            self.assertFalse(result['checks_passed'])
            self.assertIn('c2:METADATA_BOUND_IMPLEMENTATION_SCOPE_MISMATCH',result['errors'])

    def test_changed_installed_or_frozen_metadata_cannot_borrow_recorded_scope(self):
        for changed in ('actual-installed','frozen-source42','both'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);installed=root/'actual-installed';source=root/'frozen-source42'
                for folder in (installed,source):
                    (folder/'fault_injection').mkdir(parents=True)
                    (folder/'fault_injection/metadata_arrays.py').write_bytes(b'original')
                parts=['a'*64,'b'*64,'c'*64,hashlib.sha256(b'original').hexdigest()]
                stages={'c3':dict(scope=dict(implementation_sha256=sha(parts)))}
                for folder in (installed,source):
                    if changed in (folder.name,'both'):
                        (folder/'fault_injection/metadata_arrays.py').write_bytes(b'changed')
                result=metadata_scope_binding(installed,source,*parts[:3],stages)
                self.assertFalse(result['checks_passed'])
                if changed!='both':self.assertIn('METADATA_ARRAY_SOURCE_BYTES_MISMATCH',result['errors'])
                if changed!='frozen-source42':
                    self.assertIn('c3:METADATA_BOUND_IMPLEMENTATION_SCOPE_MISMATCH',result['errors'])


if __name__ == '__main__':
    unittest.main()
