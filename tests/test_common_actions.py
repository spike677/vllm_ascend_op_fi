import json
import unittest
import torch

from fault_injection.actions import independent_storage, replace_row_, scale_value
from fault_injection.core import UnsupportedCall
from fault_injection.effect_summary import numeric_change, actual_request_effect


class ActionsTest(unittest.TestCase):
    def test_scale_preserves_input_and_dtype(self):
        a = torch.tensor([0.125, -3.5, 42.], dtype=torch.float16)
        original = a.clone()
        result = scale_value(a, .5)
        self.assertTrue(torch.equal(result, original * .5))
        self.assertTrue(torch.equal(a, original))
        self.assertEqual(result.dtype, a.dtype)
        self.assertNotEqual(result.untyped_storage()._cdata, a.untyped_storage()._cdata)
        for gain in (True, float('inf'), float('nan')):
            with self.assertRaises(ValueError): scale_value(a, gain)

    def test_nonzero_offset_is_not_independent_storage(self):
        a = torch.arange(12).reshape(3, 4)
        with self.assertRaises(UnsupportedCall):
            independent_storage(a[:2], lambda value: a[1:])
        with self.assertRaises(UnsupportedCall):
            independent_storage(a, lambda value: value.float())

    def test_clone_and_supplied_conversion_are_each_once(self):
        a = torch.arange(12).reshape(3, 4).t()
        calls = []
        def conversion(value):
            calls.append(value)
            return value.contiguous()
        converted = independent_storage(a, conversion)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(converted, a))
        self.assertTrue(torch.equal(independent_storage(a), a))

    def test_distinct_storage_objects_wrapping_overlapping_memory_refuse(self):
        buffer = bytearray(32)
        a = torch.frombuffer(buffer, dtype=torch.uint8, count=16)
        b = torch.frombuffer(buffer, dtype=torch.uint8, count=16, offset=8)
        self.assertNotEqual(a.untyped_storage()._cdata, b.untyped_storage()._cdata)
        with self.assertRaisesRegex(UnsupportedCall, 'overlapping backing memory'):
            independent_storage(a, lambda value: b)

    def test_row_write_preserves_other_rows_and_original_source(self):
        source = torch.arange(12).reshape(3, 4)
        original = source.clone()
        target = independent_storage(source)
        returned = replace_row_(target, -1, source[0])
        self.assertIs(returned, target)
        self.assertTrue(torch.equal(target[:-1], source[:-1]))
        self.assertTrue(torch.equal(target[-1], source[0]))
        self.assertTrue(torch.equal(source, original))
        with self.assertRaises(ValueError): replace_row_(target, True, source[0])
        with self.assertRaises(ValueError): replace_row_(target, 3, source[0])
        with self.assertRaises(ValueError): replace_row_(target, 1, source[0].float())

    def test_integer_position_row(self):
        p = torch.tensor([7, 9, 11], dtype=torch.int64)
        result = independent_storage(p)
        replace_row_(result, -1, p[0])
        self.assertEqual(result.tolist(), [7, 9, 7])
        self.assertEqual(p.tolist(), [7, 9, 11])


class EffectTest(unittest.TestCase):
    def test_bounded_display_retains_missing_scope_and_partial_observation(self):
        values = numeric_change(torch.tensor([1.]), torch.tensor([.5]))
        event = dict(applied=True, original_execution_verified=True,
                     return_readback_exact=True, path='norm', ordinal=0, forward=0, consumers=[])
        event['effect'] = dict(comparison='SAME_ACTUAL_RETURN_BEFORE_AFTER',
            semantic_position={key: event[key] for key in ('path', 'ordinal', 'forward')}, values=values)
        row = dict(request_id='actual', phase='fault', invocation='forward:1', input={'sha256': 'a'*64},
                   mechanism={'events': [event]*12 + [dict(applied=True)]})
        report = actual_request_effect(row)
        self.assertEqual(report['status'], 'PARTIAL_EFFECT_OBSERVATION')
        self.assertEqual((report['applied_events'], report['matched_effects'], report['omitted_positions']), (13, 12, 4))
        self.assertEqual(len(report['positions']), 8)
        self.assertFalse(report['positions'][0]['actual_consumer_observed'])
        report['positions'][0]['values']['samples'].clear()
        self.assertEqual(len(values['samples']), 1)
        self.assertEqual(actual_request_effect(dict(row, request_id=None))['status'], 'EFFECT_NOT_VERIFIED')
        self.assertEqual(actual_request_effect(dict(row, phase='off', mechanism=None))['positions'], [])
        with self.assertRaises(ValueError): actual_request_effect(row, position_limit=9)

    def test_nonfinite_changes_are_typed_and_json_safe(self):
        before = torch.tensor([1., 2., 3., float('nan'), float('inf'), -float('inf'), 4.])
        after = torch.tensor([float('nan'), float('inf'), -float('inf'), float('nan'), 5., float('inf'), 3.])
        report = numeric_change(before, after)
        self.assertEqual(report['changed_elements'], 6)
        self.assertEqual(report['finite_to_nan'], 1)
        self.assertEqual(report['finite_to_positive_inf'], 1)
        self.assertEqual(report['finite_to_negative_inf'], 1)
        self.assertEqual(report['nonfinite_to_finite'], 1)
        self.assertEqual(report['max_abs_diff_finite'], 1.)
        self.assertEqual(len(report['samples']), 4)
        json.dumps(report, allow_nan=False)

    def test_finite_subtraction_overflow_is_not_nan_summary(self):
        report = numeric_change(torch.tensor([-1e308], dtype=torch.float64),
                                torch.tensor([1e308], dtype=torch.float64))
        self.assertEqual(report['finite_difference_overflow'], 1)
        self.assertIsNone(report['max_abs_diff_finite'])
        json.dumps(report, allow_nan=False)

    def test_integer_difference_does_not_lose_unit(self):
        a = torch.tensor([2**62, -2**63], dtype=torch.int64)
        b = torch.tensor([2**62+1, 2**63-1], dtype=torch.int64)
        report = numeric_change(a, b)
        self.assertEqual(report['changed_elements'], 2)
        self.assertEqual(report['max_abs_diff_finite'], 2**64-1)

    def test_chunk_boundaries_and_noncontiguous_inputs(self):
        a = torch.arange(70000, dtype=torch.float64).reshape(35000, 2).t()
        b = a.clone()
        b[-1, -1] += 2
        report = numeric_change(a, b, sample_limit=1)
        self.assertEqual(report['changed_elements'], 1)
        self.assertEqual(report['samples'][0]['flat_index'], 69999)
        self.assertEqual(report['max_abs_diff_finite'], 2.)

    def test_empty_and_unmatched_inputs(self):
        report = numeric_change(torch.empty(0), torch.empty(0))
        self.assertEqual(report['changed_elements'], 0)
        self.assertIsNone(report['max_abs_diff_finite'])
        with self.assertRaises(ValueError): numeric_change(torch.zeros(2), torch.zeros(3))
        with self.assertRaises(ValueError): numeric_change(torch.zeros(2), torch.zeros(2, dtype=torch.float64))
        with self.assertRaises(ValueError): numeric_change(torch.empty(2, device='meta'), torch.empty(2, device='meta'))
        with self.assertRaises(ValueError): numeric_change(torch.zeros(2), torch.zeros(2), sample_limit=9)


if __name__ == '__main__': unittest.main()
