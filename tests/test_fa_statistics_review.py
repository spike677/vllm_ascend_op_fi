from decimal import Decimal
import unittest

from experiments.c2c3_closure_20261117 import review_fa_statistics_124 as review


def document(value, *, dtype='torch.float32', field='Norm', operator='arbitrary.operator'):
    tensor = dict(type='torch.Tensor', dtype=dtype, shape=[1], Max=Decimal(1),
                  Min=Decimal(1), Mean=Decimal(1), Norm=Decimal(1))
    tensor[field] = value
    return {'data': {operator: {'output': [tensor]}}}


class FaStatisticsReviewTests(unittest.TestCase):
    def test_fixed_tolerance_masks_small_exact_change(self):
        result = review.direct_decimal_compare(document(Decimal('118.3081044305463')),
                                               document(Decimal('118.30810443395829')))
        self.assertEqual(len(result['exact_changes']), 1)
        self.assertEqual(result['exceedances'], [])
        self.assertFalse(result['exact_changes'][0]['reaches_0_1_percent'])

    def test_strict_boundary_and_zero_reference_without_epsilon(self):
        at = review.direct_decimal_compare(document(Decimal(0)), document(Decimal('1e-8')))
        over = review.direct_decimal_compare(document(Decimal(0)), document(Decimal('1.00000000000000000000000000001e-8')))
        self.assertEqual(at['exceedances'], [])
        self.assertEqual(len(over['exceedances']), 1)
        self.assertIsNone(over['exceedances'][0]['relative_change'])
        self.assertFalse(over['exceedances'][0]['reaches_0_1_percent'])

    def test_mean_cannot_satisfy_magnitude_goal(self):
        result = review.direct_decimal_compare(document(Decimal(1), field='Mean'), document(Decimal(2), field='Mean'))
        self.assertTrue(result['exceedances'])
        self.assertFalse(result['exceedances'][0]['reaches_0_1_percent'])

    def test_null_is_preserved_and_nan_is_incomparable(self):
        result = review.direct_decimal_compare(document(None), document(None))
        self.assertEqual(len(result['unobservable_fields']), 1)
        self.assertEqual(result['unobservable_fields'][0]['baseline'], None)
        result = review.direct_decimal_compare(document(Decimal('NaN')), document(Decimal('NaN')))
        self.assertEqual(len(result['incomparable_fields']), 1)
        self.assertIsNone(result['first_exceedance'])

    def test_boolean_numeric_extrema_require_boolean_dtype(self):
        self.assertEqual(review.classify(True, 'torch.bool', 'Max'), ('finite', Decimal(1)))
        self.assertEqual(review.classify(False, 'torch.float32', 'Min'), ('unsupported', None))
        self.assertEqual(review.classify(True, 'torch.bool', 'Mean'), ('unsupported', None))

    def test_first_change_is_not_forced_to_attention(self):
        b = document(Decimal(1), operator='earlier.add')
        f = document(Decimal(2), operator='earlier.add')
        b['data'].update(document(Decimal(1), operator='later.attention')['data'])
        f['data'].update(document(Decimal(3), operator='later.attention')['data'])
        result = review.direct_decimal_compare(b, f)
        self.assertEqual(result['first_exceedance']['identity'], 'earlier.add/output/0')

    def test_structure_mismatch_refuses_numeric_alignment(self):
        b, f = document(Decimal(1)), document(Decimal(2))
        f['data']['arbitrary.operator']['output'][0]['shape'] = [2]
        with self.assertRaisesRegex(ValueError, 'TENSOR_STRUCTURE_DIFFERS'):
            review.direct_decimal_compare(b, f)

    def test_signed_infinity_transition_kept(self):
        result = review.direct_decimal_compare(document(Decimal(1)), document(Decimal('Infinity')))
        self.assertEqual(result['first_exceedance']['fault'], '+inf')
        self.assertIsNone(result['first_exceedance']['relative_change'])


if __name__ == '__main__':
    unittest.main()
