import copy
import unittest

from experiments.c2c3_closure_20261117.review_moe8_group_124 import check_native, probe_digest


def native(injecting):
    arguments = {key: dict(object_id=i, storage_id=i+10, shape=[2], dtype='test',
                          device='npu:0', stride=[1], offset=0)
                 for i, key in enumerate(('permuted_tokens', 'sorted_indices', 'probs'))}
    arguments.update(padded_mode=False, restore_shape=None)
    consumed = copy.deepcopy(arguments)
    if injecting:
        for key in ('sorted_indices', 'probs'):
            consumed[key]['object_id'] += 100
            consumed[key]['storage_id'] += 100
    return dict(original_calls=1, original_returned=True, arguments=arguments,
                consumed_arguments=consumed)


class MoE8RecordedNativeTests(unittest.TestCase):
    def test_frozen_probe_environment_digest_keeps_original_separators(self):
        self.assertEqual(probe_digest(dict(torch='2.9.0+cpu', torch_npu='2.9.0.post1+gitee7ba04',
                                          device='Ascend910B4')),
                         '71655f68200a317bfac09e54bbbc1d8e8d42e5ab285194568884993c93c7408b')

    def test_fault_and_off_consumed_arguments(self):
        check_native(native(True), True)
        check_native(native(False), False)

    def test_expert_hidden_cannot_change(self):
        value = native(True)
        value['consumed_arguments']['permuted_tokens']['object_id'] += 1
        with self.assertRaisesRegex(ValueError, 'EXPERT_HIDDEN_REPLACED'):
            check_native(value, True)

    def test_paired_probability_remap_required(self):
        value = native(True)
        value['consumed_arguments']['probs'] = copy.deepcopy(value['arguments']['probs'])
        with self.assertRaisesRegex(ValueError, 'CONSUMED_I_P_IDENTITY_NOT_MATCHING_PHASE'):
            check_native(value, True)

    def test_no_extra_original_execution(self):
        value = native(True); value['original_calls'] = 2
        with self.assertRaisesRegex(ValueError, 'ACTUAL_ORIGINAL_CALL_COUNT_OR_RETURN'):
            check_native(value, True)

    def test_baseline_equal_copy_is_not_original_argument(self):
        with self.assertRaisesRegex(ValueError, 'CONSUMED_I_P_IDENTITY_NOT_MATCHING_PHASE'):
            check_native(native(True), False)

    def test_shape_change_rejected(self):
        value = native(True); value['consumed_arguments']['sorted_indices']['shape'] = [3]
        with self.assertRaisesRegex(ValueError, 'REMAP_METADATA_CHANGED'):
            check_native(value, True)


if __name__ == '__main__':
    unittest.main()
