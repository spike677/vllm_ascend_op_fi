import copy
import unittest

from experiments.c2c3_closure_20261117.review_vision17_group_124 import check_phase


def fixture():
    output = dict(shape=[1, 2, 3], dtype='torch.bfloat16', finite=True, sha256='baseline')
    event = dict(path='patch', ordinal=0, operator='aten.conv3d.default', native_calls=1,
                 original_execution_verified=True, return_readback_exact=True, inputs_unchanged=True,
                 complete_input_weight_bias_identity=['input', 'weight', 'bias'],
                 complete_output_identity={'sha256': 'native-baseline'}, baseline_padding=[0, 0, 0],
                 padding=[0, 0, 0], applied=False, output_changed=False,
                 consumers=[{'returned': True}])
    native = dict(kind='operator', operator='aten.conv3d.default', path='patch',
                  original_returned=True, weight_owner_paths=['patch'],
                  conv_geometry=dict(stride=[2, 16, 16], padding=[0, 0, 0], dilation=[1, 1, 1], groups=1),
                  input=[{'device': 'npu:0'}], output=[{'device': 'npu:0'}])
    census = dict(complete=True, dropped_calls=0, observer_hooks=0, inflight=0, tensor_mutations=0,
                  forwards=[{'status': 'returned'}], module_inventory=[dict(path='patch', role='conv3d')],
                  events=[native], operator_counts={'aten.conv3d.default': 1})
    baseline = dict(phase='baseline', independent=census, input=['identical'], output=output,
                    calls=[{}], cleanup=dict(cleanup_confirmed=True, hook_owned=False, leases=0,
                    inflight=0, ownership_verified=True, enabled=False),
                    mechanism=dict(events=[event], errors=[], simulation=False))
    fault = copy.deepcopy(baseline)
    fault['phase'] = 'once:0'
    fault['output']['sha256'] = 'fault'
    fault['mechanism']['events'][0].update(padding=[0, 7, 7], applied=True, output_changed=True,
                                        complete_output_identity={'sha256': 'native-fault'})
    return baseline, fault, event


class Vision17RecordedAuditTests(unittest.TestCase):
    def test_outer_observer_is_not_final_native_argument_proof(self):
        baseline, fault, event = fixture()
        result = check_phase(fault, 4, baseline, event, {'patch'})
        self.assertEqual(result['independent_pre_intervention_padding'], [0, 0, 0])
        self.assertEqual(result['provider_applied_padding'], [0, 7, 7])
        self.assertFalse(result['independent_consumer_lineage_complete'])

    def test_missing_per_target_effect_rejected(self):
        baseline, fault, event = fixture()
        fault['mechanism']['events'][0]['output_changed'] = False
        with self.assertRaisesRegex(ValueError, 'PER_TARGET_OUTPUT_EFFECT_UNPROVEN'):
            check_phase(fault, 4, baseline, event, {'patch'})

    def test_input_change_rejected(self):
        baseline, fault, event = fixture()
        fault['mechanism']['events'][0]['complete_input_weight_bias_identity'][0] = 'changed'
        with self.assertRaisesRegex(ValueError, 'ACTUAL_INPUT_WEIGHT_OR_BIAS_CHANGED'):
            check_phase(fault, 4, baseline, event, {'patch'})

    def test_duplicate_structural_denominator_rejected(self):
        baseline, fault, event = fixture()
        fault['independent']['module_inventory'] *= 2
        with self.assertRaisesRegex(ValueError, 'STRUCTURAL_DENOMINATOR_CHANGED'):
            check_phase(fault, 4, baseline, event, {'patch'})

    def test_dropped_observations_rejected(self):
        baseline, fault, event = fixture()
        fault['independent']['dropped_calls'] = 1
        with self.assertRaisesRegex(ValueError, 'INDEPENDENT_OBSERVER_INCOMPLETE'):
            check_phase(fault, 4, baseline, event, {'patch'})

    def test_pending_hook_rejected(self):
        baseline, fault, event = fixture()
        fault['cleanup']['hook_owned'] = True
        with self.assertRaisesRegex(ValueError, 'PHASE_CLEANUP_PENDING'):
            check_phase(fault, 4, baseline, event, {'patch'})

    def test_off_restores_complete_output(self):
        baseline, fault, event = fixture()
        off = copy.deepcopy(baseline)
        off.update(phase='off', mechanism=None)
        self.assertFalse(check_phase(off, 5, baseline, event, {'patch'})['fault_applied'])
        off['output']['sha256'] = 'fault'
        with self.assertRaisesRegex(ValueError, 'MODEL_OUTPUT_EFFECT_OR_RESTORE_MISMATCH'):
            check_phase(off, 5, baseline, event, {'patch'})


if __name__ == '__main__':
    unittest.main()
