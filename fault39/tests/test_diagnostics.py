"""CPU-only diagnostic adversarial fixtures, never NPU acceptance."""
import copy
import unittest

from demoncat_fault39.diagnostics import compare_metric, diagnose_records, resolve_stacks


def tensor(value=1., *, shape=None):
    return dict(type='torch.Tensor', dtype='torch.float32', shape=shape or [2],
                Max=value, Min=-value, Mean=None, Norm=None)


def record(value=1., scalar=None):
    out = dict(input_args=[tensor()], input_kwargs={}, output=[tensor(value)])
    if scalar is not None:
        out['input_kwargs']['padding'] = dict(type='int', value=scalar)
    return out


def context(data, sources=None):
    sources = sources or {}
    stack = {str(i): [[name], [sources.get(name, 'source:'+name)]] for i, name in enumerate(data)}
    return stack, {name: None for name in data}


def run(b, f, bsource=None, fsource=None, *, aligned=True):
    bs, bc = context(b, bsource)
    fs, fc = context(f, fsource)
    return diagnose_records(b, f, bs, fs, bc, fc, capture_metadata_aligned=aligned)


class DiagnosticTests(unittest.TestCase):
    def test_self_does_not_turn_null_into_equality(self):
        data = {'Torch.add.0.forward': record()}
        result = run(data, copy.deepcopy(data))
        self.assertIsNone(result['global_first_difference'])
        self.assertIsNone(result['first_local_tensor_change'])
        self.assertEqual(len(result['unobservable_fields']), 4)
        self.assertEqual(result['counts'], dict(baseline=1, fault=1, matched=1))

    def test_insertion_keeps_every_record_and_observation_window(self):
        a, m, c = 'Module.norm.RMSNorm.forward.0', 'Tensor.__mul__.0.forward', 'Tensor.__add__.0.forward'
        b = {a: record(), c: record(2.)}
        f = {a: record(), m: record(.5, scalar=.5), c: record(1.)}
        report = run(b, f)
        self.assertEqual([len(report['ledger'][x]) for x in ('baseline','fault')], [2,3])
        window, = report['unpaired_windows']
        self.assertEqual((window['previous_shared_anchor'], window['next_shared_anchor']), (a,c))
        self.assertEqual(window['producer_consumer_identity'], 'UNPROVEN')
        self.assertEqual(report['first_local_tensor_change']['baseline_operator'], c)
        self.assertIsNone(report['global_first_difference'])

    def test_unique_source_can_pair_renumbered_calls_without_dropping_insert(self):
        b = {'Torch.add.0.forward': record()}
        f = {'Torch.add.0.forward': record(3.), 'Torch.add.1.forward': record(2.)}
        report = run(b, f, {'Torch.add.0.forward':'actual line'},
                     {'Torch.add.0.forward':'new line', 'Torch.add.1.forward':'actual line'})
        pair, = report['pairs']
        self.assertFalse(pair['same_operator_id'])
        self.assertEqual(pair['fault_operator'], 'Torch.add.1.forward')
        self.assertEqual(len(report['unpaired_windows']), 1)

    def test_identical_repeated_context_is_ambiguous_even_when_ids_equal(self):
        b = {'Torch.add.0.forward':record(), 'Torch.add.1.forward':record()}
        source = dict.fromkeys(b, 'same loop line')
        report = run(b,b,source,source)
        self.assertEqual(report['counts']['matched'],0)
        self.assertTrue(all(r['reason']=='AMBIGUOUS_REPEATED_CONTEXT' for r in report['ledger']['baseline']))

    def test_changed_argument_retains_invocation_but_not_same_argument_attribution(self):
        key='Functional.conv3d.0.forward'
        report=run({key:record(1.,0)},{key:record(.5,7)})
        self.assertEqual(report['counts']['matched'],1)
        self.assertFalse(report['pairs'][0]['argument_descriptions_unchanged'])
        self.assertEqual(report['pairs'][0]['numerical_attribution'],'CONDITION_CHANGED_NOT_SAME_ARGUMENTS')
        self.assertTrue(report['changed_tensor_fields'])
        self.assertFalse(report['changed_tensor_fields'][0]['same_argument_descriptions'])

    def test_boolean_output_is_observation_not_input_identity(self):
        key='Torch.equal.0.forward';b={key:record()};f=copy.deepcopy(b)
        b[key]['output']=[dict(type='bool',value=True)]
        f[key]['output']=[dict(type='bool',value=False)]
        report=run(b,f)
        self.assertTrue(report['pairs'][0]['argument_descriptions_unchanged'])
        self.assertEqual(len(report['changed_scalar_outputs']),1)
        self.assertIsNone(report['global_first_difference'])

    def test_anchor_reorder_refuses_all_pairs(self):
        b={'Torch.add.0.forward':record(),'Torch.sub.0.forward':record()}
        f=dict(reversed(list(b.items())))
        report=run(b,f)
        self.assertFalse(report['anchor_order_preserved'])
        self.assertFalse(report['pairs'])
        self.assertEqual(len(report['unpaired_windows']),4)

    def test_missing_context_is_not_null_context(self):
        b={'Torch.add.0.forward':record()};bs,bc=context(b)
        for stack,construct in [({},bc),({'x':[[next(iter(b))],[]]},bc),(bs,{})]:
            report=diagnose_records(b,b,bs,stack,bc,construct,capture_metadata_aligned=True)
            self.assertFalse(report['pairs'])

    def test_configuration_mismatch_prevents_local_numeric_acceptance(self):
        b={'Torch.add.0.forward':record()};f={'Torch.add.0.forward':record(3.)}
        report=run(b,f,aligned=False)
        self.assertFalse(report['pairs'])
        self.assertFalse(report['changed_tensor_fields'])
        self.assertEqual(report['ledger']['fault'][0]['reason'],'CAPTURE_METADATA_UNALIGNED')

    def test_shape_change_is_not_statistics(self):
        b={'Torch.add.0.forward':record()};f=copy.deepcopy(b)
        f['Torch.add.0.forward']['output'][0]=tensor(9.,shape=[3])
        report=run(b,f)
        self.assertEqual(report['pairs'][0]['tensor_description_mismatches'],['/output/0'])
        self.assertFalse(report['changed_tensor_fields'])

    def test_fixed_threshold_preserves_small_exact_change(self):
        row=compare_metric(1.,1.+1e-6)
        self.assertTrue(row['exact_change'])
        self.assertFalse(row['exceeds_tolerance'])
        self.assertEqual(row['threshold'],1e-8+1e-5)

    def test_nonfinite_and_zero_do_not_get_epsilon_or_equality(self):
        self.assertEqual(compare_metric(None,None)['status'],'UNOBSERVABLE')
        self.assertEqual(compare_metric(float('nan'),float('nan'))['status'],'UNCOMPARABLE')
        self.assertIsNone(compare_metric(float('inf'),float('inf'))['exceeds_tolerance'])
        self.assertIsNone(compare_metric(0.,1.)['relative_change'])
        self.assertTrue(compare_metric(0.,1.)['exceeds_tolerance'])

    def test_duplicate_stack_assignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):
            resolve_stacks({'a':[['x'],['line']], 'b':[['x'],['line']]})

    def test_changing_stack_line_refuses_even_matching_names(self):
        b={'Torch.add.0.forward':record()}
        report=run(b,b,{next(iter(b)):'line 1'},{next(iter(b)):'line 2'})
        self.assertFalse(report['pairs'])

    def test_no_native_log_or_tensor_input_interface(self):
        with self.assertRaises(TypeError):
            diagnose_records({}, {}, {}, {}, {}, {}, capture_metadata_aligned=True,native_log='hit')


if __name__=='__main__':
    unittest.main(verbosity=2)
