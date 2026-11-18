"""Actual CPU OpOverload contracts; patched fixture hashes never qualify NPU."""
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import gemm_lineage
from fault_injection.deployments.postwrite import PostwriteProvider, PostwriteRefused
from fault_injection.deployments.postwrite_boundaries import projection_return
from fault_injection.deployments import postwrite_npu_consumers as consumers
from output39_custom_fixture import CustomOperators, CustomProjection
from test_postwrite_providers import CpuTransport


class CustomGemmBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.ops = self.enterContext(CustomOperators())
        self.model = CustomProjection()
        self.x = torch.arange(1, 9, dtype=torch.bfloat16).reshape(2, 4)
        self.func = torch.ops.vllm.unquantized_gemm.default
        self.binding = gemm_lineage.ReviewedGemmBinding()

    def proof(self, args=None, result=None):
        args = (self.x, self.model.weight) if args is None else args
        result = self.func(*args) if result is None else result
        return projection_return(self.func, args, {}, result,
            gemm_binding=self.binding, expected_weight=self.model.weight)

    def test_actual_custom_identity_source_owned_weight_and_original_return(self):
        proof, reason = self.proof()
        self.assertIsNone(reason)
        self.assertEqual(self.ops.calls, ['gemm'])
        self.assertEqual(proof['operator'], 'vllm.unquantized_gemm.default')
        self.assertEqual(proof['boundary'], 'reviewed_vllm_custom_gemm_return')
        self.assertEqual(proof['returned_tensor']['data_ptr'], self.ops.last_gemm.data_ptr())
        self.assertFalse(proof['native_matmul_claim'])
        source = proof['registered_implementation']
        self.assertTrue(source['source_and_loaded_code_verified'])
        self.assertEqual(len(source['sources']), 3)
        self.assertEqual(source['module_weight_object_identity'], id(self.model.weight))

    def test_foreign_weight_bias_dtype_geometry_and_alias_fail_closed(self):
        result = torch.ones_like(self.x)
        cases = [
            ((self.x, self.model.weight.clone()), result, 'OWNED_WEIGHT'),
            ((self.x, self.model.weight, torch.ones(4, dtype=torch.bfloat16)), result, 'NO_BIAS'),
            ((self.x.float(), self.model.weight), result, 'BF16_OPERANDS'),
            ((self.x[:, :3], self.model.weight), result, 'GEOMETRY_MISMATCH'),
            ((self.x, self.model.weight), result[:, :2], 'RETURN_GEOMETRY'),
            ((self.x, self.model.weight), self.x, 'ALIASES_OPERAND'),
            ((self.x, self.model.weight), result.float(), 'RETURN_OR_DEVICE'),
        ]
        for args, output, reason in cases:
            with self.subTest(reason=reason):
                proof, rejected = self.proof(args, output)
                self.assertIsNone(proof)
                self.assertIn(reason, rejected)
        self.assertFalse(self.ops.calls)

    def test_custom_schema_duplicate_arguments_and_source_binding_required(self):
        result = torch.ones_like(self.x)
        self.assertEqual(projection_return(self.func, (self.x, self.model.weight), {}, result)[1],
                         'CUSTOM_GEMM_SOURCE_BINDING_REQUIRED')
        proof, reason = projection_return(self.func, (self.x, self.model.weight), {'x': self.x}, result,
            gemm_binding=self.binding, expected_weight=self.model.weight)
        self.assertIsNone(proof)
        self.assertIn('DUPLICATE_PUBLIC_GEMM_ARGUMENT', reason)
        with patch.object(gemm_lineage, 'SCHEMA', gemm_lineage.SCHEMA + ' changed'):
            self.assertIn('UNREVIEWED_PUBLIC_GEMM_SCHEMA', self.proof(result=result)[1])

    def test_actual_new_library_override_is_refused_then_removal_restores_original(self):
        result = self.func(self.x, self.model.weight)
        first = self.proof(result=result)[0]['registered_implementation']['current_registration']
        library = torch.library.Library('vllm', 'IMPL')
        try:
            library.impl('unquantized_gemm', lambda x, weight, bias=None: x,
                         'PrivateUse1', allow_override=True)
            self.assertIn('vllm/unquantized_gemm/PrivateUse1', self.ops.library._op_impls)
            proof, reason = self.proof(result=result)
            self.assertIsNone(proof)
            self.assertIn('CURRENT_REGISTRATION_SHADOWED', reason)
        finally:
            library._destroy()
        self.assertEqual(self.proof(result=result)[0]['registered_implementation']['current_registration'], first)

    def test_actual_same_library_override_is_refused_despite_same_origin(self):
        result = self.func(self.x, self.model.weight)
        self.ops.library.impl('unquantized_gemm', lambda x, weight, bias=None: x,
                              'PrivateUse1', allow_override=True)
        proof, reason = self.proof(result=result)
        self.assertIsNone(proof)
        self.assertIn('CURRENT_REGISTRATION_SHADOWED', reason)

    def test_current_origin_definition_and_build_are_independently_bound(self):
        result = self.func(self.x, self.model.weight)
        original = torch._C._dispatch_dump('vllm::unquantized_gemm')
        for replace, reason in (
                (lambda dump: '\n'.join(line for line in dump.splitlines() if not line.startswith('PrivateUse1:')),
                 'DISPATCH_REGISTRATION_UNPROVEN'),
                (lambda dump: '\n'.join('PrivateUse1: registered at foreign.py:1 :: (none) [ boxed ]'
                    if line.startswith('PrivateUse1:') else line for line in dump.splitlines()),
                 'IMPL_NOT_DEFINITION_OWNER'),
                (lambda dump: '\n'.join('debug: registered at foreign.py:1' if line.startswith('debug:') else
                    'PrivateUse1: registered at foreign.py:1 :: (none) [ boxed ]'
                    if line.startswith('PrivateUse1:') else line for line in dump.splitlines()),
                 'REGISTRATION_SOURCE_ORIGIN_CHANGED')):
            with self.subTest(reason=reason), patch.object(torch._C, '_dispatch_dump', return_value=replace(original)):
                self.assertIn(reason, self.proof(result=result)[1])
        with patch.object(gemm_lineage, 'DISPATCH_TORCH_VERSION', 'unreviewed'):
            self.assertIn('DISPATCH_BUILD', self.proof(result=result)[1])

    def test_privateuse1_override_after_lossless_opens_no_fault_lease(self):
        p = PostwriteProvider(self.model, CpuTransport(), '39.output_proxy')
        p.qualify(self.x)
        p.run('baseline', self.x)
        library = torch.library.Library('vllm', 'IMPL')
        try:
            library.impl('unquantized_gemm', lambda x, weight, bias=None: x,
                         'PrivateUse1', allow_override=True)
            with patch.object(p.transport, 'open', side_effect=AssertionError('Native must not open')):
                with self.assertRaisesRegex(PostwriteRefused, 'QUALIFIED_WRITER_OR_CONSUMER_NOT_OBSERVED'):
                    p.run('fault', self.x)
            self.assertIsNone(p._last_intervention)
            self.assertFalse(p.live_lease_count)
        finally:
            library._destroy()

    def test_source_code_and_registration_owner_changes_remain_in_inventory(self):
        for change in ('source', 'code', 'library'):
            with self.subTest(change=change), CustomOperatorsReplacement(self.ops, change):
                p = PostwriteProvider(self.model, CpuTransport(), '39.output_proxy')
                with self.assertRaisesRegex(PostwriteRefused, 'NO_ELIGIBLE_WRITER'):
                    p.qualify(self.x)
                for observer in p._observations:
                    rows = observer.inventory.report()['calls']
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['original_calls'], 1)
                    self.assertEqual(rows[0]['operator'], str(self.func))
                    self.assertFalse(rows[0]['eligible'])
                    self.assertIn('CUSTOM_GEMM_BINDING_REFUSED', rows[0]['rejection'])

    def test_both_consumers_original_once_lossless_fault_off_and_frozen_report(self):
        for kind in ('rms', 'swiglu'):
            with self.subTest(kind=kind):
                model = CustomProjection(kind)
                p = PostwriteProvider(model, CpuTransport(), '39.output_proxy')
                start = len(self.ops.calls)
                selected = p.qualify(self.x)
                self.assertEqual(self.ops.calls[start:], ['gemm', kind] * 2)
                self.assertEqual(selected['consumers'], ['npu.npu_' + ('rms_norm' if kind == 'rms' else kind) + '.default'])
                observation = selected['consumer_observations'][0]
                self.assertEqual(observation['selected_elements_read'], [0])
                self.assertTrue(observation['reviewed_consumer']['schema_has_no_alias_or_write'])
                self.assertEqual(observation['operand_views'][0]['storage_identity'],
                                 selected['producer_proof']['returned_tensor']['storage_identity'])
                snapshot = p._observations[0].inventory.report()
                snapshot['calls'][0]['producer_proof']['registered_implementation']['sources'].clear()
                self.assertEqual(len(p._observations[0].inventory.report()['calls'][0]
                                     ['producer_proof']['registered_implementation']['sources']), 3)
                baseline, _ = p.run('baseline', self.x)
                _, fault = p.run('fault', self.x)
                off, _ = p.run('off', self.x)
                self.assertEqual(self.ops.calls[start:], ['gemm', kind] * 5)
                self.assertEqual(fault['new_fault_hits'], 1)
                self.assertTrue(fault['recovery_confirmed'])
                self.assertEqual(p.live_lease_count, 0)
                self.assertTrue(torch.equal(baseline, off))

    def test_refused_original_is_in_denominator_and_does_not_renumber_selected_call(self):
        class Mixed(CustomProjection):
            def forward(self, x):
                torch.ops.vllm.unquantized_gemm.default(x, self.weight.clone())
                return super().forward(x)
        p = PostwriteProvider(Mixed(), CpuTransport(), '39.output_proxy')
        p.qualify(self.x)
        for observed in p._observations:
            rows = observed.inventory.report()['calls']
            self.assertEqual(len(rows), 2)
            self.assertEqual([row['original_calls'] for row in rows], [1, 1])
            self.assertIn('OWNED_WEIGHT', rows[0]['rejection'])
            self.assertFalse(rows[0]['eligible'])
            self.assertTrue(rows[1]['eligible'])
            self.assertEqual(rows[1]['key'][-1], p._target[-1])
        self.assertEqual(p._target[-1], 1)
        self.assertEqual(self.ops.calls, ['gemm', 'gemm', 'rms'] * 2)

    def test_disjoint_view_of_same_storage_and_unknown_mutating_are_not_consumers(self):
        for model in (CustomProjection(rows_slice=True), CustomProjection(consumer='unknown')):
            with self.subTest(model=model):
                p = PostwriteProvider(model, CpuTransport(), '39.output_proxy')
                with self.assertRaisesRegex(PostwriteRefused, 'NO_ELIGIBLE_WRITER'):
                    p.qualify(self.x)
                self.assertEqual(len(p._observations[0].inventory.calls), 1)
                self.assertFalse(p._observations[0].inventory.report()['calls'][0]['eligible'])
                self.assertFalse(p.live_lease_count)

    def test_mutating_custom_write_is_still_refused_before_original_on_live_target(self):
        p = PostwriteProvider(CustomProjection(mutate=True), CpuTransport(), '39.output_proxy')
        p.qualify(self.x)
        before = self.ops.calls.count('unknown_mutating')
        with self.assertRaisesRegex(PostwriteRefused, 'WRITE_AFTER_SELECTED_WRITER_REFUSED'):
            p.run('baseline', self.x)
        self.assertEqual(self.ops.calls.count('unknown_mutating'), before)
        self.assertFalse(p.live_lease_count)
        self.assertFalse(p.query()['lossless'])

    def test_source_change_after_lossless_does_not_admit_fault(self):
        p = PostwriteProvider(self.model, CpuTransport(), '39.output_proxy')
        p.qualify(self.x)
        p.run('baseline', self.x)
        with CustomOperatorsReplacement(self.ops, 'source'):
            with self.assertRaisesRegex(PostwriteRefused, 'QUALIFIED_WRITER_OR_CONSUMER_NOT_OBSERVED'):
                p.run('fault', self.x)
        self.assertIsNone(p._last_intervention)
        self.assertFalse(p.live_lease_count)

    def test_consumer_build_change_after_lossless_refuses_before_native_open(self):
        p = PostwriteProvider(self.model, CpuTransport(), '39.output_proxy')
        public_selection = p.qualify(self.x)
        public_selection['consumer_observations'][0]['reviewed_consumer'].clear()
        public_selection['consumer_observations'].clear()
        public_selection['producer_proof']['registered_implementation']['sources'].clear()
        p.run('baseline', self.x)
        before = self.ops.calls.count('rms')
        with patch.object(self.ops.npu, '__version__', 'unreviewed'), \
                patch.object(p.transport, 'open', side_effect=AssertionError('must not open Native')):
            with self.assertRaisesRegex(PostwriteRefused, 'CONSUMER_CHANGED_BEFORE_ADMISSION'):
                p.run('fault', self.x)
        self.assertEqual(self.ops.calls.count('rms'), before)
        self.assertIsNone(p._last_intervention)
        self.assertFalse(p.live_lease_count)


class CustomOperatorsReplacement:
    def __init__(self, fixture, change):
        self.fixture, self.change = fixture, change

    def __enter__(self):
        module = self.fixture.modules[gemm_lineage.PATCH_MODULE]
        if self.change == 'source':
            self.path = Path(module.__file__)
            self.old = self.path.read_bytes()
            self.path.write_bytes(self.old + b'# unsupported source\n')
        else:
            target, field, value = ((module, 'unquantized_gemm', lambda *args: None) if self.change == 'code'
                else (self.fixture.modules[gemm_lineage.REGISTER_MODULE], 'vllm_lib', object()))
            self.patch = patch.object(target, field, value)
            self.patch.__enter__()

    def __exit__(self, *error):
        if self.change == 'source':
            self.path.write_bytes(self.old)
        else:
            self.patch.__exit__(*error)


class NpuConsumerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.ops = self.enterContext(CustomOperators())
        self.x = torch.ones(2, 4, dtype=torch.bfloat16)
        self.gamma = torch.ones(4, dtype=torch.bfloat16)
        self.rms = torch.ops.npu.npu_rms_norm.default
        self.swi = torch.ops.npu.npu_swiglu.default

    def test_actual_read_operand_identity_storage_offset_and_returned_once(self):
        base = torch.ones(3, 4, dtype=torch.bfloat16)
        view = base[1:]
        output = self.rms(view, self.gamma)
        proof, reads, reason = consumers.npu_value_consumer(self.rms, (view, self.gamma), {}, output)
        self.assertIsNone(reason)
        self.assertIs(reads[0], view)
        self.assertEqual(proof['input']['storage_offset'], 4)
        self.assertEqual(proof['input']['storage_identity'], base.untyped_storage()._cdata)
        self.assertEqual(self.ops.calls, ['rms'])

    def test_unreviewed_version_dispatch_meta_and_alias_schema_are_refused(self):
        result = self.rms(self.x, self.gamma)
        for location, attribute, value, expected in (
                (self.ops.npu, '__version__', 'unreviewed', 'BUILD'),
                (consumers, 'TORCH_VERSION', 'unreviewed', 'BUILD'),
                (consumers, 'META_SHA256', '0'*64, 'META_SOURCE'),
                (torch._C, '_dispatch_dump', lambda name: 'schema: wrong', 'DISPATCH'),
                (consumers, 'RMS_SCHEMA', consumers.RMS_SCHEMA.replace('Tensor input', 'Tensor(a!) input'), 'SCHEMA')):
            with self.subTest(expected=expected), patch.object(location, attribute, value):
                proof, _, reason = consumers.npu_value_consumer(self.rms, (self.x, self.gamma), {}, result)
                self.assertIsNone(proof)
                self.assertIn(expected, reason)

    def test_rms_input_gamma_output_dtype_geometry_and_alias_refusals(self):
        out, rstd = self.rms(self.x, self.gamma)
        cases = [((self.x.float(), self.gamma), (out, rstd), 'INPUT_VIEW'),
                 ((self.x.t(), self.gamma), (out, rstd), 'INPUT_VIEW'),
                 ((self.x, self.gamma.float()), (out, rstd), 'GAMMA'),
                 ((self.x, self.gamma, float('nan')), (out, rstd), 'EPSILON'),
                 ((self.x, self.gamma), (out, rstd.to(torch.bfloat16)), 'RETURN_GEOMETRY'),
                 ((self.x, self.gamma), (out, rstd.view(2)), 'RETURN_GEOMETRY'),
                 ((self.x, self.gamma), (self.x, rstd), 'ALIAS'),
                 ((self.x, self.x[0]), (out, rstd), 'ALIAS')]
        for args, result, expected in cases:
            with self.subTest(expected=expected):
                proof, _, reason = consumers.npu_value_consumer(self.rms, args, {}, result)
                self.assertIsNone(proof)
                self.assertIn(expected, reason)

    def test_swiglu_dim_output_alias_and_format_refusals(self):
        result = self.swi(self.x)
        cases = [((self.x, 0), result, 'LAST_EVEN_DIM'),
                 ((self.x[:, :3].contiguous(),), result, 'LAST_EVEN_DIM'),
                 ((self.x,), result.float(), 'RETURN_GEOMETRY'),
                 ((self.x,), self.x[0].view(2, 2), 'ALIAS')]
        for args, returned, expected in cases:
            with self.subTest(expected=expected):
                proof, _, reason = consumers.npu_value_consumer(self.swi, args, {}, returned)
                self.assertIsNone(proof)
                self.assertIn(expected, reason)
        with patch.object(self.ops.npu, 'get_npu_format', return_value=29):
            self.assertIn('ND_VIEWS_REQUIRED', consumers.npu_value_consumer(self.swi, (self.x,), {}, result)[2])


if __name__ == '__main__':
    unittest.main()
