"""Real Torch frames/Host/Runtime; the rotary source profile is explicitly CPU.

These fixtures reproduce direct eager __call__, inherited forward and a shared
rotary helper. They grant no production source, Native, C2/C3 or E5 qualification.
"""
import hashlib
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import serving39_output_discovery as discovery
import test_core_output39 as c1
import test_core_output39_group as group


class AscendApplyRotaryEmb(torch.nn.Module):
    def forward(self, query, key):
        return query, key


class AscendRotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.apply_rotary_emb = AscendApplyRotaryEmb()
        self._forward_method = self.forward_oot
        self.use_helper = False
        self.skip_operator = False

    def forward(self, query, key):
        return self._forward_method(query, key)

    def forward_oot(self, query, key):
        if self.use_helper:
            query, key = self.apply_rotary_emb(query, key)
        if self.skip_operator:
            return query, key
        return torch.ops.vllm.npu_rotary_embedding(query, key)

    def different_branch(self, query, key):
        return torch.ops.vllm.npu_rotary_embedding(query, key)


class ProjectionLayer(torch.nn.Module):
    def __init__(self, rotary):
        super().__init__()
        self.rotary_emb = rotary
        self.projection = c1.Projection()

    def forward(self, hidden):
        query, _ = self.rotary_emb(hidden, hidden)
        return self.projection(query)


class BackboneForward(torch.nn.Module):
    def forward(self, hidden):
        if self.fail:
            raise RuntimeError('CPU_DIRECT_BODY_FAILED')
        if self.return_none:
            return None
        if self.other_globals:
            function = types.FunctionType(type(self).forward.__code__, dict(globals()))
            self.other_globals = False
            return function(self, hidden)
        if getattr(self, 'foreign', None) is not None:
            self.foreign(hidden)
        outputs = [layer(hidden) for layer in self.layers]
        return outputs[0] + outputs[1]


class DirectBackbone(BackboneForward):
    def __init__(self):
        super().__init__()
        rotary = AscendRotaryEmbedding()
        self.layers = torch.nn.ModuleList([ProjectionLayer(rotary), ProjectionLayer(rotary)])
        self.fail = self.return_none = self.other_globals = False

    def __call__(self, *args, **kwargs):
        # The same eager boundary used by support_torch_compile: no Module
        # hooks execute for this object; forward is inherited unchanged.
        return self.forward(*args, **kwargs)


class DirectModel(c1.life.CpuModel):
    def __init__(self):
        super().__init__()
        self.model = DirectBackbone()
        self.forward_calls = 0
        with torch.no_grad():
            self.head.weight.fill_(.25)

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.forward_calls += 1
        hidden = super().forward(input_ids, positions, inputs_embeds, intermediate_tensors)
        return self.model(hidden)


class CoreOutput39DirectDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = torch.library.Library('vllm', 'FRAGMENT')
        cls.library.define('npu_rotary_embedding(Tensor query, Tensor key) -> (Tensor, Tensor)')
        cls.library.impl('npu_rotary_embedding', lambda query, key: (query.clone(), key.clone()), 'CPU')

    @classmethod
    def tearDownClass(cls):
        cls.library._destroy()

    def setUp(self):
        self.assertIsNone(sys.getprofile())
        self.enterContext(patch.object(group, 'ParallelProjections', DirectModel))
        # A declared CPU profile for the real test source, never accepted by
        # unpatched production code or claimed as the actual Ascend module.
        self.enterContext(patch.object(discovery, '_ROTARY_MODULE', __name__))
        self.enterContext(patch.object(discovery, '_ROTARY_SOURCE_SHA256', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        self.case = group.CoreOutput39GroupTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def start(self):
        self.case.start()
        self.provider = self.case.provider
        self.session = self.provider.session
        return self.case

    def healthy(self):
        self.start()
        self.case.request(0)
        self.case.request(1)
        return self.provider.p._observations[0].inventory.report()

    def test_complete_actual_direct_parent_container_and_bounded_shared_helper(self):
        c = self.start()
        for index in range(40):
            if self.provider.complete:
                break
            c.request(index)
        self.assertTrue(self.provider.complete)
        original = self.provider.group.discovery_reports[0]
        modules = {row['object_identity']: row['calls'] for row in original['modules']}
        backbone = c.model.model
        self.assertEqual(modules[id(backbone)], 1)
        self.assertEqual(modules[id(backbone.layers[0].rotary_emb)], 2)
        self.assertEqual(modules[id(c.model)], 1)  # manual/hook/frame dedup
        rows = {row['path']: row for row in self.provider.group.structural_inventory}
        self.assertTrue(rows['model.layers']['structural_noncall'])
        proof, = rows['model.layers.0.rotary_emb.apply_rotary_emb']['actual_bounded_noncall']
        self.assertEqual((proof['actual_parent_calls'], proof['actual_delegate_returns'], proof['actual_public_rotary_returns']), (2, 2, 2))
        self.assertEqual(len(proof['child_aliases']), 2)
        self.assertFalse(proof['qualification_granted'])
        frames = [row for row in original['actual_forward_frames'] if row['module_object_identity'] == id(backbone)]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]['entry_kind'], 'direct_frame')
        self.assertTrue(frames[0]['returned'] and frames[0]['nonnull_return'])
        report = self.provider.group.report()
        self.assertEqual(report['eligible_target_count'], 2)
        self.assertEqual(self.provider.group.discovery_diagnostics['structural_modules'], len(tuple(c.model.modules())))
        self.assertFalse(original['frame_failures'])
        self.assertEqual(original['frame_observer_scope']['owner'], self.provider.native_owner)
        self.assertFalse(original['frame_observer_scope']['serializable_authority'])
        self.assertIn('serving39_output_discovery.py', report['source_pins'])
        state = c.ok(c.client.finish())
        self.assertFalse(state['evidence']['c2'] or state['evidence']['c3'])
        c.ok(c.client.unregister())
        self.assertEqual(c.client.detach()['status'], 'DETACHED')
        self.assertIsNone(sys.getprofile())

    def test_source_pin_wrong_branch_and_unused_executable_remain_unobserved(self):
        self.case.model.unused = torch.nn.Linear(4, 4, dtype=torch.bfloat16).eval()
        rotary = self.case.model.model.layers[0].rotary_emb
        rotary._forward_method = rotary.different_branch
        c = self.start()
        for index in range(6):
            c.enqueue(index)
            result = c.client.collect_replay()
        self.assertEqual(result['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(self.provider.group_failure, 'POSTWRITE_INDEPENDENT_DISCOVERY_INCOMPLETE')
        unknown = [row for row in self.provider.group.structural_inventory if not row['structural_noncall']]
        self.assertEqual({row['object_identity'] for row in unknown},
                         {id(rotary.apply_rotary_emb), id(c.model.unused)})
        self.assertFalse(self.provider.group.noncall_discovery[0])
        c.ok(c.client.unregister())

    def test_unreviewed_source_never_excludes_helper(self):
        with patch.object(discovery, '_ROTARY_SOURCE_SHA256', '0'*64):
            original = self.healthy()
        self.assertFalse(original['bounded_noncall_proofs'])
        self.assertEqual(original['operator_counts']['vllm.npu_rotary_embedding.default'], 2)
        self.case.ok(self.case.client.unregister())

    def test_unexecuted_parent_cannot_exclude_its_helper(self):
        dormant = AscendRotaryEmbedding()
        self.case.model.dormant = dormant
        original = self.healthy()
        self.assertFalse(any(proof['child_identity'] == id(dormant.apply_rotary_emb)
                             for proof in original['bounded_noncall_proofs']))
        self.assertFalse(any(row['object_identity'] in (id(dormant), id(dormant.apply_rotary_emb))
                             for row in original['modules']))
        self.case.ok(self.case.client.unregister())

    def test_delegate_return_without_actual_operator_cannot_exclude_helper(self):
        self.case.model.model.layers[0].rotary_emb.skip_operator = True
        original = self.healthy()
        self.assertFalse(original['bounded_noncall_proofs'])
        self.assertNotIn('vllm.npu_rotary_embedding.default', original['operator_counts'])
        frames = [row for row in original['actual_forward_frames'] if row['attribute'] == '_forward_method']
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(row['returned'] and row['nonnull_return'] for row in frames))
        self.assertTrue(all(row['actual_rotary_returns'] == 0 for row in frames))
        self.case.ok(self.case.client.unregister())

    def test_real_helper_call_is_observed_and_never_reported_noncall(self):
        self.case.model.model.layers[0].rotary_emb.use_helper = True
        original = self.healthy()
        child = self.case.model.model.layers[0].rotary_emb.apply_rotary_emb
        self.assertEqual(next(row['calls'] for row in original['modules'] if row['object_identity'] == id(child)), 2)
        self.assertFalse(original['bounded_noncall_proofs'])
        self.case.ok(self.case.client.unregister())

    def test_foreign_profile_is_untouched_and_owner_cleanup_explicit(self):
        c = self.start()
        foreign = lambda *args: None
        sys.setprofile(foreign)
        try:
            with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_FOREIGN_PROFILE_PRESENT'):
                c.enqueue(0)
            self.assertIs(sys.getprofile(), foreign)
            self.assertFalse(c.transport.opened)
            self.assertTrue(c.client.snapshot()['ownership_retained'])
            c.ok(c.client.unregister())
            self.assertIs(sys.getprofile(), foreign)
        finally:
            sys.setprofile(None)

    def test_exception_unwind_never_counts_as_normal_return(self):
        self.case.model.model.fail = True
        c = self.start()
        with self.assertRaisesRegex(RuntimeError, 'CPU_DIRECT_BODY_FAILED'):
            c.enqueue(0)
        window = self.session.window
        self.assertTrue(window.profile_owned)
        self.assertIn('OUTPUT39_FRAME_NORMAL_RETURN_UNPROVEN', str(window.errors))
        row = next(event for event in window.events if event['module_object_identity'] == id(c.model.model))
        self.assertTrue(row['returned'])
        self.assertFalse(row['nonnull_return'])
        self.assertFalse(window.noncall_proofs)
        self.assertTrue(c.client.snapshot()['ownership_retained'])
        c.ok(c.client.unregister())
        self.assertIsNone(sys.getprofile())

    def test_none_return_is_ambiguous_and_cannot_qualify(self):
        self.case.model.model.return_none = True
        c = self.start()
        with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_FRAME_OBSERVATION_FAILED'):
            c.enqueue(0)
        self.assertFalse(c.transport.opened)
        c.ok(c.client.unregister())
        self.assertIsNone(sys.getprofile())

    def test_same_code_and_self_with_foreign_globals_rejected(self):
        self.case.model.model.other_globals = True
        c = self.start()
        with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_FRAME_OBSERVATION_FAILED'):
            c.enqueue(0)
        self.assertIn('OUTPUT39_FRAME_ACTUAL_OWNER_CHANGED', str(self.session.window.errors))
        self.assertFalse(c.transport.opened)
        c.ok(c.client.unregister())

    def test_same_code_with_foreign_module_cannot_impersonate_held_object(self):
        foreign = DirectBackbone().eval()
        object.__setattr__(self.case.model.model, 'foreign', foreign)
        c = self.start()
        with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_FRAME_OBSERVATION_FAILED'):
            c.enqueue(0)
        window = self.session.window
        self.assertIn('OUTPUT39_FRAME_FOREIGN_MODULE', str(window.errors))
        self.assertFalse(any(row['module_object_identity'] == id(foreign) for row in window.events))
        self.assertFalse(c.transport.opened)
        c.ok(c.client.unregister())

    def test_foreign_profile_replacement_blocks_cleanup_until_original_owner_restored(self):
        c = self.start()
        foreign = lambda *args: None
        def replace_profile(module, args):
            sys.setprofile(foreign)
        handle = c.model.model.layers[0].register_forward_pre_hook(replace_profile)
        try:
            with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_PROFILE_OWNER_CHANGED'):
                c.enqueue(0)
            window = self.session.window
            result = c.client.unregister()
            self.assertEqual(result['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(result['ownership_retained'])
            self.assertIs(sys.getprofile(), foreign)
            self.assertTrue(window.profile_owned)
        finally:
            handle.remove()
            sys.setprofile(self.session.window.profile_callback)
        c.ok(c.client.unregister())
        self.assertIsNone(sys.getprofile())

    def test_profile_remove_after_side_effect_error_keeps_first_error_and_retry(self):
        c = self.start()
        original = sys.setprofile
        def fail_after(callback):
            original(callback)
            if callback is None:
                raise RuntimeError('CPU_AFTER_PROFILE_REMOVED')
        with patch.object(sys, 'setprofile', side_effect=fail_after):
            with self.assertRaisesRegex(RuntimeError, 'CPU_AFTER_PROFILE_REMOVED'):
                c.enqueue(0)
        self.assertIsNone(sys.getprofile())
        self.assertFalse(self.session.window.profile_owned)
        self.assertTrue(c.client.snapshot()['ownership_retained'])
        self.assertIn('CPU_AFTER_PROFILE_REMOVED', str(self.provider.failures))
        c.ok(c.client.unregister())
        self.assertIsNone(sys.getprofile())

    def test_profile_remove_before_side_effect_retains_owner_until_explicit_retry(self):
        c = self.start()
        original = sys.setprofile
        def fail_before(callback):
            if callback is None:
                raise RuntimeError('CPU_BEFORE_PROFILE_REMOVED')
            original(callback)
        with patch.object(sys, 'setprofile', side_effect=fail_before):
            with self.assertRaisesRegex(RuntimeError, 'CPU_BEFORE_PROFILE_REMOVED'):
                c.enqueue(0)
            window = self.session.window
            self.assertIs(sys.getprofile(), window.profile_callback)
            result = c.client.unregister()
            self.assertEqual(result['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(result['ownership_retained'])
            self.assertIs(sys.getprofile(), window.profile_callback)
        c.ok(c.client.unregister())
        self.assertIsNone(sys.getprofile())

    def test_foreign_thread_cannot_read_or_remove_held_window_and_late_hook_refused(self):
        c = self.start()
        seen, failures = [], []
        def concurrent(module, args):
            window = self.session.window
            seen.append(window)
            def other_thread():
                for action in (window.readback, lambda: window.close(unwound=True)):
                    try:
                        action()
                    except RuntimeError as error:
                        failures.append(str(error))
            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join()
            self.assertIs(sys.getprofile(), window.profile_callback)
        handle = c.model.model.layers[0].register_forward_pre_hook(concurrent)
        try:
            c.request(0)
        finally:
            handle.remove()
        self.assertEqual(failures, ['POSTWRITE_SYNCHRONOUS_OWNER_REQUIRED']*2)
        window = seen[0]
        before = window.observer.inventory.report()
        with self.assertRaisesRegex(RuntimeError, 'POSTWRITE_LATE_OBSERVER_CALLBACK'):
            window._enter('late', c.model.model)
        self.assertEqual(window.observer.inventory.report(), before)
        self.assertFalse(window.profile_owned or window.frames or window.entries)
        c.ok(c.client.unregister())

    def test_current_delegate_change_in_real_hook_refuses_before_native(self):
        c = self.start()
        rotary = c.model.model.layers[0].rotary_emb
        original = rotary._forward_method
        def change(module, args):
            rotary._forward_method = rotary.different_branch
        handle = rotary.register_forward_pre_hook(change)
        try:
            with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_FRAME_CURRENT_BINDING_CHANGED'):
                c.enqueue(0)
            self.assertFalse(c.transport.opened)
            self.assertTrue(c.client.snapshot()['ownership_retained'])
        finally:
            handle.remove()
            rotary._forward_method = original
        c.ok(c.client.unregister())

    def test_source_change_is_checked_at_complete_boundary_not_profile_hot_path(self):
        c = self.start()
        changed, reads = [], []
        path = Path(__file__).resolve()
        original = Path.read_bytes
        def read(current):
            raw = original(current)
            if current.resolve() == path:
                reads.append(bool(changed))
                if changed:
                    return raw+b'\n# source changed after installation\n'
            return raw
        def mark(module, args):
            changed.append(True)
        handle = c.model.model.layers[0].register_forward_pre_hook(mark)
        try:
            with patch.object(Path, 'read_bytes', read):
                with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_FRAME_SOURCE_CHANGED'):
                    c.enqueue(0)
        finally:
            handle.remove()
        self.assertEqual(reads, [False, True])  # one install, one boundary; no per-frame I/O
        self.assertFalse(c.transport.opened)
        c.ok(c.client.unregister())

    def test_next_request_delegate_drift_cannot_reuse_previous_noncall_proof(self):
        c = self.start()
        for index in range(6):
            c.request(index)
        self.assertIsNotNone(self.provider.group)
        rotary = c.model.model.layers[0].rotary_emb
        original = rotary._forward_method
        rotary._forward_method = rotary.different_branch
        try:
            with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_ROTARY_BRANCH_BINDING_CHANGED'):
                c.enqueue(6)
            self.assertTrue(c.client.snapshot()['ownership_retained'])
        finally:
            rotary._forward_method = original
        c.recovery(7)


if __name__ == '__main__':
    unittest.main()
