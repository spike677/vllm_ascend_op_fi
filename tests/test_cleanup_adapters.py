"""Cleanup ownership tests; all tensors/providers are CPU fixtures, no NPU calls."""
from pathlib import Path
import subprocess
import sys
import textwrap
import types
import unittest
from unittest.mock import patch

import torch

from fault_injection.alias import AliasSession
from fault_injection.attention import AttentionSession
from fault_injection.module_faults import ModuleFaultSession
from fault_injection.quantization import QuantizationSession
from test_alias_auto import SimulatedFormat
from test_quant_auto import CPUProvider, AscendW8A8DynamicLinearMethod
from test_attention_interface import Registry, Masks, Model, AttentionBlock, Backend, Tensor


class QuantUnit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.ones(3, 3, dtype=torch.int8))
        self.register_buffer('bias', torch.zeros(3, dtype=torch.float16))
        self.quant_method = AscendW8A8DynamicLinearMethod()
        self.tp_size, self.tp_rank = 2, 0

    def forward(self, x):
        return self.quant_method.apply(self, x)


class OnceFailedHandle:
    def __init__(self, handle):
        self.handle, self.first = handle, True

    def remove(self):
        self.handle.remove()
        if self.first:
            self.first = False
            raise RuntimeError('one removal reply lost')


class CleanupAdapters(unittest.TestCase):
    def run_child(self, source):
        code = 'import sys\nsys.path.insert(0, "tests")\n' + textwrap.dedent(source)
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_install_and_cleanup_errors_both_survive_and_owner_can_retry(self):
        for kind in ('module', 'alias', 'attention', 'quant'):
            with self.subTest(adapter=kind), patch(
                    'fault_injection.attention._interfaces', return_value=(Registry, Masks)):
                if kind == 'module':
                    model = torch.nn.RMSNorm(3).eval()
                    factory = lambda: ModuleFaultSession(model, '18.api_half')
                elif kind == 'alias':
                    model = torch.nn.Identity()
                    factory = lambda: AliasSession(model, _format_provider=SimulatedFormat())
                elif kind == 'attention':
                    config = types.SimpleNamespace(_attn_implementation='eager')
                    model = Model([('arbitrary', AttentionBlock(config))])
                    factory = lambda: AttentionSession(model, Backend())
                else:
                    model = torch.nn.Sequential(QuantUnit()).eval()
                    factory = lambda: QuantizationSession(model, runtime=CPUProvider(model))
                session = factory()
                original_pre = model.register_forward_pre_hook
                def pre(*args, **kwargs):
                    return OnceFailedHandle(original_pre(*args, **kwargs))
                try:
                    with patch.object(model, 'register_forward_pre_hook', side_effect=pre), patch.object(
                            model, 'register_forward_hook', side_effect=ValueError('install fixture failure')):
                        with self.assertRaisesRegex(RuntimeError, 'install fixture failure.*one removal reply lost') as caught:
                            session.__enter__()
                    self.assertIsInstance(caught.exception.__cause__, ValueError)
                    self.assertTrue(session._locked)
                    with self.assertRaises(RuntimeError):
                        factory().__enter__()
                finally:
                    session.close()
                self.assertFalse(session._locked)
                with factory():
                    pass

    def test_module_refuses_close_during_forward_and_can_close_after(self):
        model = torch.nn.Sequential(torch.nn.RMSNorm(3)).eval()
        session = ModuleFaultSession(model, '18.api_half')
        session.__enter__()
        def inside(*args):
            with self.assertRaisesRegex(RuntimeError, 'active model forward'):
                session.close()
            self.assertTrue(session.active)
        hook = model.register_forward_pre_hook(inside)
        try:
            model(torch.ones(1, 3))
            session.assert_coverage()
        finally:
            hook.remove()
            session.close()
        self.assertTrue(session.closed)

    def test_alias_refuses_close_during_forward_and_can_close_after(self):
        model = torch.nn.Identity()
        session = AliasSession(model, _format_provider=SimulatedFormat())
        session.__enter__()
        def inside(*args):
            with self.assertRaisesRegex(RuntimeError, 'active model forward'):
                session.close()
            self.assertTrue(session._active)
        hook = model.register_forward_pre_hook(inside)
        try:
            model(torch.ones(1))
        finally:
            hook.remove()
            session.close()
        self.assertTrue(session._closed)

    def test_quant_refuses_close_during_forward_and_can_close_after(self):
        model = torch.nn.Sequential(QuantUnit()).eval()
        session = QuantizationSession(model, runtime=CPUProvider(model))
        session.__enter__()
        def inside(*args):
            with self.assertRaisesRegex(RuntimeError, 'active model forward'):
                session.close()
            self.assertTrue(session._active)
        hook = model.register_forward_pre_hook(inside)
        try:
            model(torch.ones(2, 3, dtype=torch.float16))
            session.assert_coverage()
        finally:
            hook.remove()
            session.close()
        self.assertTrue(session._closed)

    def test_attention_refuses_active_close_then_recovers_failed_handle(self):
        config = types.SimpleNamespace(_attn_implementation='eager')
        model = Model([('arbitrary', AttentionBlock(config))])
        with patch('fault_injection.attention._interfaces', return_value=(Registry, Masks)):
            session = AttentionSession(model, Backend())
            session.__enter__()
            def inside():
                with self.assertRaisesRegex(RuntimeError, 'active model forward'):
                    session.close()
                self.assertTrue(session._active)
            model.before = inside
            q, k, v = Tensor((1, 2, 3, 4)), Tensor((1, 1, 3, 4)), Tensor((1, 1, 3, 4))
            model(q, k, v, None)
            session._handles[0] = OnceFailedHandle(session._handles[0])
            with self.assertRaisesRegex(RuntimeError, 'one removal reply lost'):
                session.close()
            self.assertFalse(session._closed)
            self.assertEqual(config._attn_implementation, 'eager')
            with self.assertRaisesRegex(RuntimeError, 'owns the process registry'):
                AttentionSession(model, Backend()).__enter__()
            session.close()
            self.assertTrue(session._closed)

    def test_quant_hook_retry_retains_lease(self):
        model = torch.nn.Sequential(QuantUnit()).eval()
        session = QuantizationSession(model, runtime=CPUProvider(model))
        session.__enter__()
        session._hooks[0] = OnceFailedHandle(session._hooks[0])
        with self.assertRaisesRegex(RuntimeError, 'one removal reply lost'):
            session.close()
        self.assertFalse(session._closed)
        self.assertNotIn('apply', vars(model[0].quant_method))
        with self.assertRaisesRegex(RuntimeError, 'owns the process scope'):
            QuantizationSession(model, runtime=CPUProvider(model)).__enter__()
        session.close()
        with QuantizationSession(model, runtime=CPUProvider(model)):
            pass

    def test_foreign_conv_wrapper_is_not_overwritten_and_owner_can_retry(self):
        model = torch.nn.Conv3d(1, 1, (1, 16, 16), stride=(1, 16, 16)).eval()
        original = torch.nn.functional.conv3d
        session = ModuleFaultSession(model, '17.spatial_padding')
        session.__enter__()
        foreign = lambda *args, **kwargs: original(*args, **kwargs)
        torch.nn.functional.conv3d = foreign
        try:
            with self.assertRaisesRegex(RuntimeError, 'Foreign conv3d'):
                session.close()
            self.assertIs(torch.nn.functional.conv3d, foreign)
            self.assertFalse(session.closed)
        finally:
            torch.nn.functional.conv3d = original
            session.close()

    def test_retained_module_hooks_do_not_admit_new_faults_after_close_failure(self):
        model = torch.nn.Sequential(torch.nn.RMSNorm(3)).eval()
        x = torch.ones(1, 3)
        baseline = model(x)
        session = ModuleFaultSession(model, '18.api_half')
        session.__enter__()
        class FailBeforeRemove:
            def __init__(self, handle):
                self.handle, self.first = handle, True
            def remove(self):
                if self.first:
                    self.first = False
                    raise RuntimeError('still installed')
                self.handle.remove()
        session.handles = [FailBeforeRemove(h) for h in session.handles]
        with self.assertRaisesRegex(RuntimeError, 'still installed'):
            session.close()
        self.assertFalse(session.active)
        self.assertTrue(model._forward_pre_hooks)
        self.assertTrue(torch.equal(model(x), baseline))
        self.assertEqual(session.events, [])
        session.close()
        self.assertFalse(model._forward_pre_hooks)

    def test_alias_failed_mode_exit_never_pops_another_mode_on_retry(self):
        self.run_child('''
            import torch
            from fault_injection.alias import AliasSession
            from test_alias_auto import SimulatedFormat
            model = torch.nn.Identity()
            original = torch.as_strided
            session = AliasSession(model, _format_provider=SimulatedFormat())
            session.__enter__()
            mode = session._mode
            calls = []
            class LostReply:
                def __exit__(self, *args):
                    calls.append(1)
                    mode.__exit__(*args)
                    raise RuntimeError('mode cleanup reply lost')
            session._mode = LostReply()
            try: session.close()
            except RuntimeError as e: assert 'mode cleanup reply lost' in str(e)
            else: raise AssertionError('uncertain mode close accepted')
            assert torch.as_strided is original
            assert not model._forward_hooks and not model._forward_pre_hooks
            assert session.report()['mode_cleanup_pending'] and not session._closed
            try: session.close()
            except RuntimeError as e: assert 'process restart required' in str(e)
            else: raise AssertionError('unknown mode state released')
            assert len(calls) == 1
            try: AliasSession(model, _format_provider=SimulatedFormat()).__enter__()
            except RuntimeError: pass
            else: raise AssertionError('quarantined process admitted another alias owner')
        ''')

    def test_native_quant_failed_finish_quarantines_despite_python_restore(self):
        self.run_child('''
            import torch
            from fault_injection.quantization import QuantizationSession
            from test_quant_auto import CPUProvider
            from test_cleanup_adapters import QuantUnit
            class SimulatedNativeProvider(CPUProvider):
                # Only tests controller uncertainty. No NPU/library is used.
                native, test_only = True, False
            model = torch.nn.Sequential(QuantUnit()).eval()
            provider = SimulatedNativeProvider(model)
            provider.cleanup_error = True
            session = QuantizationSession(model, runtime=provider)
            session.__enter__()
            try: model(torch.ones(2, 3, dtype=torch.float16))
            except RuntimeError as e: assert 'cleanup failed' in str(e)
            else: raise AssertionError('finish failure missing')
            for _ in range(2):
                try: session.close()
                except RuntimeError as e: assert 'Native quantization cleanup unconfirmed' in str(e)
                else: raise AssertionError('unknown native state released')
            assert session.report()['native_cleanup_pending'] and not session._closed
            assert not model._forward_hooks and not model._forward_pre_hooks
            assert 'apply' not in vars(model[0].quant_method)
            try: QuantizationSession(model, runtime=CPUProvider(model)).__enter__()
            except RuntimeError: pass
            else: raise AssertionError('native uncertainty admitted a new owner')
        ''')


if __name__ == '__main__':
    unittest.main()
