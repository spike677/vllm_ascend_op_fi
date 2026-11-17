"""Explicit CPU doubles; these tests never load CANN or claim Native effects."""
import ctypes as C
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fault_injection.core import UnsupportedCall
from fault_injection import quantization
from fault_injection import quant_native_isolation as isolation
from fault_injection import quant_group_native as group


# Read from actual CANN8.5.1 SDK build a3, manifest SHA
# 9c243c468fe42c12cc2af091dfb8642f4c9249f6b842ee2810e3306916654f0e.
SDK_A3_SIZES = dict(view=80, call=280, config=72, lease=392, event=360, binding=48, journal=104,
    plan=264, child=408, receipt=1032, state=160, public_input=104, completion=1368, runtime=416, raw=704)


def mapped(mode):
    return dict(mode=mode, path='/verified/renamed-' + mode + '.data', base=17, inode=23, device=29,
                sha256='a' * 64, symbols={})


class QuantIsolationTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(quantization._LEASE.locked())

    def test_old_preload_blocks_group_loader_before_cdll_or_path_reads(self):
        with patch.object(isolation, 'mapped_native_modes', return_value=(mapped('legacy'),)), patch.object(group.C, 'CDLL') as load:
            with self.assertRaisesRegex(UnsupportedCall, 'CANNOT_COLOAD'):
                group.QuantGroupNative.load('/missing', approved_build_sha256='a' * 64)
            load.assert_not_called()

    def test_group_mapping_blocks_all_old_managed_loaders_before_backend_constructor(self):
        for name in ('from_portable_v2_backend', 'from_scoped_backend', 'from_existing_backend'):
            with self.subTest(loader=name), patch.object(isolation, 'mapped_native_modes', return_value=(mapped('group'),)), \
                    patch.object(quantization.importlib.util, 'spec_from_file_location') as imported:
                with self.assertRaisesRegex(UnsupportedCall, 'CANNOT_COLOAD'):
                    getattr(quantization.PinnedQuantRuntime, name)('/missing', context_provider=lambda x: {}, approved_build_sha256='a' * 64)
                imported.assert_not_called()

    def test_existing_old_object_rechecks_group_mapping_at_arm(self):
        runtime = quantization.PinnedQuantRuntime.__new__(quantization.PinnedQuantRuntime)
        runtime.backend, runtime.poisoned = NS(arm_validated=Mock()), False
        with patch.object(isolation, 'mapped_native_modes', return_value=(mapped('group'),)):
            with self.assertRaisesRegex(UnsupportedCall, 'CANNOT_COLOAD'):
                runtime.arm(NS(input=object()), fault=True)
        runtime.backend.arm_validated.assert_not_called()
        self.assertTrue(runtime.poisoned)

    def test_failed_loader_releases_check_load_lock(self):
        factory = isolation.managed_loader('legacy')(Mock(side_effect=RuntimeError('construction failed')))
        with patch.object(isolation, 'mapped_native_modes', return_value=()):
            with self.assertRaisesRegex(RuntimeError, 'construction failed'):
                factory()
            self.assertFalse(quantization._LEASE.locked())
            with isolation.managed('group'):
                self.assertTrue(quantization._LEASE.locked())

    def test_concurrent_check_and_load_are_one_process_lease(self):
        entered, finish = threading.Event(), threading.Event()
        rows, errors = [], []
        @isolation.managed_loader('group')
        def slow_load():
            entered.set()
            if not finish.wait(3):
                raise RuntimeError('fixture synchronization failed')
            rows.append(mapped('group'))
        def worker():
            try:
                slow_load()
            except BaseException as error:
                errors.append(error)
        other_factory = Mock()
        with patch.object(isolation, 'mapped_native_modes', side_effect=lambda: tuple(rows)):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaisesRegex(UnsupportedCall, 'LEASE_BUSY'):
                    isolation.managed_loader('legacy')(other_factory)()
            finally:
                finish.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            other_factory.assert_not_called()
            # Releasing the lock did not unload the group DSO.
            with self.assertRaisesRegex(UnsupportedCall, 'CANNOT_COLOAD'):
                isolation.managed_loader('legacy')(other_factory)()

    def test_owned_old_session_arm_reuses_its_lock_and_still_checks_mapping(self):
        runtime = quantization.PinnedQuantRuntime.__new__(quantization.PinnedQuantRuntime)
        runtime.backend, runtime.poisoned = NS(arm_validated=Mock()), False
        runtime._managed_session_thread = threading.get_ident()
        self.assertTrue(quantization._LEASE.acquire(False))
        try:
            with patch.object(isolation, 'mapped_native_modes', return_value=(mapped('legacy'),)):
                runtime.arm(NS(input=object()), fault=False)
                runtime.backend.arm_validated.assert_called_once()
                self.assertTrue(quantization._LEASE.locked())
            with patch.object(isolation, 'mapped_native_modes', return_value=(mapped('legacy'), mapped('group'))):
                with self.assertRaisesRegex(UnsupportedCall, 'CANNOT_COLOAD'):
                    runtime.arm(NS(input=object()), fault=True)
                self.assertEqual(runtime.backend.arm_validated.call_count, 1)
        finally:
            quantization._LEASE.release()


class Function:
    def __init__(self, function):
        self.function = function
    def __call__(self, *args):
        return self.function(*args)


class FakeLibrary:
    """Only exercises Python ownership/ABI adapters; not a Native executor."""
    def __init__(self):
        self._handle = id(self)
        self.state = group.RuntimeState()
        self.remove_status = self.end_status = self.cancel_status = 0
        self.recover_calls = self.end_calls = 0
        operations = dict(abi=self.abi, state=self.query, install=self.install,
                          public_begin=self.begin, public_end=self.end, remove=self.remove, recover=self.recover,
                          cancel_empty=lambda *args: self.cancel_status)
        for name in ('abi', 'state', 'raw', 'install', 'public_begin', 'public_end', 'pass', 'continue', 'seal', 'stop', 'drain', 'cancel_empty', 'recover', 'remove'):
            setattr(self, 'dcat_dqg_' + name + '_v1', Function(operations.get(name, lambda *args: 0)))
    def abi(self, output):
        output = C.cast(output, C.POINTER(group.AbiSizes)).contents
        for name, size in SDK_A3_SIZES.items():
            setattr(output, name, size)
        return 0
    def query(self, output):
        C.memmove(output, C.byref(self.state), C.sizeof(self.state))
        return 0
    def install(self, config, observed, finished):
        self.state.core.installed = 1
        return 0
    def begin(self, config, public, token):
        C.cast(token, C.POINTER(group.U64)).contents.value = 1
        self.state.public_active = 1
        return 0
    def end(self, config, token, quant, scale, failed, result):
        self.end_calls += 1
        self.state.public_active = 0
        return -17 if failed else self.end_status
    def remove(self, *args):
        return self.remove_status
    def recover(self, *args):
        self.recover_calls += 1
        return 0


class QuantGroupBindingTests(unittest.TestCase):
    def setUp(self):
        self.scanner = patch.object(isolation, 'mapped_native_modes', return_value=())
        self.scanner.start()
        self.lib = FakeLibrary()
        self.runtime = group.QuantGroupNative(self.lib, dict(sdk_abi_sizes=SDK_A3_SIZES), _token=group._VERIFIED_LOAD)
        self.runtime._verify_mapped = Mock()
        self.config = group.Config(owner_hi=1, owner_lo=2, generation=1, thread=threading.get_native_id(), device=0,
                                   max_targets=8, max_calls=64, max_device_bytes=1024, max_host_bytes=4096)
        self.runtime.install(self.config, lambda call: 0, lambda done: 0)
    def tearDown(self):
        # Fixture-only global reset; no real Native ownership exists in these doubles.
        group._LIVE_OWNERS.clear()
        self.scanner.stop()
        self.assertFalse(quantization._LEASE.locked())
    def test_all_ctypes_layouts_match_root_actual_sdk_a3_receipt(self):
        self.assertEqual({name: C.sizeof(value) for name, value in group.ABI_TYPES.items()}, SDK_A3_SIZES)
    def test_sdk_layout_disagreement_refuses_binding(self):
        wrong = dict(SDK_A3_SIZES, runtime=408)
        with self.assertRaisesRegex(UnsupportedCall, 'LAYOUT_DIFFERS'):
            group.QuantGroupNative(FakeLibrary(), dict(sdk_abi_sizes=wrong), _token=group._VERIFIED_LOAD)
    def test_callback_exception_returns_error_and_keeps_callable_alive(self):
        self.lib.state.core.installed = 0
        other = group.QuantGroupNative(FakeLibrary(), dict(sdk_abi_sizes=SDK_A3_SIZES), _token=group._VERIFIED_LOAD)
        other._verify_mapped = Mock()
        def bad(_):
            raise RuntimeError('fixture callback failure')
        other.install(self.config, bad, bad)
        self.assertEqual(other._admission(C.pointer(group.Call())), -15)
        self.assertEqual(other._completed(C.pointer(group.Completion())), -15)
        self.assertIsNone(other._callback_phase)
        self.assertEqual(len(other.callback_errors), 2)
        for _ in range(128):
            self.assertEqual(other._admission(C.pointer(group.Call())), -15)
        self.assertEqual(other.callback_error_count, 130)
        self.assertEqual(other.callback_error_overflow, 2)
        public = other.callback_errors
        public[0]['phase'] = 'forged'
        self.assertEqual(other.callback_errors[0]['phase'], 'admission')
        self.assertIs(group._LIVE_OWNERS[other._registry_key], other)
    def test_second_owner_cannot_replace_existing_callback_retainer(self):
        other = group.QuantGroupNative(self.lib, dict(sdk_abi_sizes=SDK_A3_SIZES), _token=group._VERIFIED_LOAD)
        other._verify_mapped = Mock()
        with self.assertRaisesRegex(UnsupportedCall, 'OWNER_ALREADY_PRESENT'):
            other.install(self.config, lambda call: 0, lambda done: 0)
        self.assertIs(group._LIVE_OWNERS[self.runtime._registry_key], self.runtime)
    def test_exception_formatting_failure_still_returns_explicit_c_status(self):
        class BrokenRepr(Exception):
            def __repr__(self):
                raise RuntimeError('repr failed')
        other = group.QuantGroupNative(FakeLibrary(), dict(sdk_abi_sizes=SDK_A3_SIZES), _token=group._VERIFIED_LOAD)
        other._verify_mapped = Mock()
        def bad(_):
            raise BrokenRepr()
        other.install(self.config, bad, bad)
        self.assertEqual(other._admission(C.pointer(group.Call())), -15)
        self.assertEqual(other.callback_errors[0]['error'], '<exception representation failed>')
        self.assertIsNone(other._callback_phase)
    def test_callback_status_cannot_wrap_into_native_success(self):
        other = group.QuantGroupNative(FakeLibrary(), dict(sdk_abi_sizes=SDK_A3_SIZES), _token=group._VERIFIED_LOAD)
        other._verify_mapped = Mock()
        other.install(self.config, lambda call: 2 ** 32, lambda done: None)
        self.assertEqual(other._admission(C.pointer(group.Call())), -15)
        self.assertEqual(other._completed(C.pointer(group.Completion())), -15)
        self.assertEqual(other.callback_error_count, 2)
    def test_failed_public_call_keeps_input_and_never_attempts_late_recovery(self):
        value = object()
        self.runtime.tensor_view = Mock(return_value=group.View())
        def original():
            self.assertIs(self.runtime._held_input, value)
            self.lib.state.census.preparing = 1
            self.lib.state.quarantined = 1
            raise RuntimeError('actual public call failed')
        with self.assertRaisesRegex(RuntimeError, 'actual public call failed'):
            self.runtime.invoke(value, original)
        self.assertIs(self.runtime._held_input, value)
        self.assertEqual(self.lib.recover_calls, 0)
        self.assertEqual(self.lib.end_calls, 1)
        self.assertIsNone(self.runtime._managed_session_thread)
    def test_success_calls_original_once_and_returns_independent_snapshot(self):
        value, quant, scale = object(), object(), object()
        self.runtime.tensor_view = Mock(return_value=group.View())
        original = Mock(return_value=(quant, scale))
        output, receipt = self.runtime.invoke(value, original)
        self.assertEqual(output, (quant, scale))
        self.assertEqual(original.call_count, 1)
        self.assertIsNone(self.runtime._held_input)
        receipt['core']['target'][0] = 999
        state = self.runtime.query()
        state['core']['device_bytes'] = 999
        self.assertEqual(self.runtime.query()['core']['device_bytes'], 0)
    def test_rejected_public_end_is_not_repeated_or_recovered_after_return(self):
        self.runtime.tensor_view = Mock(return_value=group.View())
        self.lib.end_status = -17
        self.lib.state.quarantined = 1
        original = Mock(return_value=(object(), object()))
        value = object()
        with self.assertRaisesRegex(UnsupportedCall, 'public_end:-17'):
            self.runtime.invoke(value, original)
        self.assertEqual(original.call_count, 1)
        self.assertEqual(self.lib.end_calls, 1)
        self.assertEqual(self.lib.recover_calls, 0)
        self.assertIs(self.runtime._held_input, value)
    def test_pending_remove_retains_callbacks_and_python_owner(self):
        self.lib.remove_status = -10
        admission = self.runtime._admission
        with self.assertRaisesRegex(UnsupportedCall, 'remove:-10'):
            self.runtime.remove()
        self.assertIs(self.runtime._admission, admission)
        self.assertIs(group._LIVE_OWNERS[self.runtime._registry_key], self.runtime)
    def test_empty_cancel_keeps_callbacks_until_separate_successful_remove(self):
        admission = self.runtime._admission
        self.lib.cancel_status = -10
        self.assertFalse(self.runtime.cancel_empty())
        self.assertIs(group._LIVE_OWNERS[self.runtime._registry_key], self.runtime)
        self.lib.cancel_status = 0
        self.assertTrue(self.runtime.cancel_empty())
        self.assertIs(self.runtime._admission, admission)
        self.runtime.remove()
        self.assertNotIn(self.runtime._registry_key, group._LIVE_OWNERS)
    def test_nonempty_or_wrong_owner_cancellation_is_an_error(self):
        self.lib.cancel_status = -1
        with self.assertRaisesRegex(UnsupportedCall, 'EMPTY_CANCEL_IDENTITY'):
            self.runtime.cancel_empty()
        self.assertIs(group._LIVE_OWNERS[self.runtime._registry_key], self.runtime)
    def test_cpu_input_has_no_native_fallback(self):
        import torch
        with self.assertRaisesRegex(UnsupportedCall, 'ACTUAL_PUBLIC_TENSOR'):
            self.runtime.tensor_view(torch.ones(2, 4), 'input', 0)
    def test_private_constructor_and_unknown_row_hash_are_rejected(self):
        with self.assertRaisesRegex(UnsupportedCall, 'VERIFIED'):
            group.QuantGroupNative(self.lib, {})
        with self.assertRaisesRegex(UnsupportedCall, 'PROOF_REQUIRED'):
            group.key('0' * 64)
        with self.assertRaisesRegex(UnsupportedCall, 'ADMISSION_CALLBACK'):
            self.runtime.admit(group.Lease())


if __name__ == '__main__':
    unittest.main()
