"""CPU ctypes contract tests. Mock addresses/hashes are not NPU acceptance."""
import ctypes
import threading
import unittest

from demoncat_fault39.provider import CtypesProvider, NativeError, SERVING_INPUT_FIELDS
from demoncat_fault39.schema import Fault39Config
from test_contracts import FakeAbi, FakeFunction


VALUES = [0x1000, 32, 0x2000, 48, 2, 8, 3, 0, 0, 8, 1, 1, 8]


class ServingAbi(FakeAbi):
    def __init__(self):
        super().__init__()
        self.bindings = []
        self.next_code = 0
        self.fault39_set_serving_inputs_v1 = FakeFunction(self.bind_serving)

    def bind_serving(self, owner, values, count):
        payload = None if values is None else [values[i] for i in range(count)]
        self.bindings.append((owner.decode(), payload, count))
        return self.next_code


class ServingProviderTests(unittest.TestCase):
    def test_exact_cache_is_additive_and_requires_actual_owner_context(self):
        abi = ServingAbi()
        provider = CtypesProvider(abi)
        config = Fault39Config()
        call = dict(cache_address=0x1234, cache_epoch=9, target_signature='actual-native-signature')
        with self.assertRaisesRegex(ValueError, 'OWNED_CONTEXT'):
            provider.configure_cache('owner', config, 0, 0, **call)
        with provider.context('owner', ['request']):
            with self.assertRaisesRegex(RuntimeError, 'EXACT_CACHE_CONFIGURATION_UNAVAILABLE'):
                provider.configure_cache('owner', config, 0, 0, **call)
        seen = []
        def exact(raw, address, epoch, generation):
            seen.append((raw._obj.max_hits, raw._obj.target_signature, address, epoch))
            generation._obj.value = 71
            return 0
        abi.fault39_configure_cache_v2 = FakeFunction(exact)
        with provider.context('owner', ['request']):
            self.assertEqual(provider.configure_cache('owner', config, 0, 0, **call), 71)
            for changed in ({'cache_epoch': 0}, {'cache_address': True}, {'cache_address': 2**64}):
                with self.assertRaises(ValueError):
                    provider.configure_cache('owner', config, 0, 0, **dict(call, **changed))
            with self.assertRaises(ValueError):
                provider.configure_cache('owner', Fault39Config(max_hits=2), 0, 0, **call)
            with self.assertRaises(ValueError):
                provider.configure_cache('other', config, 0, 0, **call)
        self.assertEqual(seen, [(1, b'actual-native-signature', 0x1234, 9)])

    def test_legacy_attach_does_not_require_optional_guard_symbol(self):
        abi = FakeAbi()
        provider = CtypesProvider(abi, evidence_kind='CPU_CONTRACT_MOCK')
        with provider.context('owner', ['request']):
            with self.assertRaisesRegex(RuntimeError, 'SERVING_LIVE_INPUT_GUARD_UNAVAILABLE'):
                provider.bind_serving_inputs('owner', VALUES)

    def test_field_order_and_exact_uint64_marshalling(self):
        self.assertEqual(SERVING_INPUT_FIELDS, ('self_address', 'self_storage_bytes', 'mat2_address',
            'mat2_storage_bytes', 'M', 'K', 'N', 'self_offset', 'mat2_offset',
            'self_stride0', 'self_stride1', 'mat2_stride0', 'mat2_stride1'))
        abi = ServingAbi()
        provider = CtypesProvider(abi, evidence_kind='CPU_CONTRACT_MOCK')
        with provider.context('owner', ['actual-cpu-fixture-request']):
            provider.bind_serving_inputs('owner', VALUES)
        self.assertEqual(abi.bindings, [('owner', VALUES, 13)])
        self.assertEqual(abi.fault39_set_serving_inputs_v1.argtypes,
                         [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
        self.assertIsNone(abi.context)
        self.assertFalse(provider._context_local.active)
        self.assertIsNone(provider._context_local.owner)

    def test_missing_or_foreign_context_never_calls_native_setter(self):
        abi = ServingAbi()
        provider = CtypesProvider(abi)
        with self.assertRaisesRegex(RuntimeError, 'OWNED_SAME_THREAD'):
            provider.bind_serving_inputs('owner', VALUES)
        with provider.context('owner', ['request']):
            with self.assertRaisesRegex(RuntimeError, 'OWNED_SAME_THREAD'):
                provider.bind_serving_inputs('foreign', VALUES)
        self.assertEqual(abi.bindings, [])

    def test_context_cannot_be_inherited_by_foreign_thread(self):
        abi = ServingAbi()
        provider = CtypesProvider(abi)
        errors = []
        def other_thread():
            try:
                provider.bind_serving_inputs('owner', VALUES)
            except RuntimeError as exc:
                errors.append(str(exc))
        with provider.context('owner', ['request']):
            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join()
            self.assertEqual(len(errors), 1)
            self.assertIn('OWNED_SAME_THREAD', errors[0])
            provider.bind_serving_inputs('owner', VALUES)
        self.assertEqual(len(abi.bindings), 1)

    def test_invalid_python_values_poison_native_context_before_raising(self):
        for values in (None, VALUES[:-1], VALUES + [0], [True] + VALUES[1:],
                       [-1] + VALUES[1:], [2**64] + VALUES[1:], [1.0] + VALUES[1:],
                       dict(enumerate(VALUES))):
            with self.subTest(values=values):
                abi = ServingAbi()
                provider = CtypesProvider(abi)
                with provider.context('owner', ['request']):
                    with self.assertRaisesRegex(ValueError, 'EXACTLY_13_UINT64'):
                        provider.bind_serving_inputs('owner', values)
                self.assertEqual(abi.bindings, [('owner', None, 0)])

    def test_native_refusal_propagates_and_context_is_cleared(self):
        abi = ServingAbi()
        abi.next_code = -1
        provider = CtypesProvider(abi)
        with self.assertRaisesRegex(NativeError, 'set_serving_inputs: INVALID_CONFIG'):
            with provider.context('owner', ['request']):
                provider.bind_serving_inputs('owner', VALUES)
        self.assertIsNone(abi.context)
        self.assertIsNone(provider._context_local.owner)
        with self.assertRaisesRegex(RuntimeError, 'OWNED_SAME_THREAD'):
            provider.bind_serving_inputs('owner', VALUES)


if __name__ == '__main__':
    unittest.main()
