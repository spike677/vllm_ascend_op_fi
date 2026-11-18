"""Exact v2 Native writer control; CPU tests do not confer Native authority."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .writer26_inventory import require


class Config(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('version', 'size', 'mechanism', 'max_targets')]
    _fields_ += [(n, C.c_uint64) for n in ('owner_hi', 'owner_lo', 'generation', 'thread', 'max_calls', 'max_device_bytes', 'max_host_bytes')]


class Call(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('version', 'size', 'mechanism', 'dimensions')]
    _fields_ += [(n, C.c_int32) for n in ('device', 'dtype', 'format', 'reserved')]
    _fields_ += [(n, C.c_uint64) for n in ('token', 'thread', 'independent_call', 'registration', 'cache',
        'producer_sequence', 'output', 'output_bytes', 'residual', 'residual_bytes')]
    _fields_ += [('shape', C.c_int64 * 4)]


class Lease(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('version', 'size', 'mode', 'reserved')]
    _fields_ += [(n, C.c_uint64) for n in ('owner_hi', 'owner_lo', 'generation', 'thread')]
    _fields_ += [('target', C.c_uint64 * 4), ('expected', Call)]
    _fields_ += [(n, C.c_uint64) for n in ('residual_allocation_end', 'output_allocation_begin', 'shift_bytes')]


class Receipt(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('version', 'size', 'mode', 'mechanism')]
    _fields_ += [(n, C.c_int32) for n in ('error', 'writer_status', 'consumer_status', 'reserved')]
    _fields_ += [(n, C.c_uint64) for n in ('owner_hi', 'owner_lo', 'generation', 'token', 'independent_call',
        'registration', 'cache', 'producer_sequence', 'output', 'output_bytes', 'residual', 'residual_bytes',
        'replacement', 'stream', 'executor')]
    _fields_ += [('target', C.c_uint64 * 4)]
    _fields_ += [(n, C.c_uint64) for n in ('writer_calls', 'consumer_calls', 'changed_residual_bytes', 'writer_complete',
        'consumer_complete', 'output_readback_exact', 'padding_readback_exact', 'residual_unchanged',
        'residual_restored_exact', 'reference_equal', 'lossless_qualified', 'off_exact', 'drained', 'released')]


class State(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('version', 'size', 'mechanism', 'reserved')]
    _fields_ += [(n, C.c_int32) for n in ('error', 'reserved2')]
    _fields_ += [(n, C.c_uint64) for n in ('installed', 'stopped', 'sealed', 'owner_hi', 'owner_lo', 'generation',
        'thread', 'targets', 'calls', 'pending', 'executing', 'device_bytes', 'host_bytes', 'descriptor_pending')]


class RuntimeState(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('core', State)]
    _fields_ += [(n, C.c_uint64) for n in ('callback_active', 'completion_active', 'prepared_fia', 'fia_executor',
        'descriptor', 'callback_calls', 'passthrough_calls', 'admitted_calls', 'completion_calls', 'rejected_calls', 'foreign_calls')]
    _fields_ += [('last_rejection', C.c_int32), ('reserved', C.c_int32)]


ADMIT = C.CFUNCTYPE(C.c_int, C.POINTER(Call))
COMPLETE = C.CFUNCTYPE(C.c_int, C.POINTER(Receipt))
MODES = dict(reference=1, lossless=2, fault=3, off=4)


def packet(value):
    result = {}
    for name, _ in value._fields_:
        item = getattr(value, name)
        result[name] = packet(item) if isinstance(item, C.Structure) else list(item) if isinstance(item, C.Array) else int(item)
    return result


def key_words(sha256):
    require(isinstance(sha256, str) and len(sha256) == 64 and all(n in '0123456789abcdef' for n in sha256),
            'FULL_TARGET_SHA256_REQUIRED')
    return tuple(int(sha256[i:i + 16], 16) for i in range(0, 64, 16))


class NativeWriter26Group:
    def __init__(self, *, mechanism, bundle, manifest_sha256, generation=1, owner=None):
        require(mechanism in ('26.residual_stomp', '26.fia_destination'), 'EXACT_WRITER_MECHANISM_REQUIRED')
        root = Path(bundle).resolve(strict=True)
        raw = (root / 'build.json').read_bytes()
        require(hashlib.sha256(raw).hexdigest() == manifest_sha256, 'GROUP_BUILD_MANIFEST_CHANGED')
        build = json.loads(raw)
        require(build.get('schema') == 'demoncat.writer26.group_deployment.v2' and build.get('legacy_writers_absent') is True
                and build.get('actual_npu_calls') == 0, 'DISTINCT_NEW_GROUP_ABI_BUILD_REQUIRED')
        for name, expected in build['files'].items():
            path = (root / name).resolve(strict=True)
            require(path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    'GROUP_BUILD_FILE_CHANGED:' + name)
        require(all(not (root / name).exists() for name in
                    ('libfault26_qmm_output_underflow_abi0.so', 'libfault26_fia_writer_proxy_abi0.so')),
                'LEGACY_WRITER_MUST_NOT_SHARE_V2_BUNDLE')
        require(os.environ.get('TASK_QUEUE_ENABLE') == '0', 'SERIAL_NATIVE_WRITER_REQUIRED')
        self.lib = C.CDLL(str(root / 'libdcat26_group_v2.so'), mode=os.RTLD_LAZY | os.RTLD_NOLOAD)
        self.root = root
        self.build, self.build_identity, self.sha256 = build, manifest_sha256, build['files']['libdcat26_group_v2.so']
        from ..metadata_arrays import LoadedWriter26MetadataArrays
        self._metadata_arrays = LoadedWriter26MetadataArrays(bundle=root, manifest_sha256=manifest_sha256)
        self.mechanism, self.fia = mechanism, mechanism == '26.fia_destination'
        self.owner = owner or uuid.uuid4().hex
        words = key_words(hashlib.sha256(self.owner.encode()).hexdigest())
        self.config = Config(version=2, size=C.sizeof(Config), mechanism=2 if self.fia else 1,
            max_targets=512, owner_hi=words[0], owner_lo=words[1], generation=generation,
            thread=threading.get_native_id(), max_calls=8192, max_device_bytes=64 * 1024 * 1024, max_host_bytes=512 * 1024 * 1024)
        self.admit_callback = self.complete_callback = None
        self.bound_functions = {}
        signatures = dict(state=[C.POINTER(RuntimeState)], install=[C.POINTER(Config), ADMIT, COMPLETE],
            **{'pass': [C.POINTER(Config), C.POINTER(Call)]}, **{'continue': [C.POINTER(Lease)]},
            receipt=[C.POINTER(Config), C.c_uint64, C.POINTER(Receipt)],
            acknowledge=[C.POINTER(Config), C.c_uint64], seal=[C.POINTER(Config), C.POINTER(C.c_uint64 * 4), C.c_uint64],
            stop=[C.POINTER(Config)], drain=[C.POINTER(Config)], restore=[C.POINTER(Config), C.c_uint64],
            remove=[C.POINTER(Config), ADMIT, COMPLETE])
        for name, signature in signatures.items():
            function = getattr(self.lib, 'dcat26_group_' + name + '_v2')
            function.argtypes, function.restype = signature, C.c_int
            self.bound_functions[name] = function
        require([C.sizeof(t) for t in (Config, Call, Lease, Receipt, State, RuntimeState)] == [72, 144, 248, 296, 136, 240],
                'GROUP_NATIVE_ABI_SIZE_CHANGED')
        self.fia_typed = None
        if self.fia and 'fia_typed_profile' in build:
            from .writer26_fia_typed import NativeTypedFia
            self.fia_typed = NativeTypedFia(self, root)

    def fn(self, name, *args):
        require(threading.get_native_id() == self.config.thread, 'FOREIGN_GROUP_NATIVE_THREAD')
        status = self.bound_functions[name](*args)
        require(status == 0, 'NATIVE_GROUP_' + name.upper() + '_FAILED:' + str(status))

    def metadata_arrays(self):
        return self._metadata_arrays.read()

    def environment_bindings(self):
        """New builds only; retain actual holders, never expose a lookup as use."""
        from ..environment import NativeSymbolHolder
        require(threading.get_native_id() == self.config.thread, 'FOREIGN_GROUP_NATIVE_THREAD')
        require(self.build.get('bound_symbol_profile') == 'demoncat.writer26.cached_addresses.v1',
                'FRESH_WRITER26_CACHED_ADDRESS_BUILD_REQUIRED')
        symbols = ('aclrtGetDevice', 'aclGetDataType', 'aclGetFormat', 'aclGetViewOffset',
            'aclGetViewShape', 'aclGetStorageShape', 'aclGetViewStrides', 'aclGetRawTensorAddr',
            'aclCreateTensor', 'aclDestroyTensor', 'aclrtMalloc', 'aclrtFree', 'aclrtMemcpy', 'aclrtSynchronizeStream')
        name = 'dcat26_group_cached_address_v1'
        group = NativeSymbolHolder(self.lib, getattr(self.lib, name), name,
            str(self.root/'libdcat26_group_v2.so'), self.sha256, tuple(enumerate(symbols)))
        custom_path = self.root/'libcust_opapi.so'
        require(hashlib.sha256(custom_path.read_bytes()).hexdigest() == self.build['files']['libcust_opapi.so'],
                'WRITER26_CUSTOM_PROXY_CHANGED')
        custom = C.CDLL(str(custom_path), mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_NOLOAD)
        name = 'dcat26_custom_cached_address_v1'
        proxy = NativeSymbolHolder(custom, getattr(custom, name), name, str(custom_path),
            self.build['files']['libcust_opapi.so'], ((0, 'aclnnAddRmsNormBias'), (1, 'aclnnAddRmsNormBiasGetWorkspaceSize')))
        return (group, proxy)

    def state(self):
        value = RuntimeState(version=2, size=C.sizeof(RuntimeState))
        self.fn('state', C.byref(value))
        require(value.version == 2 and value.size == C.sizeof(value), 'GROUP_STATE_ABI_CHANGED')
        return packet(value)

    def install(self, admit, complete):
        require(self.admit_callback is self.complete_callback is None, 'GROUP_CALLBACK_ALREADY_OWNED')
        self.admit_callback, self.complete_callback = ADMIT(admit), COMPLETE(complete)
        try:
            self.fn('install', C.byref(self.config), self.admit_callback, self.complete_callback)
        except BaseException:
            self.admit_callback = self.complete_callback = None
            raise

    def passthrough(self, call):
        self.fn('pass', C.byref(self.config), C.byref(call))

    def execute(self, call, mode, target, allocation=None, *, typed_before=None):
        lease = Lease(version=2, size=C.sizeof(Lease), mode=MODES[mode], owner_hi=self.config.owner_hi,
            owner_lo=self.config.owner_lo, generation=self.config.generation, thread=self.config.thread,
            target=(C.c_uint64 * 4)(*key_words(target)), expected=call)
        if not self.fia:
            require(allocation is not None, 'ACTUAL_LIVE_ALLOCATION_REQUIRED')
            lease.residual_allocation_end = allocation['residual_block'][1]
            lease.output_allocation_begin = allocation['output_block'][0]
            lease.shift_bytes = allocation['shift']
        if self.fia_typed is not None:
            require(typed_before is not None, 'CURRENT_TYPED_FIA_ADMISSION_PROOF_REQUIRED')
            return self.fia_typed.execute(lease, typed_before)
        require(typed_before is None, 'TYPED_FIA_REQUIRES_NEW_BOUND_ABI')
        self.fn('continue', C.byref(lease))
        return packet(lease)

    def receipt(self, token):
        value = Receipt(version=2, size=C.sizeof(Receipt))
        self.fn('receipt', C.byref(self.config), token, C.byref(value))
        require(value.version == 2 and value.size == C.sizeof(value) and value.token == token, 'GROUP_RECEIPT_IDENTITY_CHANGED')
        return packet(value)

    def acknowledge(self, token):
        self.fn('acknowledge', C.byref(self.config), token)
        return self.receipt(token)

    def seal(self, targets):
        keys = ((C.c_uint64 * 4) * len(targets))(*[(C.c_uint64 * 4)(*key_words(target)) for target in targets])
        self.fn('seal', C.byref(self.config), keys, len(targets))

    def stop(self):
        self.fn('stop', C.byref(self.config))

    def drain(self):
        self.fn('drain', C.byref(self.config))

    def restore(self, token):
        self.fn('restore', C.byref(self.config), token)

    def close(self):
        self.stop()
        self.drain()
        require(not self.state()['core']['pending'], 'EXPLICIT_LIVE_TARGET_RESTORE_REQUIRED')
        self.fn('remove', C.byref(self.config), self.admit_callback, self.complete_callback)
        self.admit_callback = self.complete_callback = None
        state = self.state()
        require(all(state['core'][n] == 0 for n in ('installed', 'pending', 'executing', 'device_bytes', 'host_bytes', 'descriptor_pending'))
                and not state['descriptor'] and not state['callback_active'] and not state['completion_active'],
                'NATIVE_GROUP_RESOURCES_NOT_RELEASED')
        return state
