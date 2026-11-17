"""Verified low-level quant group binding; no Serving/C2/C3 authority.

The managed public scope keeps the actual input alive. It never claims Python
ownership of outputs that have not returned, and cannot revive an expired
Native execute stack. Failed Native ownership retains this object/callbacks.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from .core import UnsupportedCall
from . import quant_native_isolation as isolation

U32, U64, I32, I64 = C.c_uint32, C.c_uint64, C.c_int32, C.c_int64
Key = U64 * 4


def _fields(names, kind=U64):
    return [(name, kind) for name in names.split()]


class Versioned(C.Structure):
    def __init__(self, **values):
        super().__init__(**values)
        self.version, self.size = 1, C.sizeof(type(self))


class View(C.Structure):
    _fields_ = (_fields('address bytes descriptor') + _fields('rows columns stride0 stride1 offset', I64)
                + _fields('dtype format', I32) + _fields('dimensions reserved', U32))


class Call(Versioned):
    _fields_ = (_fields('version size', U32) + _fields('token thread lifetime_token') + _fields('device reserved', I32)
                + [('input', View), ('quant', View), ('scale', View)])


class Config(Versioned):
    _fields_ = (_fields('version size max_targets max_calls', U32)
                + _fields('owner_hi owner_lo generation thread max_device_bytes max_host_bytes') + _fields('device reserved', I32))


class Lease(Versioned):
    _fields_ = (_fields('version size mode reserved', U32) + _fields('owner_hi owner_lo generation thread')
                + [('target', Key), ('actual_row_proof', Key), ('expected', Call)])


class NativeEvent(C.Structure):
    _fields_ = (_fields('kind reserved', U32) + _fields('sequence registration thread executor stream')
                + _fields('device status', I32) + _fields('original_calls workspace_required workspace workspace_capacity')
                + _fields('smooth_tensor group_index_tensor offset_tensor') + [('destination_dtype', I64)]
                + [('input', View), ('quant', View), ('scale', View)])


class ChildBinding(C.Structure):
    _fields_ = _fields('parent_token child_lease owner_hi owner_lo generation') + _fields('role reserved', U32)


class JournalState(C.Structure):
    _fields_ = _fields('total capacity overflow errors model_parents native_prepares native_executes '
                      'helper_prepares model_prepares unclassified_prepares preparing pending executing')


class ChildPlan(C.Structure):
    _fields_ = (_fields('role reserved', U32) + _fields('child_lease parent_token')
                + [('input', View), ('quant', View), ('scale', View)])


class ChildReceipt(C.Structure):
    _fields_ = ([('plan', ChildPlan)]
                + _fields('prepare_enter prepare_exit execute_enter execute_exit registration executor stream '
                          'prepare_original_calls execute_original_calls prepared returned drained readback '
                          'workspace_required workspace workspace_capacity workspace_group_owned')
                + _fields('prepare_status execute_status', I32))


class Receipt(Versioned):
    _fields_ = (_fields('version size mode reserved', U32) + _fields('error last_error', I32)
                + _fields('owner_hi owner_lo generation parent_token lifetime_token')
                + [('target', Key), ('actual_row_proof', Key), ('reference', ChildReceipt), ('effect', ChildReceipt)]
                + _fields('input_before_captured input_unchanged reference_effect_compared changed_quant_bytes changed_scale_bytes '
                          'published publication_exact public_bytes_restored released unsafe_allocation_address unsafe_allocation_bytes'))


class State(C.Structure):
    _fields_ = (_fields('installed stopped sealed owner_hi owner_lo generation targets admitted_model_calls pending_parent '
                       'native_child_prepares native_child_executes device_bytes host_bytes owned_descriptors stopped_targets '
                       'drained_targets owned_workspace_bytes borrowed_workspace_bytes quarantined_bytes') + [('error', I32)])


class PublicInput(Versioned):
    _fields_ = _fields('version size', U32) + [('thread', U64)] + _fields('device reserved', I32) + [('input', View)]


class Completion(Versioned):
    _fields_ = (_fields('version size', U32) + [('call', Call), ('core', Receipt)]
                + _fields('admitted prepare_enter prepare_exit execute_enter execute_exit') + _fields('status reserved', I32))


class RuntimeState(Versioned):
    _fields_ = (_fields('version size', U32) + [('core', State), ('census', JournalState)]
                + _fields('last_sequence public_active public_token native_stack_active callback_active completion_active '
                          'quarantined overlap_entries public_gws_entries public_execute_entries passthrough_calls admitted_calls '
                          'completed_calls rejected_calls unreturned_public_scope orphan_descriptors abi_exceptions')
                + _fields('last_error reserved', I32))


class RawRecord(Versioned):
    _fields_ = (_fields('version size kind reserved', U32)
                + [('parent', Call), ('native', NativeEvent), ('binding', ChildBinding)])


ABI_TYPES = dict(view=View, call=Call, config=Config, lease=Lease, event=NativeEvent, binding=ChildBinding,
                 journal=JournalState, plan=ChildPlan, child=ChildReceipt, receipt=Receipt, state=State,
                 public_input=PublicInput, completion=Completion, runtime=RuntimeState, raw=RawRecord)


class AbiSizes(Versioned):
    _fields_ = _fields('version size', U32) + [(name, U64) for name in ABI_TYPES]


Admission = C.CFUNCTYPE(C.c_int, C.POINTER(Call))
Completed = C.CFUNCTYPE(C.c_int, C.POINTER(Completion))
_VERIFIED_LOAD = object()
_LIVE_OWNERS = {}
PRODUCTION_SOURCES = ('batch_scale_group_core.h', 'batch_scale_group_runtime.h', 'batch_scale_group_runtime.cpp')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(value):
    if isinstance(value, C.Structure):
        return {name: snapshot(getattr(value, name)) for name, *_ in value._fields_}
    if isinstance(value, C.Array):
        return [snapshot(item) for item in value]
    return value


def copy_struct(value):
    return type(value).from_buffer_copy(bytes(value))


def key(sha256):
    if not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256) or sha256 == '0' * 64:
        raise UnsupportedCall('ACTUAL_NONZERO_SHA256_PROOF_REQUIRED')
    return Key.from_buffer_copy(bytes.fromhex(sha256))


class QuantGroupNative:
    def __init__(self, library, identity, *, _token=None):
        if _token is not _VERIFIED_LOAD:
            raise UnsupportedCall('USE_VERIFIED_QUANT_GROUP_LOADER')
        self.lib, self.identity = library, identity
        self.config = None
        self._admission = self._completed = None
        self._callback_phase = None
        self._managed_session_thread = None
        self._held_input = None
        self._registry_key = None
        self._callback_errors = []
        self.callback_error_count = self.callback_error_overflow = 0
        self._bind()

    @property
    def callback_errors(self):
        return [dict(row) for row in self._callback_errors]

    @classmethod
    @isolation.managed_loader('group')
    def load(cls, bundle, *, approved_build_sha256):
        root = Path(bundle).resolve(strict=True)
        manifest_path = root / 'build.json'
        if (not isinstance(approved_build_sha256, str) or not re.fullmatch('[0-9a-f]{64}', approved_build_sha256)
                or _sha(manifest_path) != approved_build_sha256):
            raise UnsupportedCall('QUANT_GROUP_APPROVED_MANIFEST_CHANGED')
        manifest = json.loads(manifest_path.read_bytes())
        if (manifest.get('schema') != 'demoncat.quant.group_deployment.v1' or manifest.get('cann') != '8.5.1'
                or manifest.get('actual_npu_calls') != 0 or manifest.get('legacy_c1_library_present') is not False
                or manifest.get('library') != 'runtime/vendor/op_api/lib/libcust_opapi.so'
                or manifest.get('library') not in manifest.get('files', {})):
            raise UnsupportedCall('QUANT_GROUP_DEPLOYMENT_IDENTITY_INCOMPLETE')
        for relative, expected in manifest['files'].items():
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or _sha(path) != expected:
                raise UnsupportedCall('QUANT_GROUP_BUNDLE_FILE_CHANGED:' + relative)
        for name in PRODUCTION_SOURCES:
            expected = _sha(Path(__file__).parent / 'native_adapters' / name)
            if (manifest['sources'].get('fault_injection/native_adapters/' + name) != expected
                    or manifest['files'].get(name) != expected):
                raise UnsupportedCall('QUANT_GROUP_SOURCE_DOES_NOT_MATCH_INSTALLED_BINDING:' + name)
        real = Path(os.environ.get('DCAT_REAL_OPAPI', ''))
        sdk = manifest['sdk_identity']
        if (not real.is_absolute() or str(real) != sdk.get('real_opapi') or not real.is_file()
                or _sha(real) != sdk.get('real_opapi_sha256') or os.environ.get('DCAT_DQ_GROUP_V1_ALLOWED') != '1'
                or os.environ.get('VLLM_SERVER_DEV_MODE') != '1' or os.environ.get('DCAT_DQ_V2_ENABLE') == '1'):
            raise UnsupportedCall('QUANT_GROUP_REAL_OPAPI_OR_PROCESS_MODE_CHANGED')
        path = (root / manifest['library']).resolve(strict=True)
        library = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        identity = dict(path=str(path), library_sha256=_sha(path), build_sha256=approved_build_sha256,
                        actual_image_id=manifest['actual_image_id'], source_commit=manifest['source_commit'],
                        sdk_abi_sizes=sdk['abi_sizes'], actual_npu_acceptance=False)
        result = cls(library, identity, _token=_VERIFIED_LOAD)
        result._verify_mapped()
        return result

    def _bind(self):
        pointer = C.POINTER
        arguments = dict(abi=[pointer(AbiSizes)], state=[pointer(RuntimeState)], raw=[U64, pointer(RawRecord)],
            install=[pointer(Config), Admission, Completed], public_begin=[pointer(Config), pointer(PublicInput), pointer(U64)],
            public_end=[pointer(Config), U64, pointer(View), pointer(View), C.c_int, pointer(Completion)],
            **{'pass': [pointer(Config), pointer(Call)], 'continue': [pointer(Lease)]},
            seal=[pointer(Config), pointer(Key), U64, U64, pointer(Key)], stop=[pointer(Config), pointer(Key)],
            drain=[pointer(Config), pointer(Key)], recover=[pointer(Config)], remove=[pointer(Config), Admission, Completed])
        for name, argtypes in arguments.items():
            function = getattr(self.lib, 'dcat_dqg_' + name + '_v1')
            function.argtypes, function.restype = argtypes, C.c_int
        sizes = AbiSizes()
        self._checked('abi', C.byref(sizes))
        actual = {name: getattr(sizes, name) for name in ABI_TYPES}
        if actual != {name: C.sizeof(cls) for name, cls in ABI_TYPES.items()} or actual != self.identity['sdk_abi_sizes']:
            raise UnsupportedCall('QUANT_GROUP_ACTUAL_DSO_SDK_CTYPES_LAYOUT_DIFFERS')

    def _verify_mapped(self):
        rows = isolation.require_mode('group')
        if len(rows) != 1:
            raise UnsupportedCall('EXACTLY_ONE_APPROVED_GROUP_DSO_MAPPING_REQUIRED')
        row = rows[0]
        if row['path'] != self.identity['path'] or row['sha256'] != self.identity['library_sha256']:
            raise UnsupportedCall('QUANT_GROUP_ACTUAL_MAPPED_SOURCE_CHANGED')
        for name in isolation.MARKERS['group']:
            if row['symbols'].get(name) != C.cast(getattr(self.lib, name), C.c_void_p).value:
                raise UnsupportedCall('QUANT_GROUP_HANDLE_SYMBOL_OWNER_CHANGED')

    def _checked(self, name, *arguments):
        status = getattr(self.lib, 'dcat_dqg_' + name + '_v1')(*arguments)
        if status:
            raise UnsupportedCall('QUANT_GROUP_NATIVE_REJECTED:' + name + ':' + str(status))
        return status

    def _state(self):
        state = RuntimeState()
        self._checked('state', C.byref(state))
        return state

    def query(self):
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            value = snapshot(self._state())
            value['binding_diagnostics'] = dict(callback_error_count=self.callback_error_count,
                callback_error_overflow=self.callback_error_overflow, errors=self.callback_errors)
            return value

    def raw(self, sequence):
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            row = RawRecord()
            self._checked('raw', sequence, C.byref(row))
            return snapshot(row)

    def install(self, config, observed, finished):
        if (type(config) is not Config or config.thread != threading.get_native_id()
                or not callable(observed) or not callable(finished) or self.config is not None):
            raise UnsupportedCall('QUANT_GROUP_ACTUAL_OWNER_AND_CALLBACKS_REQUIRED')
        with isolation.managed('group'):
            self._verify_mapped()
            state = self._state()
            if (state.core.installed or state.core.pending_parent or state.public_active
                    or state.callback_active or state.completion_active or state.quarantined or state.abi_exceptions):
                raise UnsupportedCall('QUANT_GROUP_NATIVE_OWNER_ALREADY_PRESENT_OR_UNCERTAIN')
            registry_key = (self.lib._handle, config.owner_hi, config.owner_lo, config.generation)
            if registry_key in _LIVE_OWNERS:
                raise UnsupportedCall('QUANT_GROUP_CALLBACK_OWNER_ALREADY_REGISTERED')
            self.config = copy_struct(config)
            def callback(function, phase):
                def run(pointer):
                    self._callback_phase = phase
                    try:
                        status = function(copy_struct(pointer.contents))
                        if type(status) is not int or not -(2 ** 31) <= status < 2 ** 31:
                            raise TypeError('Callback must explicitly return a signed C int Native control status')
                        return status
                    except BaseException as error:
                        try:
                            self.callback_error_count += 1
                            if len(self._callback_errors) < 128:
                                try:
                                    detail = repr(error)
                                except BaseException:
                                    detail = '<exception representation failed>'
                                self._callback_errors.append(dict(phase=phase, error=detail[:2048], truncated=len(detail) > 2048))
                            else:
                                self.callback_error_overflow += 1
                        except BaseException:
                            # Even diagnostic allocation/formatting failure
                            # must return a deterministic C error status.
                            pass
                        return -15
                    finally:
                        self._callback_phase = None
                return run
            self._admission, self._completed = Admission(callback(observed, 'admission')), Completed(callback(finished, 'completion'))
            self._registry_key = registry_key
            _LIVE_OWNERS[self._registry_key] = self
            try:
                self._checked('install', C.byref(self.config), self._admission, self._completed)
            except BaseException:
                state = self._state()
                if not state.core.installed and not state.core.pending_parent and not state.callback_active and not state.completion_active:
                    _LIVE_OWNERS.pop(self._registry_key, None)
                    self.config = self._admission = self._completed = None
                raise

    def pass_call(self, call):
        if self._callback_phase != 'admission' or type(call) is not Call:
            raise UnsupportedCall('QUANT_GROUP_PASS_REQUIRES_ACTUAL_ADMISSION_CALLBACK')
        return self._checked('pass', C.byref(self.config), C.byref(call))

    def admit(self, lease):
        if self._callback_phase != 'admission' or type(lease) is not Lease:
            raise UnsupportedCall('QUANT_GROUP_LEASE_REQUIRES_ACTUAL_ADMISSION_CALLBACK')
        return self._checked('continue', C.byref(lease))

    @staticmethod
    def tensor_view(value, role, device):
        import torch
        expected = {'input': (torch.float16, torch.bfloat16), 'quant': (torch.int8,), 'scale': (torch.float32,)}
        if (not isinstance(value, torch.Tensor) or value.device.type != 'npu' or value.device.index != device
                or value.dtype not in expected[role] or not value.is_contiguous() or value.storage_offset() != 0
                or value.ndim != (1 if role == 'scale' else 2)):
            raise UnsupportedCall('QUANT_GROUP_ACTUAL_PUBLIC_TENSOR_REQUIRED:' + role)
        result = View()
        result.address, result.bytes = value.data_ptr(), value.numel() * value.element_size()
        result.rows, result.columns = value.shape[0], value.shape[1] if value.ndim == 2 else 0
        result.stride0, result.stride1 = value.stride(0), value.stride(1) if value.ndim == 2 else 0
        result.dtype = {torch.float32: 0, torch.float16: 1, torch.int8: 2, torch.bfloat16: 27}[value.dtype]
        result.format, result.dimensions = 2, value.ndim
        return result

    def invoke(self, input_tensor, original):
        if self.config is None or not callable(original) or self._held_input is not None:
            raise UnsupportedCall('QUANT_GROUP_PUBLIC_SCOPE_NOT_READY')
        with isolation.managed('group'):
            self._verify_mapped()
            self._managed_session_thread = threading.get_ident()
            self._held_input = input_tensor  # Strong input ownership, including uncertain failed Native executors.
            token, result = U64(), Completion()
            begun = end_attempted = False
            try:
                public = PublicInput()
                public.thread, public.device = threading.get_native_id(), self.config.device
                public.input = self.tensor_view(input_tensor, 'input', self.config.device)
                self._checked('public_begin', C.byref(self.config), C.byref(public), C.byref(token))
                begun = True
                output = original()  # Exactly once; Native helpers remain separately visible in the raw census.
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise UnsupportedCall('QUANT_GROUP_PUBLIC_Q_SCALE_RETURN_REQUIRED')
                quant = self.tensor_view(output[0], 'quant', self.config.device)
                scale = self.tensor_view(output[1], 'scale', self.config.device)
                end_attempted = True
                self._checked('public_end', C.byref(self.config), token.value, C.byref(quant), C.byref(scale), 0, C.byref(result))
                self._verify_mapped()
                return output, snapshot(result)
            except BaseException:
                if begun and not end_attempted:
                    # A failed public operation may already have destroyed its
                    # outputs. Never reconstruct them or request late recovery.
                    self.lib.dcat_dqg_public_end_v1(C.byref(self.config), token.value, None, None, 1, C.byref(result))
                raise
            finally:
                self._managed_session_thread = None
                state = self._state()
                if (not state.public_active and not state.core.pending_parent and not state.census.preparing
                        and not state.census.pending and not state.census.executing and not state.quarantined
                        and not state.abi_exceptions and not state.orphan_descriptors and not state.core.quarantined_bytes):
                    self._held_input = None

    def seal(self, target_hashes, scope_sha256, *, unresolved_candidates):
        values = [key(value) for value in target_hashes]
        array = (Key * len(values))(*values)
        scope = key(scope_sha256)
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            self._checked('seal', C.byref(self.config), array, len(array), unresolved_candidates, C.byref(scope))

    def stop(self, target_sha256):
        target = key(target_sha256)
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            self._checked('stop', C.byref(self.config), C.byref(target))

    def drain(self, target_sha256):
        target = key(target_sha256)
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            status = self.lib.dcat_dqg_drain_v1(C.byref(self.config), C.byref(target))
            if status not in (0, -10):
                raise UnsupportedCall('QUANT_GROUP_DRAIN_IDENTITY_OR_RUNTIME_FAILURE:' + str(status))
            return status == 0

    def recover(self):
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            self._checked('recover', C.byref(self.config))
            return snapshot(self._state())

    def remove(self):
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            self._checked('remove', C.byref(self.config), self._admission, self._completed)
            _LIVE_OWNERS.pop(self._registry_key, None)
            self.config = self._admission = self._completed = self._held_input = None
