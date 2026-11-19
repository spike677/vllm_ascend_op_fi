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
from .prerequisites import expectation, require_assessment, require_identity
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


METADATA_DIMENSIONS, METADATA_CAPACITY = 8, 8192


class MetadataView(C.Structure):
    _fields_ = (_fields('descriptor address') + [('offset', I64)] + _fields('dtype format', I32)
                + _fields('stage last_getter', U32) + _fields('called_mask null_arrays truncated_arrays')
                + [('getter_status', I32 * 7), ('reserved', U32), ('counts', U64 * 3), ('retained', U64 * 3)]
                + [(name, I64 * METADATA_DIMENSIONS) for name in ('shape', 'strides', 'storage')] + [('result', View)])


class MetadataRecord(Versioned):
    _fields_ = (_fields('version size', U32) + _fields('sequence public_entry parent_token thread')
                + _fields('device ready', I32) + [('capture_complete', U64)]
                + [(name, MetadataView) for name in ('input', 'quant', 'scale')])


class MetadataState(Versioned):
    _fields_ = _fields('version size', U32) + _fields('snapshot_size record_size dimensions capacity total retained overflow storage_bytes')


METADATA_ABI_TYPES = dict(snapshot=MetadataView, record=MetadataRecord, state=MetadataState)


def read_metadata_state(function, expected_sizes):
    """Copy only the native log header; never query a descriptor or device."""
    value = MetadataState()
    code = int(function(C.byref(value)))
    sizes = dict(snapshot=value.snapshot_size, record=value.record_size, state=value.size)
    if (code or value.version != 1 or sizes != {name: C.sizeof(kind) for name, kind in METADATA_ABI_TYPES.items()}
            or sizes != expected_sizes or value.dimensions != METADATA_DIMENSIONS or value.capacity != METADATA_CAPACITY
            or value.retained != min(value.total, value.capacity) or value.overflow != value.total - value.retained
            or value.storage_bytes != value.capacity * value.record_size):
        raise UnsupportedCall('QUANT_GROUP_METADATA_DIAGNOSTIC_LAYOUT_OR_COUNTS_CHANGED')
    return snapshot(value)


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


def inspect_bundle(bundle,approved_build_sha256):
    """Aggregate inert file/source/SDK/process prerequisites before any CDLL."""
    root=Path(bundle).resolve()
    def digest(path):
        try:return _sha(path)
        except OSError:return None
    checks=[]
    def check(field,expected,actual,code):
        checks.append(expectation('deployment_identity','quant_bundle.'+field,expected,actual,code))
    manifest_path=root/'build.json'
    check('approved_manifest_sha256',approved_build_sha256,digest(manifest_path),'QUANT_GROUP_APPROVED_MANIFEST_CHANGED')
    check('approved_manifest_sha256_valid',True,bool(isinstance(approved_build_sha256,str) and
        re.fullmatch('[0-9a-f]{64}',approved_build_sha256)),'QUANT_GROUP_APPROVED_MANIFEST_CHANGED')
    try:manifest=json.loads(manifest_path.read_bytes())
    except (OSError,ValueError):manifest={}
    if type(manifest) is not dict:manifest={}
    for field,expected in dict(schema='demoncat.quant.group_deployment.v1',cann='8.5.1',actual_npu_calls=0,
            legacy_c1_library_present=False,library='runtime/vendor/op_api/lib/libcust_opapi.so').items():
        check(field,expected,manifest.get(field),'QUANT_GROUP_DEPLOYMENT_IDENTITY_INCOMPLETE')
    files=manifest.get('files',{})
    valid_files=type(files) is dict and 1<=len(files)<=1024 and all(type(k) is str and type(v) is str for k,v in files.items())
    check('file_manifest',True,valid_files,'QUANT_GROUP_DEPLOYMENT_IDENTITY_INCOMPLETE')
    if not valid_files:files={}
    check('library_in_file_manifest',True,manifest.get('library') in files,'QUANT_GROUP_DEPLOYMENT_IDENTITY_INCOMPLETE')
    for relative,expected in files.items():
        path=(root/relative).resolve()
        actual=digest(path) if path.is_relative_to(root) else 'OUTSIDE_BUNDLE'
        if actual!=expected:check('files.'+relative,expected,actual,'QUANT_GROUP_BUNDLE_FILE_CHANGED:'+relative)
    sources=manifest.get('sources',{})
    if type(sources) is not dict:sources={}
    for name in PRODUCTION_SOURCES:
        expected=digest(Path(__file__).parent/'native_adapters'/name)
        check('source.'+name,dict(source=expected,file=expected),
            dict(source=sources.get('fault_injection/native_adapters/'+name),file=files.get(name)),
            'QUANT_GROUP_SOURCE_DOES_NOT_MATCH_INSTALLED_BINDING:'+name)
    sdk=manifest.get('sdk_identity',{})
    if type(sdk) is not dict:sdk={}
    real=Path(os.environ.get('DCAT_REAL_OPAPI',''))
    check('real_opapi_absolute_file',True,real.is_absolute() and real.is_file(),
        'QUANT_GROUP_REAL_OPAPI_OR_PROCESS_MODE_CHANGED')
    check('real_opapi',sdk.get('real_opapi'),str(real) if real.is_absolute() else None,
        'QUANT_GROUP_REAL_OPAPI_OR_PROCESS_MODE_CHANGED')
    check('real_opapi_sha256',sdk.get('real_opapi_sha256'),digest(real) if real.is_absolute() else None,
        'QUANT_GROUP_REAL_OPAPI_OR_PROCESS_MODE_CHANGED')
    for field,pattern in [('actual_image_id',r'sha256:[0-9a-f]{64}'),('source_commit',r'[0-9a-f]{40}')]:
        value=manifest.get(field)
        check(field+'_valid',True,isinstance(value,str) and re.fullmatch(pattern,value) is not None,
            'QUANT_GROUP_DEPLOYMENT_IDENTITY_INCOMPLETE')
    check('sdk_abi_sizes',True,type(sdk.get('abi_sizes')) is dict and type(sdk.get('metadata_abi_sizes')) is dict,
        'QUANT_GROUP_SDK_ABI_MANIFEST_REQUIRED')
    for key,expected in [('DCAT_DQ_GROUP_V1_ALLOWED','1'),('VLLM_SERVER_DEV_MODE','1')]:
        check(key,expected,os.environ.get(key),'QUANT_GROUP_REAL_OPAPI_OR_PROCESS_MODE_CHANGED')
    check('legacy_v2_enabled',False,os.environ.get('DCAT_DQ_V2_ENABLE')=='1','QUANT_GROUP_REAL_OPAPI_OR_PROCESS_MODE_CHANGED')
    return root,manifest,require_assessment(checks,UnsupportedCall)


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
        root,manifest,report = inspect_bundle(bundle,approved_build_sha256)
        sdk=manifest['sdk_identity']
        path = (root / manifest['library']).resolve(strict=True)
        try:library = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        except OSError as error:
            require_identity(False,'quant_bundle','load','load approved DSO',str(error),
                'QUANT_GROUP_NATIVE_LOAD_FAILED',UnsupportedCall)
        identity = dict(path=str(path), library_sha256=_sha(path), build_sha256=approved_build_sha256,
                        actual_image_id=manifest['actual_image_id'], source_commit=manifest['source_commit'],
                        sdk_abi_sizes=sdk['abi_sizes'], sdk_metadata_abi_sizes=sdk['metadata_abi_sizes'], actual_npu_acceptance=False)
        result = cls(library, identity, _token=_VERIFIED_LOAD)
        result._verify_mapped()
        result.identity_prerequisites=report
        return result

    def _bind(self):
        pointer = C.POINTER
        arguments = dict(abi=[pointer(AbiSizes)], state=[pointer(RuntimeState)], raw=[U64, pointer(RawRecord)],
            metadata_state=[pointer(MetadataState)], metadata=[U64, pointer(MetadataRecord)], metadata_parent=[U64, pointer(MetadataRecord)],
            install=[pointer(Config), Admission, Completed], public_begin=[pointer(Config), pointer(PublicInput), pointer(U64)],
            public_end=[pointer(Config), U64, pointer(View), pointer(View), C.c_int, pointer(Completion)],
            **{'pass': [pointer(Config), pointer(Call)], 'continue': [pointer(Lease)]},
            seal=[pointer(Config), pointer(Key), U64, U64, pointer(Key)], stop=[pointer(Config), pointer(Key)],
            drain=[pointer(Config), pointer(Key)], cancel_empty=[pointer(Config)],
            recover=[pointer(Config)], remove=[pointer(Config), Admission, Completed])
        missing=[name for name in arguments if not hasattr(self.lib,'dcat_dqg_'+name+'_v1')]
        require_identity(not missing,'quant_bundle','required_symbols',list(arguments),{'missing':missing},
            'QUANT_GROUP_REQUIRED_SYMBOLS_MISSING',UnsupportedCall)
        for name, argtypes in arguments.items():
            function = getattr(self.lib, 'dcat_dqg_' + name + '_v1')
            function.argtypes, function.restype = argtypes, C.c_int
        sizes = AbiSizes()
        self._checked('abi', C.byref(sizes))
        actual = {name: getattr(sizes, name) for name in ABI_TYPES}
        expected={name:C.sizeof(kind) for name,kind in ABI_TYPES.items()}
        require_identity(actual==expected and actual==self.identity['sdk_abi_sizes'],
            'quant_bundle','sdk_dso_ctypes_layout',dict(ctypes=expected,sdk=self.identity['sdk_abi_sizes']),
            actual,'QUANT_GROUP_ACTUAL_DSO_SDK_CTYPES_LAYOUT_DIFFERS',UnsupportedCall)
        read_metadata_state(self.lib.dcat_dqg_metadata_state_v1, self.identity['sdk_metadata_abi_sizes'])

    def _verify_mapped(self):
        rows = isolation.require_mode('group')
        require_identity(len(rows)==1,'quant_bundle','mapped_dso_count',1,len(rows),
            'EXACTLY_ONE_APPROVED_GROUP_DSO_MAPPING_REQUIRED',UnsupportedCall)
        row = rows[0]
        require_identity(row['path']==self.identity['path'] and row['sha256']==self.identity['library_sha256'],
            'quant_bundle','loaded_mapping',dict(path=self.identity['path'],sha256=self.identity['library_sha256']),
            dict(path=row['path'],sha256=row['sha256']),'QUANT_GROUP_ACTUAL_MAPPED_SOURCE_CHANGED',UnsupportedCall)
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
            value['metadata_diagnostics'] = read_metadata_state(self.lib.dcat_dqg_metadata_state_v1,
                                                                self.identity['sdk_metadata_abi_sizes'])
            value['binding_diagnostics'] = dict(callback_error_count=self.callback_error_count,
                callback_error_overflow=self.callback_error_overflow, errors=self.callback_errors)
            return value

    def raw(self, sequence):
        return self.raw_range(sequence, sequence)[0]

    def raw_range(self, first, last):
        if (type(first) is not int or type(last) is not int or not 1 <= first <= last <= 32768):
            raise UnsupportedCall('BOUNDED_ACTUAL_GROUP_CENSUS_RANGE_REQUIRED')
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            rows = []
            for sequence in range(first, last + 1):
                row = RawRecord()
                self._checked('raw', sequence, C.byref(row))
                value = snapshot(row)
                if row.kind == 1:
                    captured = MetadataRecord()
                    self._checked('metadata_parent', sequence, C.byref(captured))
                    raw = snapshot(captured)
                    if (captured.version != 1 or captured.size != C.sizeof(MetadataRecord)
                            or captured.parent_token != sequence or captured.capture_complete != 1
                            or captured.thread != row.parent.thread or captured.device != row.parent.device
                            or any(raw[name]['result'] != value['parent'][name] for name in ('input', 'quant', 'scale'))):
                        raise UnsupportedCall('QUANT_GROUP_METADATA_DIAGNOSTIC_PARENT_CHANGED')
                    value['metadata_diagnostic'] = raw
                rows.append(value)
            return rows

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

    def cancel_empty(self):
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            status = self.lib.dcat_dqg_cancel_empty_v1(C.byref(self.config))
            if status not in (0, -10):
                raise UnsupportedCall('QUANT_GROUP_EMPTY_CANCEL_IDENTITY_OR_STATE_FAILURE:' + str(status))
            return status == 0

    def remove(self):
        with isolation.managed('group', session_runtime=self):
            self._verify_mapped()
            self._checked('remove', C.byref(self.config), self._admission, self._completed)
            _LIVE_OWNERS.pop(self._registry_key, None)
            self.config = self._admission = self._completed = self._held_input = None
