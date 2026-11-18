"""Experimental read-only Fault39 refusal observation; never a C3/E5 authority.

No Core/provider methods are replaced. The Native copy reads lease.original,
whereas the two counterfactual controls explicitly use the existing fresh read.
Attach only to a disposable, already initialized Worker and check errors after
each real request. Full BF16 bytes are transient and are never serialized here.
"""
from __future__ import annotations

import ctypes
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import marshal
from pathlib import Path
import struct
import sys
import threading
import time
import weakref


class DiagnosticRefused(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise DiagnosticRefused(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


class OriginalInfo(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint32), ('little_endian_bf16', ctypes.c_uint32),
        ('handle', ctypes.c_uint64), ('owner', ctypes.c_uint64), ('generation', ctypes.c_uint64),
        ('tensor_address', ctypes.c_uint64), ('elements', ctypes.c_uint64),
        ('span_bytes', ctypes.c_uint64), ('device', ctypes.c_int32), ('applied', ctypes.c_uint32)]


def summarize_bf16_le(payload):
    """Bit classification only, preserving signed NaNs and every payload bit."""
    require(type(payload) is bytes and 0 < len(payload) <= 1024*1024 and len(payload) % 2 == 0,
            'EXACT_NONEMPTY_BOUNDED_BF16_SPAN_REQUIRED')
    counts = dict(finite=0, positive_infinity=0, negative_infinity=0, quiet_nan=0, signaling_nan=0)
    first, first_by_class = None, {}
    for index, (bits,) in enumerate(struct.iter_unpack('<H', payload)):
        if bits & 0x7f80 != 0x7f80:
            label = 'finite'
        elif bits & 0x7f == 0:
            label = 'negative_infinity' if bits & 0x8000 else 'positive_infinity'
        else:
            label = 'quiet_nan' if bits & 0x40 else 'signaling_nan'
        counts[label] += 1
        if label != 'finite' and label not in first_by_class:
            item = dict(index=index, bits_hex=f'0x{bits:04x}', classification=label,
                        sign_bit=int(bool(bits & 0x8000)))
            first_by_class[label] = item
            if first is None:
                first = item
    return dict(representation='ALL_LOGICAL_SPAN_BF16_BITS_LITTLE_ENDIAN', bytes=len(payload),
        elements=len(payload)//2, sha256=hashlib.sha256(payload).hexdigest(), counts=counts,
        nonfinite_elements=len(payload)//2-counts['finite'], first_nonfinite=first,
        first_by_class=first_by_class, full_span=True, raw_payload_saved=False)


def copy_lease_original(lease, *, explicit_cpu_contract=False):
    """Copy and verify the exact same Native lease, with no device read/write."""
    from fault_injection.deployments.postwrite import _NativeLease, NativePostwriteLibrary
    require(type(lease) is _NativeLease, 'EXACT_NATIVE_LEASE_REQUIRED')
    require(type(explicit_cpu_contract) is bool, 'TYPED_CPU_CONTRACT_REQUIRED')
    lease._check()
    library = lease.library
    if not explicit_cpu_contract:
        require(type(library) is NativePostwriteLibrary and isinstance(library.lib, ctypes.CDLL),
                'ACTUAL_NATIVE_LIBRARY_REQUIRED')
        require(lease.tensor.device.type == 'npu', 'ACTUAL_NPU_LEASE_REQUIRED')
    function = getattr(library.lib, 'dcat_pw_copy_original_v1', None)
    require(function is not None, 'FRESH_SAME_LEASE_ORIGINAL_EXPORT_REQUIRED')
    if not explicit_cpu_contract:
        function.argtypes = [ctypes.c_uint64]*3 + [ctypes.c_void_p, ctypes.c_uint64,
                                                  ctypes.POINTER(OriginalInfo)]
        function.restype = ctypes.c_int
    before = lease.receipt()
    opening = before.get('open') or {}
    require(opening.get('returned') is True and opening.get('return_code') == 0,
            'ACTUAL_OPEN_SUCCESS_REQUIRED')
    size = lease.elements*2
    require(lease.tensor.element_size() == 2 and str(lease.tensor.dtype) == 'torch.bfloat16'
            and 0 < size <= 1024*1024 and opening.get('bytes') == size,
            'ACTUAL_BF16_OPEN_SPAN_REQUIRED')
    expected = dict(handle=lease.handle.value, owner=lease.owner, generation=lease.generation,
        tensor_address=lease.tensor.data_ptr()+lease.element_offset*2,
        elements=lease.elements, span_bytes=size, device=lease.tensor.device.index)
    # CPU descriptors use None as the actual CPU device; their ABI double uses -1.
    if explicit_cpu_contract and expected['device'] is None:
        expected['device'] = -1
    require(opening.get('data_ptr') == expected['tensor_address']
        and opening.get('device') == lease.tensor.device.index
        and all(opening.get(k) == expected[k] for k in ('handle', 'owner', 'generation')),
        'LEASE_OPEN_IDENTITY_CHANGED')
    payload = (ctypes.c_uint8*size)()
    info = OriginalInfo()
    started = time.monotonic_ns()
    code = function(lease.handle, lease.owner, lease.generation, payload, size, ctypes.byref(info))
    ended = time.monotonic_ns()
    require(type(code) is int and code == 0, 'SAME_LEASE_ORIGINAL_COPY_'+str(code))
    lease._check()
    require(lease.receipt() == before, 'DIAGNOSTIC_CHANGED_EXISTING_LEASE_RECEIPT')
    actual = {k: getattr(info, k) for k in expected}
    require(info.version == 1 and info.little_endian_bf16 == 1 and actual == expected,
            'NATIVE_ORIGINAL_METADATA_MISMATCH')
    apps = before['applications']
    require(apps and apps[-1].get('returned') is True and apps[-1]['return_code'] in (0, 4, 5),
            'DIAGNOSTIC_REQUIRES_ACTUAL_COMPLETED_APPLY')
    require(info.applied == int(apps[-1]['return_code'] == 0), 'NATIVE_APPLIED_STATE_MISMATCH')
    return dict(origin='SAME_NATIVE_LEASE_ORIGINAL', read_point='AFTER_ACTUAL_APPLY_RETURN_BEFORE_WRAPPER_RAISE',
        interval_monotonic_ns=[started, ended], native=dict(**actual, applied=bool(info.applied), copy_return_code=code),
        open_event=deepcopy(opening), apply_event=deepcopy(apps[-1]),
        span=summarize_bf16_le(bytes(payload)), explicit_cpu_contract=explicit_cpu_contract,
        additional_acl_calls=0, tensor_object_identity=id(lease.tensor),
        tensor_geometry=deepcopy(lease.geometry), element_offset=lease.element_offset)


@dataclass(frozen=True)
class DiagnosticPlan:
    predecessor_module: str = 'model.layers.0.mlp.gate_up_proj'
    refused_module: str = 'model.layers.1.mlp.gate_up_proj'
    operator: str = 'vllm.unquantized_gemm.default'
    call_ordinal: int = 0
    refused_shape: tuple = (14, 6144)
    expected_all_calls: int = 113
    expected_eligible: int = 28
    maximum_group_requests: int = 42  # 6 prefix + 31 C2 + 2 through C3 F + <=2 recovery + health.
    maximum_observations: int = 5   # healthy/fresh + single/predecessor+fresh + cumulative/two originals.


PIN_FILES = (
    'fault_injection/deployments/postwrite.py',
    'fault_injection/deployments/postwrite_coverage.py',
    'fault_injection/deployments/postwrite_boundaries.py',
    'fault_injection/deployments/postwrite_inventory.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving39_output_group.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault39_output_group.py',
    'fault_injection/native_adapters/postwrite/postwrite.cpp',
    'fault_injection/native_adapters/postwrite/postwrite.h',
    'experiments/worker_core_v11_20261118/output39_refusal_diagnostic.py',
)


class RefusalObserver:
    """One-thread trace observer of existing exact call frames, not a wrapper.

    Install after public enable on the actual group provider. No layer selector
    is passed to the injector; the two fixed paths only bound this diagnostic.
    Consumer rows are copied at close, after actual dispatch has recorded them.
    """
    def __init__(self, core_provider, source_root, source_pins, *, plan=DiagnosticPlan(),
                 explicit_cpu_contract=False):
        from fault_injection.deployments.postwrite import _NativeLease
        from fault_injection.deployments.postwrite import NativePostwriteLibrary
        from fault_injection.deployments.postwrite_coverage import ProjectionCoverage
        from demoncat.runtime.vllm_ascend.worker_extension.providers.fault39_output_group import Fault39OutputGroupProvider
        require(type(core_provider) is Fault39OutputGroupProvider, 'ACTUAL_GROUP_PROVIDER_REQUIRED')
        require(bool(core_provider.binding.simulation) == explicit_cpu_contract, 'CPU_SCOPE_MISMATCH')
        require(sys.gettrace() is None, 'FOREIGN_TRACE_OBSERVER_PRESENT')
        root = Path(source_root).resolve()
        for name in PIN_FILES:
            path = root/name
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == source_pins.get(name),
                    'DIAGNOSTIC_SOURCE_NOT_PINNED:'+name)
        self.codes = dict(apply=_NativeLease._native_call.__code__, projection=ProjectionCoverage.projection.__code__,
                          end=ProjectionCoverage.end_observed_forward.__code__)
        self.code_identity = {name:dict(qualified_name=code.co_qualname, file=code.co_filename,
            first_line=code.co_firstlineno, marshalled_code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest())
            for name,code in self.codes.items()}
        for code, relative in ((self.codes['apply'], PIN_FILES[0]), (self.codes['projection'], PIN_FILES[1])):
            require(Path(code.co_filename).resolve() == root/relative, 'LOADED_DIAGNOSTIC_BOUNDARY_NOT_PINNED')
        require(Path(core_provider.begin_actual.__func__.__code__.co_filename).resolve() == root/PIN_FILES[5],
                'LOADED_PUBLIC_PROVIDER_NOT_PINNED')
        transport = core_provider.p.transport
        if explicit_cpu_contract:
            self.native_identity = dict(scope='EXPLICIT_CPU_ABI_DOUBLE_NOT_NATIVE_BUILD', sha256=None)
        else:
            require(type(transport) is NativePostwriteLibrary, 'EXACT_LOADED_NATIVE_TRANSPORT_REQUIRED')
            native_path = Path(transport.path)
            require(hashlib.sha256(native_path.read_bytes()).hexdigest() == transport.sha256,
                    'LOADED_NATIVE_FILE_CHANGED')
            transport._verify_loaded_file(native_path)
            copier = getattr(transport.lib, 'dcat_pw_copy_original_v1', None)
            require(copier is not None, 'FRESH_SAME_LEASE_ORIGINAL_EXPORT_REQUIRED')
            address = ctypes.cast(copier, ctypes.c_void_p).value
            mappings = []
            for line in Path('/proc/self/maps').read_text().splitlines():
                fields = line.split(None, 5)
                begin,end = (int(value,16) for value in fields[0].split('-'))
                if begin <= address < end:
                    mappings.append(fields)
            require(len(mappings) == 1 and len(mappings[0]) == 6 and 'x' in mappings[0][1]
                    and mappings[0][5] == str(native_path), 'ORIGINAL_COPY_SYMBOL_NOT_IN_PINNED_DSO')
            self.native_identity = dict(scope='ACTUAL_LOADED_NATIVE_DSO', sha256=transport.sha256,
                copy_symbol_address=address,
                export='dcat_pw_copy_original_v1', build_provenance_review_required=True)
        self.core, self.plan, self.cpu = core_provider, plan, explicit_cpu_contract
        self.source_pins = {name: source_pins[name] for name in PIN_FILES}
        self.worker_identity = deepcopy(core_provider.identity)
        self.thread, self.callback = threading.get_ident(), self._trace
        self.records, self.errors, self.pending = [], [], []
        self.single_request = None
        self.healthy_request = None
        self.cumulative_request = None
        self.bound_keys, self.discovery = None, None
        self.installed = False
        # The six original discovery/C1 requests must finish first. Refuse a
        # changed denominator before the first group lossless request starts.
        self._bind()

    def _bind(self):
        group = self.core.group
        require(group is not None and group.provider is self.core.p, 'EXACT_LIVE_GROUP_REQUIRED')
        require(self.core.identity == self.worker_identity, 'DIAGNOSTIC_WORKER_LIFETIME_CHANGED')
        require(len(group.call_order) == self.plan.expected_all_calls
                and len(group.eligible) == self.plan.expected_eligible, 'FROZEN_DENOMINATOR_CHANGED')
        keys = {}
        healthy_rows = group.discovery_reports[0]['calls']
        require(len(group.discovery_reports) == 2, 'TWO_ACTUAL_HEALTHY_INVENTORIES_REQUIRED')
        for label, path in (('predecessor', self.plan.predecessor_module), ('refused', self.plan.refused_module)):
            matches = [tuple(row['key']) for row in healthy_rows if tuple(row['key']) in group.eligible
                       and row['module_path'] == path
                       and tuple(row['key'])[1:] == (self.plan.operator, self.plan.call_ordinal)]
            require(len(matches) == 1, 'FROZEN_TARGET_NOT_UNIQUELY_OBSERVED:'+label)
            keys[label] = matches[0]
        eligible_order = [key for key in group.call_order if key in group.eligible]
        require(eligible_order[:2] == [keys['predecessor'], keys['refused']],
                'FROZEN_FIRST_TWO_ELIGIBLE_TARGETS_CHANGED')
        for report in group.discovery_reports:
            row, = [row for row in report['calls'] if tuple(row['key']) == keys['refused']]
            require(tuple(row['output_geometry']['shape']) == self.plan.refused_shape,
                    'FROZEN_REFUSED_GEOMETRY_CHANGED')
        if self.bound_keys is not None:
            require(keys == self.bound_keys, 'FROZEN_TARGET_OBJECTS_CHANGED')
        self.bound_keys = keys
        self.discovery = dict(all_calls=len(group.call_order), eligible=len(group.eligible),
            call_order_sha256=digest(group.call_order), targets={k:list(v) for k,v in keys.items()})
        return group

    def _context(self, frame):
        group = self._bind()
        require(frame.f_locals.get('self') is group, 'FOREIGN_COVERAGE_FRAME')
        row = frame.f_locals['row']
        candidate = row.candidate
        require(candidate is not None and row.original_calls == 1, 'ACTUAL_PRODUCER_RETURN_REQUIRED')
        require(candidate.tensor is not None and candidate.operator == row.operator,
                'CANDIDATE_PRODUCER_MISMATCH')
        current = self.core.session.current
        require(current is not None and current['actual_internal_request_id'] == group.request_id,
                'ACTUAL_REQUEST_JOIN_REQUIRED')
        require(self.core.index+1 <= self.plan.maximum_group_requests, 'BOUNDED_REQUEST_COUNT_EXCEEDED')
        return group, row, candidate, dict(phase=self.core.phase, request_id=group.request_id,
            forward_id=group.forward_id, target_key=list(row.key), module_path=row.module_path,
            operator=row.operator, module_object_identity=row.module_object_identity,
            tensor_object_identity=id(candidate.tensor), geometry=deepcopy(row.output_geometry),
            producer_proof=deepcopy(candidate.producer_proof), request_mapping=deepcopy(current),
            native_owner=self.core.p.owner, native_generation=self.core.p.generation)

    def _append(self, record, candidate):
        require(len(self.records) < self.plan.maximum_observations, 'DIAGNOSTIC_OBSERVATION_BOUND_EXCEEDED')
        record['sequence'] = len(self.records)+1
        self.records.append(record)
        self.pending.append((record, weakref.ref(candidate)))

    def _fresh(self, frame):
        # Only actual selected phase/row frames; this never opens a new lease.
        row = frame.f_locals['row']
        if row.module_path != self.plan.refused_module or self.core.phase not in ('c2_lossless', 'c2_fault'):
            return
        group, row, candidate, context = self._context(frame)
        if row.key != self.bound_keys['refused']:
            return
        require(tuple(candidate.tensor.shape) == self.plan.refused_shape, 'FROZEN_REFUSED_GEOMETRY_CHANGED')
        phase = context['phase']
        arm = None
        if phase == 'c2_lossless' and self.healthy_request is None:
            self.healthy_request, arm = group.request_id, 'healthy'
        elif phase == 'c2_fault' and group.request_id == self.single_request:
            arm = 'single_predecessor_fault'
        if arm is None:
            return
        started = time.monotonic_ns()
        raw = self.core.p.transport.snapshot(candidate.tensor)
        ended = time.monotonic_ns()
        # ACL fresh-copy representation is native endian; supported device is LE.
        require(sys.byteorder == 'little', 'FRESH_BF16_ENDIAN_NOT_ADAPTED')
        self._append(dict(arm=arm, target='refused', origin='ACTUAL_FRESH_READ_SPAN',
            read_point='ACTUAL_PROJECTION_RETURN_BEFORE_COVERAGE_ADMISSION',
            interval_monotonic_ns=[started, ended], binding=context,
            span=summarize_bf16_le(raw), additional_acl_calls='EXISTING_READ_SPAN_SYNC_AND_D2H',
            explicit_cpu_contract=self.cpu), candidate)

    def _applied(self, frame, return_code):
        if frame.f_locals.get('operation') != 'apply':
            return
        if frame.f_locals.get('event', {}).get('returned') is not True:
            return  # An existing Native/Python exception must keep its own type.
        projection = frame.f_back
        while projection is not None and projection.f_code is not self.codes['projection']:
            projection = projection.f_back
        if projection is None or projection.f_locals.get('self') is not self.core.group:
            return  # Prefix C1 is deliberately not the additional diagnostic.
        group, row, candidate, context = self._context(projection)
        phase = context['phase']
        label = next((label for label,key in self.bound_keys.items() if key == row.key), None)
        if label is None or phase not in ('c2_fault', 'c3_fault'):
            return
        if phase == 'c2_fault' and label != 'predecessor':
            return
        if phase == 'c3_fault':
            if self.cumulative_request is None:
                self.cumulative_request = group.request_id
            require(self.cumulative_request == group.request_id, 'ONE_CUMULATIVE_DIAGNOSTIC_ATTEMPT_ONLY')
        lease = frame.f_locals['self']
        require(lease.tensor is candidate.tensor and lease.library is self.core.p.transport,
                'ACTUAL_LEASE_CANDIDATE_OBJECT_JOIN_REQUIRED')
        observed = copy_lease_original(lease, explicit_cpu_contract=self.cpu)
        require(observed['apply_event']['return_code'] == return_code, 'APPLY_FRAME_RECEIPT_MISMATCH')
        if phase == 'c2_fault':
            require(self.single_request is None and return_code == 0, 'SINGLE_PREDECESSOR_F_REQUIRED')
            self.single_request = group.request_id
        observed.update(arm='single_predecessor_fault' if phase == 'c2_fault' else 'cumulative_fault',
                        target=label, binding=context)
        self._append(observed, candidate)
        if phase == 'c3_fault' and label == 'refused' and return_code != 4:
            # Explicit diagnostic stop, not a replacement Native return code.
            # The existing Core sees a failed request and owns its recovery.
            raise DiagnosticRefused('FROZEN_NATIVE_REFUSAL_NOT_REPRODUCED')

    def _trace(self, frame, event, arg):
        if not any(frame.f_code is code for code in self.codes.values()):
            return None
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False
        try:
            require(threading.get_ident() == self.thread, 'DIAGNOSTIC_THREAD_CHANGED')
            if frame.f_code is self.codes['projection'] and event == 'call':
                if frame.f_locals.get('self') is self.core.group:
                    self._fresh(frame)
            elif frame.f_code is self.codes['apply'] and event == 'return':
                self._applied(frame, arg)
            elif frame.f_code is self.codes['end'] and event == 'return':
                if frame.f_locals.get('self') is self.core.group:
                    self._finish_consumers()
        except Exception as error:
            # Never replace the actual Native refusal. Other diagnostic errors
            # fail the current request, so existing Core recovery sees failure.
            self.errors.append(dict(event=event, function=frame.f_code.co_name,
                                    error=repr(error), records=len(self.records)))
            if not (frame.f_code is self.codes['apply'] and event == 'return'
                    and type(arg) is int and arg != 0):
                require(sys.gettrace() is self.callback, 'DIAGNOSTIC_TRACE_OWNERSHIP_CHANGED')
                sys.settrace(None)
                self.installed = False
                raise
        return self.callback

    def __enter__(self):
        require(not self.installed and sys.gettrace() is None, 'DIAGNOSTIC_TRACE_ALREADY_OWNED')
        sys.settrace(self.callback)
        self.installed = True
        return self

    def close(self):
        if self.installed:
            require(sys.gettrace() is self.callback and threading.get_ident() == self.thread,
                    'DIAGNOSTIC_TRACE_OWNERSHIP_CHANGED')
            sys.settrace(None)
            self.installed = False
        self._finish_consumers()

    def _finish_consumers(self):
        for record, reference in self.pending:
            candidate = reference()
            if candidate is None:
                self.errors.append(dict(error='ACTUAL_CANDIDATE_LIFETIME_ENDED_BEFORE_OBSERVATION',
                                        sequence=record['sequence']))
                record['consumer_status'] = 'UNOBSERVABLE_OBJECT_GONE'
                continue
            observations = deepcopy(candidate.consumer_observations)
            record['consumer_observations'] = observations
            if observations:
                record['consumer_status'] = 'ACTUAL_VALUE_CONSUMER_RETURN_RECORDED'
            elif record.get('apply_event', {}).get('return_code') == 4:
                record['consumer_status'] = 'NOT_REACHED_AFTER_ACTUAL_APPLY_REFUSAL'
            else:
                record['consumer_status'] = 'NOT_OBSERVED'
        self.pending.clear()  # Only weak references; no allocation lifetime extension.

    def __exit__(self, *unused):
        self.close()
        return False

    def report(self):
        require(not self.installed and not self.pending, 'CLOSE_DIAGNOSTIC_BEFORE_REPORT')
        return dict(schema='demoncat.output39.refusal_diagnostic.v1', source_pins=self.source_pins,
            plan=deepcopy(self.plan.__dict__), plan_sha256=digest(self.plan.__dict__),
            worker_identity=self.worker_identity, native_identity=self.native_identity,
            loaded_code_identity=self.code_identity,
            discovery=self.discovery, records=deepcopy(self.records), errors=deepcopy(self.errors),
            raw_tensors_saved=False, explicit_cpu_contract=self.cpu,
            c2=False, c3=False, e5=False, qualification_granted=False,
            causal_conclusion='REQUIRES_INDEPENDENT_ORIGINAL_REQUEST_AND_RECOVERY_REVIEW')
