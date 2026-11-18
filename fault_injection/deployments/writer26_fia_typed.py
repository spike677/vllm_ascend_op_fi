"""FIA-only complete descriptor evidence and stock-to-owned bridge proof.

Format 47 remains NCL. Public ND is accepted only for a complete current native
formal with the independently recorded GWS path, never by a format allowlist.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading

from .writer26_inventory import digest, require


ABI = dict(version=3, inventory_version=2, descriptor=280, record=656, state=104, lease=912, snapshots=2048)


class Descriptor(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('version', 'size', 'complete', 'contiguous')]
    _fields_ += [(n, C.c_int32) for n in ('dtype', 'format')]
    _fields_ += [(n, C.c_uint32) for n in ('view_rank', 'storage_rank', 'stride_rank', 'reserved')]
    _fields_ += [(n, C.c_uint64) for n in ('descriptor', 'raw', 'view_bytes', 'storage_bytes', 'element_bytes')]
    _fields_ += [('offset', C.c_int64)] + [(n, C.c_int64 * 8) for n in ('view', 'storage', 'strides')]


class Record(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('version', 'size', 'phase', 'api')]
    _fields_ += [(n, C.c_int32) for n in ('device', 'status')]
    _fields_ += [(n, C.c_uint64) for n in ('sequence', 'event_sequence', 'call', 'thread', 'executor', 'registration')]
    _fields_ += [(n, C.c_uint32) for n in ('layout_length', 'reserved')]
    _fields_ += [('layout', C.c_uint8 * 16), ('original', Descriptor), ('effective', Descriptor)]


class TraceState(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)]
    _fields_ += [(n, C.c_uint64) for n in ('capacity', 'total', 'stored', 'overflow', 'errors', 'exceptions',
                                         'before_gws', 'after_gws', 'before_execute', 'preparing', 'prepared', 'executing')]


class Snapshots(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)]
    _fields_ += [(n, C.c_uint64) for n in ('token', 'owner_hi', 'owner_lo', 'generation', 'descriptor', 'created',
                                        'released', 'native_return_success', 'exceptions')]
    _fields_ += [(n, Record) for n in ('before', 'after', 'before_execute')]


class Abi(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)]
    _fields_ += [(n, C.c_uint64) for n in ('descriptor', 'record', 'state', 'lease', 'snapshots')]


def profile(value):
    expected = dict(schema='demoncat.writer26.fia_ncl_typed.v3', abi=ABI, native_format=47, public_format=2,
        dtype='bfloat16', dimensions=3, view_storage_equal=True, offset=0, contiguous=True,
        borrowed_original_read_scope='ACTUAL_GWS_STACK_ONLY', effective_descriptor='GROUP_OWNED_ALL_SELECTED_MODES',
        stock_to_owned_reference_full_byte_comparison_required=True, old_reference_reusable=False,
        before_execute_complete_typed_readback_required=True, native_qualified=False)
    require(value == expected, 'FIA_TYPED_FRESH_BUILD_PROFILE_REQUIRED')


def ncl(value):
    require(value['version'] == 2 and value['size'] == 280 and value['complete'] == value['contiguous'] == 1
        and value['dtype'] == 27 and value['format'] == 47 and value['reserved'] == value['offset'] == 0
        and value['view_rank'] == value['storage_rank'] == value['stride_rank'] == 3 and value['element_bytes'] == 2
        and 0 < value['descriptor'] < 2**64 and 0 < value['raw'] < 2**64,
        'FIA_COMPLETE_BF16_NCL_DESCRIPTOR_REQUIRED')
    step = 1
    for index in (2, 1, 0):
        require(value['view'][index] == value['storage'][index] > 0 and value['strides'][index] == step,
                'FIA_NCL_VIEW_STORAGE_STRIDE_CHANGED')
        step *= value['view'][index]
    require(all(len(value[key]) == 8 and not any(value[key][3:]) for key in ('view', 'storage', 'strides'))
        and step * 2 == value['view_bytes'] == value['storage_bytes'] <= 8 * 1024 * 1024
        and value['raw'] + value['storage_bytes'] <= 2**64 - 1, 'FIA_NCL_FULL_STORAGE_SPAN_REQUIRED')


def path(row, phase=None):
    require(row['version'] == 2 and row['size'] == 656 and row['api'] == 1 and row['reserved'] == row['status'] == 0
        and row['device'] >= 0 and row['call'] > 0 and row['thread'] > 0 and row['event_sequence'] > 0
        and row['sequence'] > 0 and (phase is None or row['phase'] == phase)
        and len(row['layout']) == 16 and 0 < row['layout_length'] < 16
        and all(65 <= c <= 90 for c in row['layout'][:row['layout_length']])
        and not any(row['layout'][row['layout_length']:]), 'FIA_ACTUAL_TYPED_GWS_PATH_REQUIRED')


def geometry(row):
    path(row)
    ncl(row['original'])
    return dict(api=row['api'], layout=list(row['layout']), layout_length=row['layout_length'],
        descriptor={key: value for key, value in row['original'].items() if key not in ('descriptor', 'raw')})


def same_path(left, right):
    return all(left[key] == right[key] for key in ('api', 'device', 'call', 'thread', 'layout_length', 'layout'))


def bind_before(row, call, event):
    path(row, 1)
    ncl(row['original'])
    original = row['original']
    require(row['effective'] == original and row['executor'] == row['registration'] == 0
        and row['event_sequence'] == event['sequence'] and row['call'] == call['independent_call'] == event['call']
        and row['thread'] == call['thread'] == event['thread'] and row['device'] == call['device'] == event['device']
        and call['mechanism'] == 2 and call['dimensions'] == event['dimensions'] == 3
        and call['format'] == event['format'] == 47 and call['dtype'] == event['dtype'] == 27
        and original['raw'] == call['output'] == event['output']
        and original['view_bytes'] == call['output_bytes'] == event['output_bytes']
        and original['view'][:3] == call['shape'][:3] == event['shape'][:3]
        and not any(call['shape'][3:]) and original['strides'] == event['strides'],
        'FIA_TYPED_BEFORE_NOT_CURRENT_NATIVE_CALL')


def completed_records(rows, call, events, *, selected):
    expected = [1, 2, 3] if selected else [1, 2]
    require([row['phase'] for row in rows] == expected, 'COMPLETE_CURRENT_TYPED_FIA_RECORDS_REQUIRED')
    before, after = rows[:2]
    bind_before(before, call, events[0])
    for row, event, phase in zip(rows[1:], events[1:], expected[1:]):
        path(row, phase)
        ncl(row['effective'])
        require(same_path(before, row) and row['original'] == before['original']
            and row['event_sequence'] == event['sequence'] and row['executor'] == event['executor'] != 0
            and row['effective']['raw'] == event['output']
            and row['effective']['view_bytes'] == event['output_bytes']
            and row['effective']['view'] == event['shape'] and row['effective']['strides'] == event['strides']
            and geometry(row) == geometry(before), 'FIA_TYPED_ORIGINAL_OR_EFFECTIVE_CHANGED')
        require({key: value for key, value in row['effective'].items() if key not in ('descriptor', 'raw')}
            == geometry(before)['descriptor'], 'FIA_EFFECTIVE_FULL_GEOMETRY_CHANGED')
        require(row['sequence'] > before['sequence'], 'FIA_TYPED_RECORD_ORDER_CHANGED')
    require(after['registration'] == 0, 'FIA_GWS_REGISTRATION_MUST_COME_FROM_EXECUTE')
    if selected:
        current = rows[2]
        require(current['sequence'] > after['sequence'] and current['effective'] == after['effective']
            and current['registration'] == events[2]['registration'] != 0,
            'FIA_OWNED_EFFECTIVE_BEFORE_EXECUTE_CHANGED')
    else:
        require(after['effective'] == before['original'], 'UNSELECTED_FIA_DESCRIPTOR_WAS_REPLACED')
    return before


class NativeTypedFia:
    """Read only exports plus one exact typed admission; no new Native load."""
    def __init__(self, group, root):
        from .writer26_group import Lease, Config, packet
        profile(group.build.get('fia_typed_profile'))
        class TypedLease(C.Structure):
            _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('lease', Lease), ('original_before', Record)]
        self.group, self.Lease, self.packet = group, TypedLease, packet
        require([C.sizeof(value) for value in (Descriptor, Record, TraceState, TypedLease, Snapshots, Abi)]
                == [280, 656, 104, 912, 2048, 48], 'FIA_TYPED_CTYPES_LAYOUT_CHANGED')
        self.abi_fn = group.lib.dcat26_group_fia_abi_v3
        self.abi_fn.argtypes, self.abi_fn.restype = [C.POINTER(Abi)], C.c_int
        actual = Abi(version=3, size=C.sizeof(Abi))
        require(self.abi_fn(C.byref(actual)) == 0 and packet(actual) == dict(version=3, size=48,
            **{key: ABI[key] for key in ('descriptor', 'record', 'state', 'lease', 'snapshots')}),
            'FIA_ACTUAL_SDK_ABI_CHANGED')
        self.snapshot_fn = group.lib.dcat26_group_fia_snapshots_v3
        self.snapshot_fn.argtypes, self.snapshot_fn.restype = [C.POINTER(Config), C.c_uint64, C.POINTER(Snapshots)], C.c_int
        self.continue_fn = group.lib.dcat26_group_continue_fia_v3
        self.continue_fn.argtypes, self.continue_fn.restype = [C.POINTER(TypedLease)], C.c_int
        self.trace = C.CDLL(str(Path(root) / 'libfault26_readonly_trace_abi0.so'), mode=os.RTLD_LAZY | os.RTLD_NOLOAD)
        self.state_fn = self.trace.dcat26_inventory_fia_typed_state_v2
        self.state_fn.argtypes, self.state_fn.restype = [C.POINTER(TraceState)], C.c_int
        self.read_fn = self.trace.dcat26_inventory_fia_typed_read_v2
        self.read_fn.argtypes, self.read_fn.restype = [C.c_uint64, C.POINTER(Record)], C.c_int

    def _thread(self):
        require(threading.get_native_id() == self.group.config.thread, 'FOREIGN_TYPED_FIA_NATIVE_THREAD')

    def state(self, *, idle=False):
        self._thread()
        value = TraceState(version=2, size=C.sizeof(TraceState))
        require(self.state_fn(C.byref(value)) == 0, 'FIA_TYPED_STATE_READ_FAILED')
        row = self.packet(value)
        require(row['version'] == 2 and row['size'] == 104 and row['capacity'] == 32768
            and row['total'] == row['stored'] == row['before_gws'] + row['after_gws'] + row['before_execute']
            and not any(row[key] for key in ('errors', 'exceptions', 'overflow')),
            'FIA_TYPED_OBSERVER_ERROR_OR_LOSS')
        if idle:
            require(not any(row[key] for key in ('preparing', 'prepared', 'executing')),
                    'FIA_TYPED_PREPARED_OR_EXECUTING_NOT_DRAINED')
        return row

    def read(self, index):
        value = Record(version=2, size=C.sizeof(Record))
        require(self.read_fn(index, C.byref(value)) == 0, 'FIA_TYPED_RECORD_READ_FAILED')
        row = self.packet(value)
        require(row['sequence'] == index + 1, 'FIA_TYPED_RECORD_GAP')
        return row

    def delta(self, before):
        after = self.state(idle=True)
        counts = ('total', 'stored', 'before_gws', 'after_gws', 'before_execute')
        require(all(after[key] >= before[key] for key in counts), 'FIA_TYPED_COUNTER_RESET')
        rows = [self.read(index) for index in range(before['stored'], after['stored'])]
        require(all(sum(row['phase'] == phase for row in rows) == after[key] - before[key]
            for phase, key in ((1, 'before_gws'), (2, 'after_gws'), (3, 'before_execute'))), 'FIA_TYPED_COUNTER_DETAIL_LOSS')
        return after, rows

    def snapshot(self, token):
        self._thread()
        value = Snapshots(version=3, size=C.sizeof(Snapshots))
        require(self.snapshot_fn(C.byref(self.group.config), token, C.byref(value)) == 0,
                'FIA_CURRENT_CALL_SNAPSHOT_UNAVAILABLE')
        row = self.packet(value)
        require(row['version'] == 3 and row['size'] == 2048 and row['token'] == token and not row['exceptions']
            and all(row[key] == getattr(self.group.config, key) for key in ('owner_hi', 'owner_lo', 'generation')),
            'FIA_CURRENT_CALL_SNAPSHOT_IDENTITY_CHANGED')
        return value, row

    def current_before(self, call, event):
        state = self.state()
        require(state['preparing'] == 1 and state['prepared'] == state['executing'] == 0,
                'FIA_SERIAL_CURRENT_GWS_REQUIRED')
        _, snapshot = self.snapshot(call['token'])
        row = self.read(state['stored'] - 1)
        require(snapshot['before'] == row and not any(snapshot[key] for key in
            ('descriptor', 'created', 'released', 'native_return_success')), 'FIA_CURRENT_BEFORE_SNAPSHOT_CHANGED')
        bind_before(row, call, event)
        return row

    def execute(self, lease, before):
        self._thread()
        current, row = self.snapshot(lease.expected.token)
        require(row['before'] == before, 'FIA_BEFORE_CHANGED_SINCE_PUBLIC_ADMISSION')
        proposal = self.Lease(version=3, size=C.sizeof(self.Lease), lease=lease, original_before=current.before)
        require(self.continue_fn(C.byref(proposal)) == 0, 'FIA_TYPED_NATIVE_ADMISSION_REFUSED')
        return self.packet(proposal)


class FiaStockBridge:
    """Own immutable full-byte stock samples; hashes report but do not compare."""
    def __init__(self, experiment):
        self.experiment = experiment
        self.stock, self.references = {}, set()
        self.host_bytes = 0
        self.artifacts = []

    def public_output(self, row, tensor, records, *, selected):
        import torch
        from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
        experiment = self.experiment
        target, request = row['target_key'], experiment.current
        if not selected and request['phase'] not in ('inventory_1', 'inventory_2'):
            return
        require(tensor.numel() * tensor.element_size() <= 8 * 1024 * 1024,
                'FIA_STOCK_FULL_OUTPUT_BUDGET_EXCEEDED')
        cpu, output_sha, finite = complete_tensor(tensor)
        raw = bytes(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))
        require(len(raw) == records[0]['original']['view_bytes'] and (finite or request['plan']['mode'] == 'fault'),
                'FIA_COMPLETE_STOCK_OUTPUT_REQUIRED')
        signature = geometry(records[0])
        ordinal = len(self.artifacts)
        destination = experiment.folder / ('fia-public-output-%05d.bin' % ordinal)
        with destination.open('xb') as stream:
            stream.write(raw)
        artifact = dict(path=str(destination), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            tensor_sha256=output_sha, request_id=request['request_id'], phase=request['phase'], target_key=target,
            independent_call=row['native_packet']['independent_call'], native_records_sha256=digest(records),
            complete_bytes=True, numerical_e5=False)
        self.artifacts.append(artifact)
        row['fia_full_output'] = artifact
        if request['phase'] in ('inventory_1', 'inventory_2'):
            require(not selected, 'FIA_STOCK_MUST_BE_UNREPLACED')
            rows = self.stock.setdefault(target, [])
            require(len(rows) == (0 if request['phase'] == 'inventory_1' else 1)
                and all(item['request_id'] != request['request_id'] for item in rows),
                'FIA_STOCK_OCCURRENCE_AMBIGUOUS')
            require(self.host_bytes + len(raw) <= 512 * 1024 * 1024, 'FIA_STOCK_GROUP_HOST_BUDGET_EXCEEDED')
            if rows:
                require(rows[0]['bytes'] == raw and rows[0]['geometry'] == signature, 'FIA_TWO_STOCK_REQUESTS_DIFFER')
            rows.append(dict(bytes=raw, geometry=json.loads(json.dumps(signature)), request_id=request['request_id'],
                             artifact=artifact.copy()))
            self.host_bytes += len(raw)
        elif selected:
            self.before_admit(target, records[0], request['plan']['mode'])
            if request['plan']['mode'] != 'fault':
                require(self.stock[target][0]['bytes'] == raw, 'FIA_STOCK_TO_OWNED_DESCRIPTOR_FULL_BYTES_DIFFER')
                row['fia_stock_bridge_equal'] = True
                if request['plan']['mode'] == 'reference':
                    self.references.add(target)

    def before_admit(self, target, before, mode):
        rows = self.stock.get(target, ())
        require(len(rows) == 2 and rows[0]['request_id'] != rows[1]['request_id']
            and rows[0]['bytes'] == rows[1]['bytes'] and rows[0]['geometry'] == rows[1]['geometry'] == geometry(before),
            'FIA_TWO_COMPLETE_CURRENT_STOCK_PROOFS_REQUIRED')
        require(mode == 'reference' or target in self.references, 'FIA_STOCK_TO_OWNED_REFERENCE_BEFORE_DEPENDENT_MODE')

    def close(self):
        self.stock.clear()
        self.host_bytes = 0

    def report(self):
        return dict(schema='demoncat.writer26.fia_stock_bridge.v3', retained_host_bytes=self.host_bytes,
            stock_target_count=len(self.stock), reference_bridge_targets=sorted(self.references), artifacts=self.artifacts,
            native_qualification=False, c2=False, c3=False)
