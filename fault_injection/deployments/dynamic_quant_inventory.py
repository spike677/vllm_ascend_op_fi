"""Actual typed DynamicQuant V2 census, independent of selected fault counters."""
from collections import Counter
import ctypes as C


KINDS = {1: 'prepare_enter', 2: 'prepare_exit', 3: 'execute_enter', 4: 'execute_exit'}


class QuantEvent(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('version', 'size', 'kind', 'flags')]
    _fields_ += [(name, C.c_int32) for name in ('device', 'status')]
    _fields_ += [(name, C.c_uint64) for name in ('sequence', 'call', 'thread', 'executor', 'input', 'quant', 'scale')]
    _fields_ += [(name, C.c_int64) for name in ('rows', 'columns')]
    _fields_ += [(name, C.c_int32) for name in ('input_dtype', 'quant_dtype', 'scale_dtype',
                                               'input_format', 'quant_format', 'scale_format')]
    _fields_ += [('original_calls', C.c_uint64)]


class QuantState(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('version', 'size', 'reserved1', 'reserved2')]
    _fields_ += [(name, C.c_uint64) for name in ('capacity', 'total_events', 'stored_events', 'overflow_events',
        'errors', 'prepare_enter', 'prepare_exit', 'execute_enter', 'execute_exit', 'metadata_complete',
        'orphan_executions', 'registrations', 'pending', 'executing')]


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def packet(value):
    return {name: int(getattr(value, name)) for name, _ in value._fields_}


class NativeDynamicQuantInventory:
    def __init__(self, runtime):
        import torch_npu
        self.device = int(torch_npu.npu.current_device())
        self.build_identity = runtime.build_identity
        self.lib = runtime.backend.lib
        self.state_fn = self.lib.dcat_dq_inventory_state_v1
        self.read_fn = self.lib.dcat_dq_inventory_read_v1
        self.state_fn.argtypes, self.state_fn.restype = [C.POINTER(QuantState)], C.c_int
        self.read_fn.argtypes, self.read_fn.restype = [C.c_uint64, C.POINTER(QuantEvent)], C.c_int
        require(C.sizeof(QuantEvent) == C.sizeof(QuantState) == 128, 'INDEPENDENT_QUANT_ABI_SIZE_CHANGED')

    def state(self):
        value = QuantState(version=1, size=C.sizeof(QuantState))
        require(self.state_fn(C.byref(value)) == 0 and value.version == 1 and value.size == C.sizeof(value),
                'INDEPENDENT_QUANT_STATE_ABI_CHANGED')
        row = packet(value)
        require(row['capacity'] == 32768 and row['reserved1'] == row['reserved2'] == 0,
                'INDEPENDENT_QUANT_CAPACITY_OR_ABI_CHANGED')
        require(row['total_events'] == row['stored_events'] and row['overflow_events'] == row['errors'] == 0,
                'INDEPENDENT_QUANT_OVERFLOW_OR_ERROR')
        require(row['orphan_executions'] == 0, 'NATIVE_QUANT_EXECUTION_WITHOUT_ACTUAL_PREPARE')
        return row

    def read(self, index):
        value = QuantEvent(version=1, size=C.sizeof(QuantEvent))
        require(self.read_fn(index, C.byref(value)) == 0 and value.version == 1 and value.size == C.sizeof(value),
                'INDEPENDENT_QUANT_EVENT_ABI_CHANGED')
        row = packet(value)
        require(row['sequence'] == index + 1 and row['kind'] in KINDS, 'INDEPENDENT_QUANT_EVENT_GAP')
        row['kind_name'] = KINDS[row['kind']]
        return row

    def delta(self, before):
        after = self.state()
        counters = ('total_events', 'stored_events', 'prepare_enter', 'prepare_exit', 'execute_enter',
                    'execute_exit', 'metadata_complete', 'registrations')
        require(all(after[name] >= before[name] for name in counters), 'INDEPENDENT_QUANT_COUNTER_RESET')
        rows = [self.read(index) for index in range(before['stored_events'], after['stored_events'])]
        counts = Counter(row['kind_name'] for row in rows)
        require(all(counts[name] == after[name] - before[name] for name in KINDS.values()),
                'INDEPENDENT_QUANT_DETAILS_DO_NOT_MATCH_COUNTERS')
        return after, rows


def verify_public_native_call(rows, x, q, scale, native_thread, *, expected_device):
    """Exact current formals/executor registration; never shape-only ownership."""
    import torch
    require(len(rows) == 4 and [r['kind'] for r in rows] == [1, 2, 3, 4], 'ONE_ACTUAL_NATIVE_QUANT_CALL_REQUIRED')
    first = rows[0]
    stable = ('call', 'thread', 'device', 'flags', 'input', 'quant', 'scale', 'rows', 'columns',
              'input_dtype', 'quant_dtype', 'scale_dtype', 'input_format', 'quant_format', 'scale_format')
    require(first['call'] and all(all(r[key] == first[key] for key in stable) for r in rows),
            'NATIVE_QUANT_CALL_LIFETIME_CHANGED')
    require(first['flags'] == 31 and first['thread'] == native_thread and first['device'] == expected_device
            and all(r['status'] == 0 for r in rows)
            and [r['original_calls'] for r in rows] == [0, 1, 0, 1]
            and rows[1]['executor'] != 0 and rows[1]['executor'] == rows[2]['executor'] == rows[3]['executor'],
            'NATIVE_QUANT_EXECUTOR_OR_ORIGINAL_EXECUTION_UNPROVEN')
    require(x.dtype in (torch.bfloat16, torch.float16) and q.dtype == torch.int8 and scale.dtype == torch.float32
            and x.ndim == q.ndim == 2 and q.shape == x.shape and scale.shape == (x.shape[0],),
            'PUBLIC_QUANT_COMPLETE_FORMALS_REQUIRED')
    require(all(t.storage_offset() == 0 and t.is_contiguous() for t in (x, q, scale))
            and [first[name] for name in ('input', 'quant', 'scale')] == [t.data_ptr() for t in (x, q, scale)]
            and [first['rows'], first['columns']] == list(x.shape)
            and first['input_dtype'] == (27 if x.dtype == torch.bfloat16 else 1)
            and first['quant_dtype'] == 2 and first['scale_dtype'] == 0
            and all(first[name] == 2 for name in ('input_format', 'quant_format', 'scale_format')),
            'ACTUAL_PUBLIC_NATIVE_QUANT_FORMALS_DIFFER')
    return dict(call=first['call'], executor=rows[1]['executor'], native_thread=native_thread,
        registration_lifetime='THIS_ACTUAL_PREPARE_TO_EXECUTE_ONLY', original_prepare_calls=1, original_execute_calls=1,
        complete_formals=True, fault_qualification=False)
