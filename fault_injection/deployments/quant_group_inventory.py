"""Lossless model-parent/typed-child census for the isolated quant group ABI.

This reader grants no group lease. Every raw parent/helper/event is retained,
including ones that cannot be attached to a public projection or request row.
"""
from collections import Counter

from ..quant_group_native import QuantGroupNative, snapshot


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


class GroupQuantJournal:
    def __init__(self, binding, *, device):
        require(type(binding) is QuantGroupNative and type(device) is int and device >= 0,
                'REVIEWED_GROUP_BINDING_AND_ACTUAL_DEVICE_REQUIRED')
        self.binding, self.device = binding, device
        self.build_identity = dict(binding.identity)

    def state(self):
        value = self.binding.query()
        census = value['census']
        require(census['capacity'] == 32768 and census['total'] == value['last_sequence']
                and not census['overflow'] and not census['errors'], 'COMPLETE_GROUP_RAW_CENSUS_REQUIRED')
        require(not value['abi_exceptions'] and not value['orphan_descriptors'], 'GROUP_NATIVE_OWNERSHIP_UNCERTAIN')
        return dict(value, pending=census['preparing'] + census['pending'], executing=census['executing'])

    def delta(self, before):
        after = self.state()
        require(after['last_sequence'] >= before['last_sequence'], 'GROUP_RAW_CENSUS_RESET')
        rows = (self.binding.raw_range(before['last_sequence'] + 1, after['last_sequence'])
                if after['last_sequence'] > before['last_sequence'] else [])
        counts = Counter()
        for index, row in enumerate(rows, before['last_sequence'] + 1):
            if row['kind'] == 1:
                require(row['parent']['token'] == index, 'GROUP_PARENT_SEQUENCE_GAP')
                counts['model_parents'] += 1
            else:
                require(row['kind'] == 2 and row['native']['sequence'] == index
                        and row['native']['kind'] in (1, 2, 3, 4), 'GROUP_NATIVE_EVENT_SEQUENCE_GAP')
                if row['native']['kind'] == 1:
                    counts['native_prepares'] += 1
                if row['native']['kind'] == 3:
                    counts['native_executes'] += 1
        for name in ('model_parents', 'native_prepares', 'native_executes'):
            require(counts[name] == after['census'][name] - before['census'][name],
                    'GROUP_RAW_COUNTS_DO_NOT_RECONCILE:' + name)
        return after, rows


def verify_public_group_call(rows, x, q, scale, native_thread, *, expected_device):
    """Verify one actual uninstalled/pass-through model invocation, never helpers."""
    require(len(rows) == 5 and [row['kind'] for row in rows] == [1, 2, 2, 2, 2],
            'ONE_GROUP_MODEL_PARENT_AND_FOUR_REAL_NATIVE_EVENTS_REQUIRED')
    parent, native = rows[0]['parent'], [row['native'] for row in rows[1:]]
    require([row['kind'] for row in native] == [1, 2, 3, 4]
            and parent['token'] < native[0]['sequence']
            and native[0]['registration'] == native[0]['sequence']
            and all(row['registration'] == native[0]['registration'] for row in native),
            'GROUP_ACTUAL_PREPARE_EXECUTE_REGISTRATION_CHANGED')
    require(parent['thread'] == native_thread and parent['device'] == expected_device and not parent['reserved']
            and all(row['thread'] == native_thread and row['device'] == expected_device and not row['reserved']
                    and row['status'] == 0 for row in native)
            and [row['original_calls'] for row in native] == [0, 1, 0, 1]
            and native[1]['executor'] != 0 and len({row['executor'] for row in native[1:]}) == 1,
            'GROUP_NATIVE_ACTUAL_THREAD_DEVICE_OR_EXECUTION_UNPROVEN')
    require(all(not any(row['binding'].values()) for row in rows[1:]),
            'READONLY_GROUP_CENSUS_MUST_NOT_HIDE_CLASSIFIED_CHILDREN')
    for row in native:
        require(not row['smooth_tensor'] and not row['group_index_tensor'] and not row['offset_tensor']
                and row['destination_dtype'] == 2, 'UNREVIEWED_DYNAMIC_QUANT_OPTIONAL_FORMALS')
    for name, tensor, role in (('input', x, 'input'), ('quant', q, 'quant'), ('scale', scale, 'scale')):
        actual = snapshot(QuantGroupNative.tensor_view(tensor, role, expected_device))
        observed = parent[name]
        require(observed['descriptor'] > 0 and not observed['reserved'], 'ACTUAL_NATIVE_DESCRIPTOR_REQUIRED')
        require(all(observed[key] == value for key, value in actual.items() if key != 'descriptor')
                and all(row[name] == observed for row in native), 'ACTUAL_GROUP_PUBLIC_NATIVE_FORMALS_DIFFER:' + name)
    require(native[2]['stream'] != 0 and native[3]['stream'] == native[2]['stream']
            and native[2]['workspace_capacity'] >= native[1]['workspace_required']
            and native[3]['workspace'] == native[2]['workspace']
            and native[3]['workspace_capacity'] == native[2]['workspace_capacity']
            and bool(native[2]['workspace']) == bool(native[2]['workspace_capacity']),
            'ACTUAL_GROUP_WORKSPACE_OR_STREAM_UNPROVEN')
    return dict(call=parent['token'], registration=native[0]['registration'], executor=native[1]['executor'],
        stream=native[2]['stream'], native_thread=native_thread, original_model_calls=1,
        original_prepare_calls=1, original_execute_calls=1, helper_calls=0, complete_formals=True,
        registration_lifetime='THIS_ACTUAL_PREPARE_TO_EXECUTE_ONLY', fault_qualification=False)
