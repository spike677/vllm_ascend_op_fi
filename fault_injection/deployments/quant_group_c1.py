"""Single-site receipt checks for the separate quant group ABI.

These checks do not authorize a target or an all-eligible group. The serving
adapter supplies the original live call lease and retains the whole census.
"""
import ctypes

from ..quant_group_native import Call, Completion, Lease, RawRecord, Receipt, QuantGroupNative, snapshot
from .quant_group_inventory import require


def public_views_match(call, x, q, scale, device):
    for name, tensor in (('input', x), ('quant', q), ('scale', scale)):
        actual = snapshot(QuantGroupNative.tensor_view(tensor, name, device))
        require(call[name]['descriptor'] > 0 and all(call[name][field] == value
                for field, value in actual.items() if field != 'descriptor'),
                'C1_ACTUAL_PUBLIC_NATIVE_FORMALS_CHANGED:' + name)


def verify_completion(done, rows, lease, *, released):
    """Reconcile all real GWS/execute events, including the F reference helper."""
    expected = snapshot(lease)
    parent, receipt = done['call'], done['core']
    mode = expected['mode']
    require(all(value['version'] == 1 and value['size'] == ctypes.sizeof(kind) and not value['reserved']
                for value, kind in ((done, Completion), (parent, Call), (receipt, Receipt), (expected, Lease)))
            and all(row['version'] == 1 and row['size'] == ctypes.sizeof(RawRecord) and not row['reserved'] for row in rows),
            'C1_ACTUAL_VERSIONED_NATIVE_RECORDS_REQUIRED')
    require(mode in (1, 2, 3, 4) and done['admitted'] == 1 and done['status'] == 0
            and parent == expected['expected'] and parent['lifetime_token'] > 0,
            'C1_NATIVE_COMPLETION_DOES_NOT_MATCH_CURRENT_PARENT')
    require(receipt['mode'] == mode and receipt['parent_token'] == parent['token']
            and receipt['lifetime_token'] == parent['lifetime_token']
            and all(receipt[name] == expected[name] for name in
                    ('owner_hi', 'owner_lo', 'generation', 'target', 'actual_row_proof'))
            and receipt['error'] == receipt['last_error'] == 0,
            'C1_CURRENT_NATIVE_LEASE_RECEIPT_REQUIRED')
    require(all(receipt[name] == 1 for name in
                ('input_before_captured', 'input_unchanged', 'published', 'publication_exact'))
            and receipt['released'] == int(released)
            and not receipt['unsafe_allocation_address'] and not receipt['unsafe_allocation_bytes'],
            'C1_FULL_CURRENT_INPUT_AND_EXACT_PUBLICATION_REQUIRED')
    roles = (1, 2) if mode == 3 else (2,)
    require(len(rows) == 1 + 4 * len(roles) and rows[0]['kind'] == 1 and rows[0]['parent'] == parent
            and all(row['kind'] == 2 for row in rows[1:]),
            'C1_ALL_MODEL_AND_HELPER_NATIVE_EVENTS_REQUIRED')
    events = {row['native']['sequence']: row for row in rows[1:]}
    require(len(events) == 4 * len(roles) and sorted(events) == list(range(parent['token'] + 1, parent['token'] + len(rows))),
            'C1_RAW_NATIVE_EVENT_GAP_OR_DUPLICATE')
    seen = set()
    streams = set()
    for role in roles:
        child = receipt['reference' if role == 1 else 'effect']
        plan = child['plan']
        require(plan['role'] == role and plan['parent_token'] == parent['token']
                and plan['child_lease'] == parent['token'] * 4 + role and not plan['reserved']
                and all(child[name] == 1 for name in
                        ('prepare_original_calls', 'execute_original_calls', 'prepared', 'returned', 'drained', 'readback'))
                and child['prepare_status'] == child['execute_status'] == 0,
                'C1_REAL_NATIVE_CHILD_PREPARE_EXECUTE_READBACK_REQUIRED')
        ids = [child[name] for name in ('prepare_enter', 'prepare_exit', 'execute_enter', 'execute_exit')]
        require(parent['token'] < ids[0] < ids[1] < ids[2] < ids[3]
                and child['registration'] == ids[0] and child['executor'] > 0 and child['stream'] > 0,
                'C1_CHILD_ACTUAL_REGISTRATION_ORDER_REQUIRED')
        expected_binding = dict(parent_token=parent['token'], child_lease=plan['child_lease'],
            owner_hi=expected['owner_hi'], owner_lo=expected['owner_lo'], generation=expected['generation'], role=role, reserved=0)
        for kind, sequence in enumerate(ids, 1):
            row = events.get(sequence)
            require(row is not None and sequence not in seen and row['binding'] == expected_binding,
                    'C1_CHILD_RAW_OWNER_ROLE_OR_REGISTRATION_CHANGED')
            seen.add(sequence)
            event = row['native']
            require(event['kind'] == kind and event['registration'] == child['registration']
                    and event['thread'] == parent['thread'] and event['device'] == parent['device']
                    and not event['reserved'] and event['status'] == 0
                    and event['original_calls'] == int(kind in (2, 4))
                    and event['executor'] == (0 if kind == 1 else child['executor'])
                    and not any(event[name] for name in ('smooth_tensor', 'group_index_tensor', 'offset_tensor'))
                    and event['destination_dtype'] == 2
                    and all(event[name] == plan[name] for name in ('input', 'quant', 'scale')),
                    'C1_CHILD_ACTUAL_NATIVE_FORMALS_CHANGED')
            if kind >= 2:
                require(event['workspace_required'] == child['workspace_required'], 'C1_CHILD_GWS_WORKSPACE_CHANGED')
            if kind in (3, 4):
                require(event['stream'] == child['stream'] and event['workspace'] == child['workspace']
                        and event['workspace_capacity'] == child['workspace_capacity'],
                        'C1_CHILD_EXECUTE_STREAM_OR_WORKSPACE_CHANGED')
        require(child['workspace_capacity'] >= child['workspace_required']
                and bool(child['workspace']) == bool(child['workspace_capacity'])
                and child['workspace_group_owned'] == int(role == 1 and child['workspace_required'] > 0),
                'C1_HELPER_OWNED_AND_FRAMEWORK_BORROWED_WORKSPACE_REQUIRED')
        for name in ('input', 'quant', 'scale'):
            actual, correct = plan[name], dict(parent[name])
            if mode == 3 and role == 2:
                if name in ('input', 'quant'):
                    correct.update(rows=1, columns=parent['input']['rows'] * parent['input']['columns'],
                                   stride0=parent['input']['rows'] * parent['input']['columns'])
                else:
                    correct.update(rows=1, bytes=4)
            require(actual['descriptor'] > 0 and not actual['reserved'] and all(actual[field] == correct[field]
                    for field in correct if field not in ('address', 'descriptor')),
                    'C1_CHILD_DOMAIN_OR_DTYPE_CHANGED:' + name)
            if name == 'input' or mode in (1, 4):
                require(actual['address'] == parent[name]['address'], 'C1_CHILD_BORROWED_FORMAL_CHANGED')
        if mode in (2, 3):
            for name in ('quant', 'scale'):
                start, size = plan[name]['address'], parent[name]['bytes']
                require(start > 0 and all(start + size <= parent[other]['address']
                    or parent[other]['address'] + parent[other]['bytes'] <= start for other in ('input', 'quant', 'scale')),
                    'C1_OWNED_OUTPUT_ALIASES_PUBLIC_FORMAL')
        streams.add(child['stream'])
    require(len(streams) == 1 and seen == set(events), 'C1_UNCLASSIFIED_OR_FOREIGN_STREAM_CHILD')
    effect, reference = receipt['effect']['plan'], receipt['reference']['plan']
    if mode == 3:
        owned = [(plan[name]['address'], parent[name]['bytes']) for plan in (reference, effect) for name in ('quant', 'scale')]
        require(all(a + n <= b or b + m <= a for index, (a, n) in enumerate(owned) for b, m in owned[index + 1:]),
                'C1_REFERENCE_AND_EFFECT_OUTPUTS_MUST_BE_DISJOINT')
        require(receipt['reference_effect_compared'] == 1
                and receipt['changed_quant_bytes'] + receipt['changed_scale_bytes'] > 0
                and receipt['changed_quant_bytes'] <= parent['quant']['bytes']
                and receipt['changed_scale_bytes'] <= parent['scale']['bytes'],
                'C1_ACTUAL_CURRENT_INPUT_Q_OR_SCALE_EFFECT_REQUIRED')
        require(receipt['reference']['execute_exit'] < receipt['effect']['execute_enter'],
                'C1_REFERENCE_MUST_COMPLETE_BEFORE_MODEL_EFFECT')
        require(all(reference[name][field] == parent[name][field]
                    for name in ('input', 'quant', 'scale') for field in
                    ('bytes', 'rows', 'columns', 'stride0', 'stride1', 'offset', 'dtype', 'format', 'dimensions')),
                'C1_REFERENCE_MUST_USE_ACTUAL_CORRECT_DOMAIN')
        require(effect['input']['address'] == reference['input']['address'] == parent['input']['address']
                and effect['input']['bytes'] == parent['input']['bytes']
                and effect['input']['rows'] == effect['quant']['rows'] == effect['scale']['rows'] == 1
                and effect['input']['columns'] == effect['quant']['columns'] == parent['input']['rows'] * parent['input']['columns']
                and effect['scale']['bytes'] == 4,
                'C1_BATCH_SHARED_SCALE_EFFECT_DOMAIN_CHANGED')
    else:
        require(not receipt['reference_effect_compared'] and not receipt['changed_quant_bytes']
                and not receipt['changed_scale_bytes'] and receipt['reference']['prepare_enter'] == 0,
                'C1_HEALTHY_MODE_MUST_NOT_CLAIM_FAULT_HELPER')
    return dict(call=parent['token'], registration=receipt['effect']['registration'],
        executor=receipt['effect']['executor'], stream=receipt['effect']['stream'], native_thread=parent['thread'],
        original_model_calls=1, original_prepare_calls=len(roles), original_execute_calls=len(roles),
        helper_calls=int(mode == 3), complete_formals=True, native_effect=mode == 3,
        registration_lifetime='EACH_ACTUAL_CHILD_PREPARE_TO_EXECUTE_ONLY',
        target=expected['target'], actual_row_proof=expected['actual_row_proof'], fault_qualification=False)
