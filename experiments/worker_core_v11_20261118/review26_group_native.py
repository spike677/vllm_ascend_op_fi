"""Pure saved-original Native26 group checks, derived from pinned C1 reviewers.

The ABI, descriptor, original readback and stock byte checks retain the earlier
contracts. Explicit changes are every-target binding, selected-set membership
and complete-group cardinality. No new receipt is synthesized for the C1 API.
"""
from collections import Counter
import struct


def between_requests(r, previous, current, where):
    """Retain counted non-QMM cache runs, but permit no writer/event gap.

    Inventory::run increments all_runs and exactly one of qmm_runs and
    non_target_runs. FIA/AddRms have independent event counters, which must
    remain unchanged too. This is the existing C1 boundary rule with strict
    integer/key checks, not a waiver for unexplained Native activity.
    """
    fields=set(('version size flags reserved capacity total_events stored_events overflow_events errors '
        'all_runs qmm_runs non_target_runs qmm_enter qmm_exit fia_before fia_gws fia_enter fia_exit '
        'add_rms_gws add_rms_enter add_rms_exit qmm_registration').split())
    shape=(isinstance(previous,dict) and isinstance(current,dict) and set(previous)==set(current)==fields and
        all(type(x) is int and x>=0 for row in (previous,current) for x in row.values()))
    r.check(shape,'COMPLETE_INTEGER_NATIVE_BOUNDARY_STATES',where)
    if not shape:return None
    delta=current['all_runs']-previous['all_runs']
    accounted=(delta>=0 and delta==current['non_target_runs']-previous['non_target_runs'] and
        all(row['all_runs']==row['qmm_runs']+row['non_target_runs'] for row in (previous,current)) and
        all(current[k]==previous[k] for k in fields-{'all_runs','non_target_runs'}))
    r.check(accounted,'NO_UNACCOUNTED_NATIVE_WRITER_OR_EVENT_BETWEEN_REQUESTS',where)
    return dict(boundary=where,non_target_runs=delta,accounted=accounted,
                individual_non_target_operators_observed=False)


def group_outcome(b, r, value, candidate, row, qualification, where, *, mechanism=1):
    integer_packet, words, positive, valid_sha = b.integer_packet, b.words, b.positive, b.valid_sha
    MODE_IDS = b.MODE_IDS
    call = value['native_call']
    lease = value['lease'] if mechanism == 1 else value['lease']['lease']
    callback, selected = candidate['callback'], qualification['target_key']
    r.check(value['target'] == selected == callback['target_key'] and value['request_id'] == row['request_id']
        and call == callback['native_packet'] and value['mode'] == row['plan']['mode']
        and callback['claimed'] == 1 and callback.get('complete_group_current_binding_verified') is True, 'OUTCOME_ACTUAL_SELECTED_CALL', where)
    r.check(integer_packet(lease, arrays=('target',), nested=('expected',)) and len(lease['target']) == 4
        and lease['version'] == 2 and lease['size'] == 248 and lease['reserved'] == 0
        and lease['mode'] == MODE_IDS[value['mode']] and lease['target'] == words(selected)
        and [lease['owner_hi'], lease['owner_lo']] == qualification['native_owner']
        and lease['generation'] == qualification['generation'] and lease['thread'] == call['thread']
        and lease['expected'] == call, 'LEASE_OWNER_GENERATION_ACTUAL_FORMALS', where)
    if mechanism == 1:
        alloc = callback['allocation']
        r.check(lease['residual_allocation_end'] == alloc['residual_block'][1]
            and lease['output_allocation_begin'] == alloc['output_block'][0] == call['output']
            and lease['shift_bytes'] == alloc['shift'] == alloc['padding'] + 32 <= 4096,
            'LEASE_ACTUAL_ADJACENT_ALLOCATION', where)
    else:
        r.check(lease['residual_allocation_end'] == lease['output_allocation_begin'] == lease['shift_bytes'] == 0
            and call['output'] != call['residual'] and 0 < call['output_bytes'] <= call['residual_bytes'],
            'FIA_DIRECT_LIVE_RESIDUAL_DESTINATION', where)
    observed, receipt, released = (value[k] for k in ('observed_consumer_receipt', 'consumer_receipt', 'released_receipt'))
    r.check(observed == receipt and released == dict(receipt, released=1) and receipt['released'] == 0,
            'FIRST_NATIVE_RECEIPT_AND_ACK_ONLY_RELEASE', where)
    for label, p in (('consumer', receipt), ('released', released)):
        r.check(integer_packet(p, arrays=('target',)) and len(p['target']) == 4
            and p['version'] == 2 and p['size'] == 296 and p['reserved'] == p['error'] == p['writer_status'] == p['consumer_status'] == 0
            and p['mode'] == lease['mode'] and p['mechanism'] == mechanism and p['target'] == lease['target']
            and [p['owner_hi'], p['owner_lo']] == qualification['native_owner'] and p['generation'] == qualification['generation']
            and (p['registration'] == call['registration'] if mechanism == 1 else positive(p['registration']) and positive(p['executor']))
            and all(p[k] == call[k] for k in ('token', 'independent_call', 'cache', 'producer_sequence',
                'output', 'output_bytes', 'residual', 'residual_bytes')) and positive(p['stream']), 'NATIVE_RECEIPT_IDENTITY', where+':'+label)
        r.check(p['writer_calls'] == p['consumer_calls'] == 1 and all(p[k] == 1 for k in
            ('writer_complete', 'consumer_complete', 'output_readback_exact', 'padding_readback_exact', 'residual_restored_exact')),
            'NATIVE_EXECUTION_CONSUMPTION_RECOVERY', where+':'+label)
    mode = value['mode']
    if mode == 'fault':
        if mechanism == 1:
            r.check(0 < receipt['changed_residual_bytes'] <= 32 and receipt['residual_unchanged'] == 0
                and receipt['replacement'] == call['output'] - alloc['shift']
                == call['residual'] + call['residual_bytes'] - 32 and receipt['lossless_qualified'] == 1,
                'QMM_ACTUAL_32_BYTE_RESIDUAL_EFFECT', where)
        else:
            r.check(0 < receipt['changed_residual_bytes'] <= call['output_bytes'] and receipt['residual_unchanged'] == 0
                and receipt['replacement'] == call['residual'] and receipt['lossless_qualified'] == 1,
                'FIA_ACTUAL_RESIDUAL_DESTINATION_EFFECT', where)
    else:
        r.check(receipt['changed_residual_bytes'] == 0 and receipt['residual_unchanged'] == receipt['reference_equal'] == 1,
                'REFERENCE_LOSSLESS_OFF_CHANGED_RESIDUAL', where)
        if mode == 'lossless':
            r.check(positive(receipt['replacement']) and receipt['replacement'] not in (call['output'], call['residual'])
                and receipt['lossless_qualified'] == 1, 'LOSSLESS_REAL_TEMPORARY_DESTINATION', where)
        else:
            r.check(receipt['replacement'] == call['output'], 'REFERENCE_OFF_ORIGINAL_DESTINATION', where)
        if mode == 'off':
            r.check(receipt['off_exact'] == receipt['lossless_qualified'] == 1, 'EXACT_OFF_AFTER_LOSSLESS', where)
    r.check(valid_sha(value['writer_output_sha256']) and valid_sha(value['residual_before_sha256'])
        and value['residual_before_sha256'] == value['residual_after_sha256']
        and value['public_consumer_lifetime'] is True, 'FULL_RESIDUAL_RESTORATION_HASH', where)



def typed_group_review(b, t, inventory, artifacts, *, expected_count, isolated=False):
    trace_state, records = t.trace_state, t.records
    r = b.Review(); rows = inventory['rows']; bridges = inventory['fia_typed_stock_bridge']
    modes=[row['plan']['mode'] for row in rows]
    selected_order=[v['target_key'] for v in inventory['candidates'] if v['status']=='OBSERVED_NOT_QUALIFIED']
    if isolated:
        r.check(inventory['execution_scope']=='ALL_STABLE_ELIGIBLE_TARGETS_ISOLATED_F_OFF_FRESH_REQUEST_PAIRS' and
            inventory['independent_per_target_replay'] is True and inventory['coverage']['trigger']=='once',
            'EXPLICIT_ISOLATED_TYPED_SCOPE')
        expected_modes=[None,None,'reference','lossless','off']+['fault','off']*len(selected_order)+[None]
    else:
        expected_modes=[None,None,'reference','lossless','off','fault']
        if inventory.get('coverage',{}).get('trigger')=='every_match':expected_modes.append('fault')
        expected_modes.extend(('off',None))
    r.check(modes==expected_modes,'COMPLETE_TYPED_GENERATION_PHASES')
    public = inventory['public_output_observer']; outcomes = inventory['outcomes']
    public_by_call = {v['independent_call']:v for v in public['calls'] if v.get('independent_call')}
    r.check(len(public_by_call) == len(public['calls']) == (len(expected_modes)-1)*expected_count
        and public['owned_api_wrappers'] == public['inflight_public_calls'] == public['tensor_views_created'] == 0
        and public['source_scope'] == 'ACTUAL_FIA_PUBLIC_RETURN_ONLY' and not public['refusals'], 'ALL_ACTUAL_PUBLIC_FIA_RETURNS_AND_RESTORE')
    selected = {v['target_key'] for v in inventory['candidates'] if v['status'] == 'OBSERVED_NOT_QUALIFIED'}
    stock, seen_artifacts, seen_calls, typed_count = {}, [], set(), 0
    for index, row in enumerate(rows):
        before, after, values = row['fia_typed_before'], row['fia_typed_after'], row['fia_typed_records']
        label = row['phase']
        trace_state(b, r, before, label + ':before'); trace_state(b, r, after, label + ':after')
        if index:
            r.check(before == rows[index-1]['fia_typed_after'], 'NO_UNACCOUNTED_TYPED_CALL_BETWEEN_REQUESTS', label)
        r.check([v['sequence'] for v in values] == list(range(before['stored'] + 1, after['stored'] + 1))
            and all(after[k] >= before[k] for k in ('stored', 'total', 'before_gws', 'after_gws', 'before_execute')),
            'FULL_TYPED_SEQUENCE_WITHOUT_DROP_OR_RESET', label)
        counts = Counter(v['phase'] for v in values)
        owned=(1 if isolated and 5<=index<len(expected_modes)-1 else len(selected)) if row['plan']['mode'] else 0
        r.check(counts == Counter({1:expected_count, 2:expected_count, **({3:owned} if owned else {})})
            and all(after[name]-before[name] == counts[phase] for name,phase in
                    (('before_gws', 1), ('after_gws', 2), ('before_execute', 3))), 'COMPLETE_TYPED_DENOMINATOR', label)
        typed_count += len(values)
        calls = [e for e in row['native_events'] if e['kind_name'] == 'fia_before']
        r.check(len(calls) == expected_count and not seen_calls.intersection(v['call'] for v in calls), 'FULL_UNIQUE_FIA_REQUEST_CALLS', label)
        seen_calls.update(v['call'] for v in calls)
        callback_map = {c['independent_call']:c for c in row['callbacks']}
        for ordinal, start in enumerate(calls):
            call_id = start['call']; loc = label + ':' + str(ordinal)
            actual = [v for v in values if v['call'] == call_id]
            events = [v for v in row['native_events'] if v['call'] == call_id and v['kind_name'] in ('fia_before', 'fia_gws', 'fia_enter', 'fia_exit')]
            callback = callback_map.get(call_id)
            chosen = callback is not None and callback['claimed'] == 1
            if isolated:
                target=callback.get('target_key') if callback else None
                expected_chosen=(target in selected if 2<=index<=4 else
                    target==selected_order[(index-5)//2] if 5<=index<len(expected_modes)-1 else False)
                r.check(chosen is expected_chosen,'ONLY_CURRENT_ISOLATED_TYPED_TARGET_REPLACED',loc)
            geometry = r.section(loc, records, b, r, actual, events, selected=chosen, where=loc)
            if callback is None:
                r.check(index == len(expected_modes)-1 and not chosen, 'CALLBACK_MISSING_BEFORE_NATIVE_REMOVE', loc)
                continue
            r.check(callback['fia_typed_before'] == actual[0] and callback['fia_typed_public_records'] == actual
                and not callback.get('public_output_refusals'), 'CALLBACK_CURRENT_TYPED_PUBLIC_RECORDS', loc)
            kept = callback['writer_output_retained']; observed = public_by_call[call_id]
            r.check(kept['evidence'] == 'ACTUAL_PUBLIC_FIA_RETURN' and kept['independent_call'] == call_id
                and kept['entry_sequence'] == events[0]['sequence'] and kept['exit_sequence'] == events[-1]['sequence']
                and b.positive(kept['api_callable_identity']) and b.positive(kept['dispatcher_packet_identity'])
                and b.positive(kept['storage_identity']) and observed['tensor_object_identity'] == kept['tensor_object_identity']
                and observed['api'] == kept['api'] == callback['public_output_scope'] and observed['original_calls'] == 1
                and observed['request_id'] == row['request_id'] and observed['retained'] is True and not observed['errors']
                and observed['typed_records'] == actual, 'LITERAL_PUBLIC_RETURN_OBJECT_AND_ORIGINAL_CALL', loc)
            public_before, public_after = observed['typed_before'], observed['typed_after']
            trace_state(b, r, public_before, loc + ':public_before')
            trace_state(b, r, public_after, loc + ':public_after')
            r.check(public_before['stored'] == actual[0]['sequence'] - 1
                and public_after['stored'] == actual[-1]['sequence']
                and all(public_after[name] - public_before[name] == expected for name, expected in
                    (('before_gws', 1), ('after_gws', 1), ('before_execute', int(chosen)))),
                'ONLY_CURRENT_NATIVE_CALL_INSIDE_LITERAL_PUBLIC_RETURN', loc)
            expected_format = dict(public_format=2, native_format=47, shape=kept['shape'], stride=kept['stride'],
                dtype='torch.bfloat16', storage_offset=0, contiguous=True, full_native_formal_matched=True, exact_format_match=False)
            r.check(observed['format_observation'] == expected_format and kept['format'] == 47
                and kept['storage_address'] == start['output'] and kept['bytes'] == start['output_bytes']
                and kept['shape'] == start['shape'][:3] and kept['stride'] == start['strides'][:3], 'PUBLIC_ND_IS_CURRENT_TYPED_NCL_FORMAL', loc)
            if index < 2 or chosen:
                artifact = callback['fia_full_output']
                r.check(artifact['request_id'] == row['request_id'] and artifact['phase'] == label
                    and artifact['target_key'] == callback['target_key'] and artifact['independent_call'] == call_id,
                    'ORIGINAL_FULL_OUTPUT_ARTIFACT_CALL_JOIN', loc)
                raw = r.section(loc + ':full_output_bin', artifacts.bytes, artifact, actual)
                seen_artifacts.append(artifact)
                if raw is None:
                    continue
                finite = sum((bits & 0x7f80) == 0x7f80 for bits, in struct.iter_unpack('<H', raw)) == 0
                r.check(finite or row['plan']['mode'] == 'fault', 'HEALTHY_STOCK_AND_CONTROL_FULL_BYTES_FINITE', loc)
                if index < 2:
                    previous = stock.setdefault(callback['target_key'], [])
                    r.check(not chosen and len(previous) == index, 'TWO_UNREPLACED_STOCK_OCCURRENCES', loc)
                    if previous:
                        r.check(raw == previous[0][0] and geometry == previous[0][1], 'TWO_STOCK_FULL_BYTES_AND_GEOMETRY_EQUAL', loc)
                    previous.append((raw, geometry, artifact))
                else:
                    baseline = stock.get(callback['target_key'], [])
                    r.check(len(baseline) == 2 and callback['target_key'] in selected
                        and geometry == baseline[0][1], 'FRESH_SELECTED_STOCK_GEOMETRY', loc)
                    if len(baseline) != 2:
                        continue
                    if row['plan']['mode'] != 'fault':
                        r.check(raw == baseline[0][0] and callback.get('fia_stock_bridge_equal') is True,
                            'FRESH_OWNED_REFERENCE_LOSSLESS_OFF_FULL_STOCK_BYTES', loc)
            else:
                r.check('fia_full_output' not in callback, 'UNSELECTED_CALL_NOT_REPORTED_AS_NEW_REFERENCE', loc)
    owned_total=5*len(selected) if isolated else sum(x is not None for x in expected_modes)*len(selected)
    r.check(seen_artifacts == bridges['artifacts'] and len(seen_artifacts) == 2*expected_count + owned_total
        and len(artifacts.files) == len(seen_artifacts) and len(stock) == expected_count,
        'COMPLETE_STOCK_AND_SELECTED_OUTPUT_ARTIFACT_SET')
    r.check(bridges['schema'] == 'demoncat.writer26.fia_stock_bridge.v3' and bridges['retained_host_bytes'] == 0
        and bridges['stock_target_count'] == 0 and set(bridges['reference_bridge_targets']) == selected and len(bridges['reference_bridge_targets']) == len(selected), 'STOCK_OWNER_ZERO_AND_ONLY_FRESH_REFERENCE')
    core = inventory['native_state']['core']
    qualification = dict(native_owner=[core['owner_hi'],core['owner_lo']],generation=core['generation'])
    for outcome in outcomes:
        loc = outcome['mode'] + ':' + outcome['request_id']
        row = next(v for v in rows if v['request_id'] == outcome['request_id'])
        callback = next(v for v in row['callbacks'] if v['native_packet']['token'] == outcome['native_call']['token'])
        actual = callback['fia_typed_public_records']; public_evidence = outcome['fia_public_typed']
        snapshot = outcome['fia_typed_consumer_snapshot']; lease = outcome['lease']
        r.check(lease['version'] == 3 and lease['size'] == 912 and lease['original_before'] == actual[0], 'ACTUAL_TYPED_ADMISSION_ABI')
        r.check(b.integer_packet(snapshot, nested=('before', 'after', 'before_execute'))
            and snapshot['version'] == 3 and snapshot['size'] == 2048
            and snapshot['token'] == outcome['native_call']['token']
            and [snapshot['owner_hi'], snapshot['owner_lo']] == qualification['native_owner']
            and snapshot['generation'] == qualification['generation'] and snapshot['descriptor'] == snapshot['exceptions'] == 0
            and snapshot['created'] == snapshot['released'] == snapshot['native_return_success'] == 1
            and [snapshot['before'], snapshot['after'], snapshot['before_execute']] == actual
            and snapshot == public_evidence['snapshots'], 'OWNED_DESCRIPTOR_RELEASE_AND_CONSUMER_TIME_SNAPSHOT')
        retained = callback['writer_output_retained']; binding = callback['public_consumer_binding']
        r.check(public_evidence['public_tensor_identity'] == retained['tensor_object_identity'] == binding['writer_tensor_object_identity']
            and public_evidence['public_storage_identity'] == retained['storage_identity']
            and outcome['writer_output_sha256'] == callback['fia_full_output']['tensor_sha256'], 'LITERAL_RETURN_STILL_HELD_THROUGH_CONSUMER')
        receipt = outcome['consumer_receipt']; effective = actual[2]['effective']
        r.check(receipt['executor'] == actual[2]['executor'] and receipt['registration'] == actual[2]['registration']
            and receipt['replacement'] == effective['raw'], 'NATIVE_RECEIPT_ACTUAL_TYPED_EXECUTOR_DESTINATION')
        if outcome['mode'] != 'fault':
            r.check(public_evidence['stock_equal'] is True, 'NONFAULT_STOCK_BRIDGE_RECEIPT')
    final = rows[-1]['fia_typed_after']
    trace_state(b, r, final, 'final')
    return dict(**r.summary(), typed_records=typed_count, full_output_bins_rehashed=len(artifacts.files),
        full_output_bytes_rehashed=sum(v['bytes'] for v in artifacts.files.values()), stock_targets=len(stock),
        initial_typed_records_outside_windows=rows[0]['fia_typed_before']['stored'],
        source_bound_descriptor_receipts_reconciled=not r.errors, residual_and_logits_tensor_bins_available=False,
        final_typed_state_boundary='LAST_ACTUAL_HEALTH_REQUEST_AFTER', separate_post_cleanup_typed_state_available=False,
        public_qualification_granted=False, c2=False, c3=False, e5=False)
