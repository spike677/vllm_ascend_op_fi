"""Read-only source62 single typed-FIA review; no runtime imports or execution.

Reuse the independent QMM reader's request, lifetime, receipt, recovery,
metadata-array and provenance checks with explicit FIA semantics. This file
adds full typed descriptor joins and rehashes the original stock/output bins.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys


SOURCE = 'dc40a87d5e21cd8a36ba24e1fa542bcd0e543bc2'
MANIFEST = '352bb926eb5f0ea73612413aae743546e05724889f37b9d03617cdc10bf01a67'
BUILD = '2a6e4b861fe0342cb4274614d6edeb187bb36e3c87acab9ecc99be6f0e01bbae'
ORIGINALS = {
    'result': ('result.json', 'c47ec0438492479e8b1ace37fcd9176d4031051a74b1648f0ad7863ed25045dd'),
    'inventory': ('writer-v2/candidate-experiment/writer_inventory.json', '20ddf4e7045fe636ea5c21d1c80557348baa7146ecdf806197c16ae82ac29a43'),
    'serving': ('writer-v2/runtime/serving_receipt.json', 'f67623cbff02e1833c0f2b8c6ace18f3cc06c3423c2c39eec9e113a680626094')}
SELECTED = ('0825e9f4e3d656493f4a945c24e754961fd48c76fdbb5905f1a43c45243065a6', 17)
SCOPE = 'ONE_ACTUAL_TYPED_FIA_SITE_FIXED_FRESH_PREFILL_NOT_ALL_ELIGIBLE_NOT_C2_C3'
EXTRA_SOURCES = ('fault_injection/deployments/writer26_fia_typed.py',
    'fault_injection/native_adapters/writer26_fia_typed_api.h',
    'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_fia_typed.h')
BASE_SHA = '8151f93dca35ead3b32b98d66881e5bc23432884bf5af8c1c6694f1b59fb8b5d'


def shared():
    path = Path(__file__).with_name('review_native26_single_source54.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('INDEPENDENT_SHARED_REVIEWER_CHANGED')
    spec = importlib.util.spec_from_file_location('_independent_native26_shared', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor(b, r, value, where):
    shape = value['view'][:3]
    elements = math.prod(shape)
    r.check(b.integer_packet(value, arrays=('view', 'storage', 'strides'))
        and value['version'] == 2 and value['size'] == 280 and value['complete'] == value['contiguous'] == 1
        and value['dtype'] == 27 and value['format'] == 47 and value['offset'] == value['reserved'] == 0
        and value['view_rank'] == value['stride_rank'] == 3 and value['storage_rank'] in (1, 3)
        and value['element_bytes'] == 2 and 0 < value['descriptor'] < 2**64 and 0 < value['raw'] < 2**64,
        'COMPLETE_TYPED_BF16_NCL_DESCRIPTOR', where)
    expected_storage = shape if value['storage_rank'] == 3 else [elements]
    r.check(all(len(value[k]) == 8 for k in ('view', 'storage', 'strides'))
        and not any(value['view'][3:]) and not any(value['strides'][3:])
        and value['storage'] == expected_storage + [0]*(8-len(expected_storage))
        and b.contiguous(shape, value['strides'][:3]) and value['view_bytes'] == value['storage_bytes'] == elements*2
        and 0 < elements*2 <= 8*1024**2 and value['raw'] + elements*2 < 2**64, 'EXACT_NCL_VIEW_FLAT_STORAGE_AND_STRIDES', where)
    return {k:v for k,v in value.items() if k not in ('descriptor', 'raw')}


def trace_state(b, r, state, where):
    r.check(b.integer_packet(state) and state['version'] == 2 and state['size'] == 104 and state['capacity'] == 32768
        and 0 <= state['total'] == state['stored'] == state['before_gws'] + state['after_gws'] + state['before_execute'] <= state['capacity']
        and all(state[k] == 0 for k in ('overflow', 'errors', 'exceptions', 'preparing', 'prepared', 'executing')),
        'COMPLETE_QUIESCENT_TYPED_TRACE_ABI', where)


def records(b, r, values, native_events, *, selected, where):
    phases = [1, 2, 3] if selected else [1, 2]
    r.check([v['phase'] for v in values] == phases and len(native_events) == 4, 'EXACT_TYPED_PHASES_AND_NATIVE_LIFECYCLE', where)
    before, after = values[:2]
    start, gws, enter, end = native_events
    r.check([v['kind_name'] for v in native_events] == ['fia_before', 'fia_gws', 'fia_enter', 'fia_exit']
        and all(v['call'] == start['call'] and v['thread'] == start['thread'] and v['status'] == 0 for v in native_events)
        and start['sequence'] < gws['sequence'] < enter['sequence'] < end['sequence']
        and gws['executor'] == enter['executor'] == end['executor'] != 0 and enter['registration'] > 0,
        'ACTUAL_ORIGINAL_FIA_GWS_EXECUTE_RETURN_ORDER', where)
    geometry = descriptor(b, r, before['original'], where + ':original')
    r.check(before['effective'] == before['original'] and before['executor'] == before['registration'] == 0
        and before['original']['raw'] == start['output'] and before['original']['view_bytes'] == start['output_bytes']
        and before['original']['view'] == start['shape'] and before['original']['strides'] == start['strides']
        and start['format'] == 47 and start['dtype'] == 27 and start['dimensions'] == 3,
        'BORROWED_BEFORE_IS_ACTUAL_ORIGINAL_FORMAL', where)
    for index, (value, phase, event) in enumerate(zip(values, phases, (start, gws, enter))):
        r.check(b.integer_packet(value, arrays=('layout',), nested=('original', 'effective'))
            and value['version'] == 2 and value['size'] == 656 and value['phase'] == phase and value['api'] == 1
            and value['reserved'] == value['status'] == 0 and value['device'] == start['device']
            and value['call'] == start['call'] and value['thread'] == start['thread']
            and value['event_sequence'] == event['sequence'] and b.positive(value['sequence'])
            and value['layout_length'] == 3 and value['layout'] == [84, 78, 68] + [0]*13,
            'CURRENT_TYPED_RECORD_NATIVE_REQUEST_JOIN', where + ':' + str(phase))
        effective = descriptor(b, r, value['effective'], where + ':effective:' + str(phase))
        r.check(value['original'] == before['original'] and effective == geometry, 'UNCHANGED_COMPLETE_TYPED_GEOMETRY', where)
        if index:
            r.check(value['sequence'] > values[index-1]['sequence'] and value['executor'] == event['executor']
                and value['effective']['raw'] == event['output'] and value['effective']['view_bytes'] == event['output_bytes']
                and value['effective']['view'] == event['shape'] and value['effective']['strides'] == event['strides'],
                'ACTUAL_EFFECTIVE_DESCRIPTOR_AT_GWS_AND_EXECUTE', where)
    r.check(after['registration'] == 0, 'REGISTRATION_COMES_FROM_ACTUAL_EXECUTE', where)
    if selected:
        r.check(after['effective']['descriptor'] != before['original']['descriptor']
            and values[2]['effective'] == after['effective'] and values[2]['registration'] == enter['registration'],
            'OWNED_DESCRIPTOR_STILL_IDENTICAL_BEFORE_EXECUTE', where)
    else:
        r.check(after['effective'] == before['original'], 'UNSELECTED_STOCK_DESCRIPTOR_UNCHANGED', where)
    return dict(api=before['api'], layout=before['layout'], layout_length=before['layout_length'], descriptor=geometry)


class FiaArtifacts:
    def __init__(self, b, root, *, recorded_root='/evidence'):
        self.b, self.root = b, Path(root)
        self.recorded_root = PurePosixPath(str(recorded_root).replace('\\', '/'))
        self.files = {}

    def bytes(self, artifact, records):
        recorded = PurePosixPath(artifact['path'].replace('\\', '/'))
        if not recorded.is_relative_to(self.recorded_root):
            raise ValueError('EXPLICIT_EVIDENCE_BINARY_MAPPING_REQUIRED')
        path = self.b.resolve_under(self.root, str(recorded.relative_to(self.recorded_root)))
        size = records[0]['original']['view_bytes']
        if not 0 < size <= 8*1024**2 or path.stat().st_size != artifact['bytes'] or artifact['bytes'] != size:
            raise ValueError('COMPLETE_OUTPUT_BIN_LENGTH_CHANGED')
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != artifact['sha256']:
            raise ValueError('ORIGINAL_OUTPUT_BIN_SHA_CHANGED')
        h = hashlib.sha256(json.dumps(['torch.bfloat16', records[0]['original']['view'][:3]]).encode())
        h.update(raw)
        if h.hexdigest() != artifact['tensor_sha256']:
            raise ValueError('COMPLETE_TENSOR_HEADER_AND_BYTES_SHA_CHANGED')
        if artifact['native_records_sha256'] != self.b.digest(records) or artifact['complete_bytes'] is not True:
            raise ValueError('OUTPUT_BIN_TYPED_RECORD_JOIN_CHANGED')
        self.files[str(path)] = dict(bytes=len(raw), sha256=actual)
        return raw


def typed_review(b, inventory, artifacts, *, expected_count=30):
    r = b.Review(); rows = inventory['rows']; bridges = inventory['fia_typed_stock_bridge']
    public = inventory['public_output_observer']; outcomes = inventory['outcomes']
    public_by_call = {v['independent_call']:v for v in public['calls'] if v.get('independent_call')}
    r.check(len(public_by_call) == len(public['calls']) == 7*expected_count
        and public['owned_api_wrappers'] == public['inflight_public_calls'] == public['tensor_views_created'] == 0
        and public['source_scope'] == 'ACTUAL_FIA_PUBLIC_RETURN_ONLY' and not public['refusals'], 'ALL_ACTUAL_PUBLIC_FIA_RETURNS_AND_RESTORE')
    selected = inventory['single_site_c1']['selection']['target_key']
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
        r.check(counts == Counter({1:expected_count, 2:expected_count, **({3:1} if row['plan']['mode'] else {})})
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
            geometry = r.section(loc, records, b, r, actual, events, selected=chosen, where=loc)
            if callback is None:
                r.check(index == 7 and not chosen, 'CALLBACK_MISSING_BEFORE_NATIVE_REMOVE', loc)
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
                    r.check(len(baseline) == 2 and callback['target_key'] == selected
                        and geometry == baseline[0][1], 'FRESH_SELECTED_STOCK_GEOMETRY', loc)
                    if len(baseline) != 2:
                        continue
                    if row['plan']['mode'] != 'fault':
                        r.check(raw == baseline[0][0] and callback.get('fia_stock_bridge_equal') is True,
                            'FRESH_OWNED_REFERENCE_LOSSLESS_OFF_FULL_STOCK_BYTES', loc)
            else:
                r.check('fia_full_output' not in callback, 'UNSELECTED_CALL_NOT_REPORTED_AS_NEW_REFERENCE', loc)
    r.check(seen_artifacts == bridges['artifacts'] and len(seen_artifacts) == 2*expected_count + 5
        and len(artifacts.files) == len(seen_artifacts) and len(stock) == expected_count,
        'COMPLETE_STOCK_AND_SELECTED_OUTPUT_ARTIFACT_SET')
    r.check(bridges['schema'] == 'demoncat.writer26.fia_stock_bridge.v3' and bridges['retained_host_bytes'] == 0
        and bridges['stock_target_count'] == 0 and bridges['reference_bridge_targets'] == [selected], 'STOCK_OWNER_ZERO_AND_ONLY_FRESH_REFERENCE')
    qualification = inventory['single_site_c1']['reference_lossless_off']
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
    final = inventory['single_site_c1']['typed_fia_state']
    trace_state(b, r, final, 'final')
    r.check(final == rows[-1]['fia_typed_after'], 'FINAL_TYPED_STATE_IS_LAST_HEALTH_READBACK')
    return dict(**r.summary(), typed_records=typed_count, full_output_bins_rehashed=len(artifacts.files),
        full_output_bytes_rehashed=sum(v['bytes'] for v in artifacts.files.values()), stock_targets=len(stock),
        initial_typed_records_outside_windows=rows[0]['fia_typed_before']['stored'],
        source_bound_descriptor_receipts_reconciled=not r.errors, residual_and_logits_tensor_bins_available=False,
        public_qualification_granted=False, c2=False, c3=False, e5=False)


def build_profile(b, build):
    expected = dict(schema='demoncat.writer26.fia_ncl_typed.v3', abi=dict(version=3, inventory_version=2,
        descriptor=280, record=656, state=104, lease=912, snapshots=2048), native_format=47, public_format=2,
        dtype='bfloat16', dimensions=3, storage_layouts=['MATCHED_VIEW', 'EXACT_FLAT_SPAN'], offset=0, contiguous=True,
        borrowed_original_read_scope='ACTUAL_GWS_STACK_ONLY', effective_descriptor='GROUP_OWNED_ALL_SELECTED_MODES',
        stock_to_owned_reference_full_byte_comparison_required=True, old_reference_reusable=False,
        before_execute_complete_typed_readback_required=True, native_qualified=False)
    r = b.Review()
    r.check(build['fia_typed_profile'] == expected, 'FROZEN_NEW_TYPED_BUILD_AND_NO_OLD_REFERENCE_REUSE')
    return r.summary()


def review(b, document, inventory, serving, root):
    common = b.review(document, inventory, serving, b.Artifacts(root), mechanism=2, scope=SCOPE, build_sha=BUILD,
        expected_count=30, expected_resolved=29, expected_aliases=397, expected_selection=SELECTED)
    guard = b.Review()
    typed = guard.section('typed_evidence', typed_review, b, inventory, FiaArtifacts(b, root))
    return dict(common=common, typed=typed, schema_checks=guard.summary(),
        consistent=common['consistent'] and typed is not None and typed['consistent'] and not guard.errors,
        public_qualification_granted=False, c2=False, c3=False, e5=False,
        limits=[
            'The typed stage reports output files actually rehashed and compares stock/control bytes; residual and complete logits remain saved runtime hash/Native readback receipts.',
            'Initial warmup typed records have saved state counters but are outside the eight detailed request windows.',
            'Typed descriptor identity, callback lifetime, original-call count, stream synchronization and restoration are source-bound original receipts, not a new live NPU observation.',
            'This is one selected typed FIA destination in a complete 30-call census, not all 29 resolved targets, C2, C3 or another mechanism.',
            'Independent E5 statistics reported an earlier slot_indices difference and incomparable preceding nulls; no fault-only numerical localization is granted.'])


def refusal(inventory, document):
    """Report a raw incomplete/refused run without manufacturing a success path."""
    return dict(original_first_failure=inventory.get('first_failure'), blockers=inventory.get('blockers'),
        failure_events=inventory.get('failure_events'), completed_phases=[r.get('phase') for r in inventory.get('rows', [])],
        completed_outcome_modes=[r.get('mode') for r in inventory.get('outcomes', [])],
        pending_outcomes=len(inventory.get('pending_outcomes', [])), owned_hooks=inventory.get('owned_hooks'),
        owned_public_api_wrappers=inventory.get('owned_public_api_wrappers'),
        native_state=inventory.get('native_state'), driver_error=document.get('error'), cleanup_error=document.get('cleanup_error'),
        public_qualification_granted=False, single_site_source_bound_evidence_reconciled=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-root', 'source-root', 'native-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = dict(schema='demoncat.writer26.typed_fia_single_source62.independent_review.v1', read_only=True,
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), public_qualification_granted=False, c2=False, c3=False, e5=False,
        single_site_source_bound_evidence_reconciled=False)
    try:
        b = shared(); sys.addaudithook(b.forbid_runtime)
        paths, originals = {}, {}
        for name, (relative, expected) in ORIGINALS.items():
            path = b.resolve_under(args.evidence_root, relative); actual = b.file_sha(path)
            originals[name] = dict(path=str(path), sha256=actual, bytes=path.stat().st_size)
            if actual != expected:
                raise ValueError('ORIGINAL_SOURCE62_SHA_MISMATCH:' + name)
            paths[name] = path
        report['originals'] = originals
        document, inventory, serving = (b.read(paths[name]) for name in ('result', 'inventory', 'serving'))
        report['provenance'] = b.provenance(document, inventory, args.source_root, args.native_root,
            pins=dict(source_commit=SOURCE, source_manifest=MANIFEST, build=BUILD), extra_sources=EXTRA_SOURCES,
            setup_sources=(*b.SETUP_SOURCES, EXTRA_SOURCES[0]))
        report['typed_build'] = build_profile(b, b.read(b.resolve_under(args.native_root, 'build.json')))
        if inventory.get('first_failure') or inventory.get('blockers') or len(inventory.get('rows', [])) != 8:
            report.update(status='ORIGINAL_RUN_REFUSED_OR_INCOMPLETE_NO_QUALIFICATION', refusal=refusal(inventory, document))
        else:
            report['evidence'] = review(b, document, inventory, serving, args.evidence_root)
            good = all(report[key]['consistent'] for key in ('provenance', 'typed_build', 'evidence'))
            report.update(single_site_source_bound_evidence_reconciled=good,
                status='SOURCE_BOUND_SINGLE_TYPED_FIA_RECONCILED_NOT_C2_C3_E5' if good else 'EVIDENCE_GAP_OR_CONTRADICTION')
    except (OSError, KeyError, ValueError, TypeError, IndexError, OverflowError) as error:
        report.update(status='EVIDENCE_GAP_OR_CONTRADICTION', error=type(error).__name__ + ':' + str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['single_site_source_bound_evidence_reconciled'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
