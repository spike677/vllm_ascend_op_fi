"""Read-only, stdlib-only reconciliation of source54's single QMM experiment.

This consumes saved evidence, never imports the runtime or treats a serialized
qualification flag as authority. Native/source receipts and absent tensor bins
remain explicitly distinguished from independently rehashed artifact bytes.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys


SOURCE_COMMIT = 'ce82ca5f26f4c7326caa69af654c1755b013a888'
SOURCE_MANIFEST = 'e0a02e6c5db6869bffbecd395ee9cf36fb9aac29f53c4f4c23b6d81648d9bd74'
BUILD_SHA = 'b81b8b9bb1f0b5c0e56109775a987bf503b88a00adcfb873964a258e8bf4e03e'
RESULT_SHA = 'c933c19f810be88fb29a04266af03a134d5ce683cd423a26abbd9b7bb36dd2e5'
INVENTORY_SHA = 'a545a0044e99ddb7f38ea9cc2bb67a6cb697f248082e39afc3816eba67f0236f'
SERVING_SHA = '7db70e3822f09021e36d694060ac9d066ca57b3c9fd1989b2e803a4d70d35d51'
TRACE_SHA = 'df36a932fe10e25818e2b485fef74eaf04724e29712e3a93b01cd4497f3aabc7'
WRITER_SHA = '69fb4feafb41b078a874f462041d41ff5832faf9ba2e5d12ca2401f8d501273c'
SELECTED_KEY = 'c6f6a9afc52579cd2d4ea232dc04550fc820d0e4ec3e6513219385458f2d559e'
SCOPE = 'ONE_ACTUAL_QMM_SITE_FIXED_FRESH_PREFILL_NOT_ALL_ELIGIBLE_NOT_C2_C3'
PHASES = ('inventory_1', 'inventory_2', 'single_site_reference', 'single_site_lossless',
          'single_site_qualification_off', 'single_site_fault', 'single_site_recovery_off', 'health')
MODES = (None, None, 'reference', 'lossless', 'off', 'fault', 'off', None)
MODE_IDS = dict(reference=1, lossless=2, fault=3, off=4)
KINDS = {1:'qmm_enter', 2:'qmm_exit', 3:'fia_before', 4:'fia_gws', 5:'fia_enter',
         6:'fia_exit', 7:'add_rms_gws', 8:'add_rms_enter', 9:'add_rms_exit', 10:'qmm_registration'}
STABLE_NATIVE = ('cache', 'registration', 'device', 'dtype', 'format', 'dimensions',
    'residual_dimensions', 'output_bytes', 'residual_bytes', 'shape', 'strides', 'residual_shape', 'residual_strides')
SYMBOLS = {'libfault26_readonly_trace_abi0.so':'dcat26_trace_metadata_arrays_v1',
    'libopapi.so':'dcat20_metadata_arrays_v1', 'libopapi_nn.so':'dcat20_metadata_arrays_v1',
    'libdcat26_group_v2.so':'dcat26_group_metadata_arrays_v1'}
COUNTERS = ('capture_calls', 'active_captures', 'getter_calls', 'active_getters', 'getter_failures',
    'getter_exceptions', 'capture_exceptions', 'acquired_arrays', 'released_arrays', 'live_arrays',
    'peak_live_arrays', 'ownership_errors', 'revision')
ZERO_COUNTERS = ('active_captures', 'active_getters', 'getter_failures', 'getter_exceptions',
                 'capture_exceptions', 'live_arrays', 'ownership_errors')
MONOTONIC = tuple(k for k in COUNTERS if k not in ('active_captures', 'active_getters', 'live_arrays'))
SOURCES = ('experiments/c2c3_closure_20261117/providers26_group_probe.py',
    'fault_injection/deployments/writer26_group.py', 'fault_injection/deployments/writer26_experiment.py',
    'fault_injection/deployments/writer26_inventory.py', 'fault_injection/deployments/writer26_public.py',
    'fault_injection/deployments/writer26_loader.py', 'fault_injection/deployments/residual26.py',
    'fault_injection/metadata_arrays.py', 'fault_injection/native_adapters/writer26_group_core.h',
    'fault_injection/native_adapters/writer26_group_runtime.cpp',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving26_group.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving26_inventory.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving_context.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving20.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/call_binding.py')
SETUP_SOURCES = (SOURCES[10], SOURCES[1], SOURCES[2], SOURCES[4])


def digest(value):
    # The producer uses json.dumps DEFAULT separators, including spaces.
    h = hashlib.sha256()
    for block in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(value):
        h.update(block.encode())
    return h.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('DUPLICATE_JSON_KEY:' + key)
        value[key] = item
    return value


def read(path):
    path = Path(path)
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError('BOUNDED_JSON_REQUIRED')
    def invalid(value):
        raise ValueError('NONFINITE_JSON:' + value)
    with path.open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique_pairs, parse_constant=invalid)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def positive(value):
    return type(value) is int and value > 0


def integer_packet(value, arrays=(), nested=()):
    return (all(type(v) is int for k, v in value.items() if k not in (*arrays, *nested))
        and all(isinstance(value[k], list) and all(type(v) is int for v in value[k]) for k in arrays))


def words(value):
    if not valid_sha(value):
        raise ValueError('FULL_TARGET_SHA_REQUIRED')
    return [int(value[i:i+16], 16) for i in range(0, 64, 16)]


def resolve_under(root, relative):
    item = PurePosixPath(relative)
    if item.is_absolute() or '..' in item.parts or '\\' in str(relative):
        raise ValueError('ARTIFACT_PATH_TRAVERSAL')
    root = Path(root).resolve(strict=True)
    path = (root / Path(*item.parts)).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('ARTIFACT_ESCAPES_ROOT')
    return path


class Artifacts:
    """Only explicit /evidence mapping; no glob search or name-only fallback."""
    def __init__(self, root):
        self.root, self.cache = Path(root), {}

    def allocator(self, reference):
        recorded = PurePosixPath(reference['path'])
        if not recorded.is_relative_to('/evidence'):
            raise ValueError('EXPLICIT_EVIDENCE_MAPPING_REQUIRED')
        path = resolve_under(self.root, str(recorded.relative_to('/evidence')))
        expected = reference['sha256']
        if not valid_sha(expected) or not valid_sha(reference['complete_snapshot_sha256']):
            raise ValueError('ALLOCATOR_HASH_MISSING')
        key = (str(path), expected)
        if key not in self.cache:
            if file_sha(path) != expected:
                raise ValueError('ALLOCATOR_ARTIFACT_CHANGED')
            self.cache[key] = read(path)
        return self.cache[key]


class Review:
    def __init__(self):
        self.checks, self.errors, self.examples = 0, Counter(), {}

    def check(self, value, code, where=''):
        self.checks += 1
        if not value:
            self.errors[code] += 1
            examples = self.examples.setdefault(code, [])
            if len(examples) < 3:
                examples.append(str(where)[:180])
        return bool(value)

    def section(self, context, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, ValueError, IndexError, OSError, OverflowError) as error:
            self.check(False, 'SCHEMA_OR_ARTIFACT_UNAVAILABLE', context + ':' + type(error).__name__ + ':' + str(error))
            return None

    def summary(self):
        return dict(checks=self.checks, consistent=not self.errors,
                    issues=[dict(code=k, count=n, examples=self.examples[k]) for k, n in sorted(self.errors.items())])


def target_key(callback):
    call = callback['native_packet']
    signature = {name:call[name] for name in ('mechanism', 'device', 'dtype', 'format', 'dimensions',
        'output_bytes', 'residual_bytes', 'shape', 'cache', 'registration')}
    return digest(dict(module=callback['module_object_identity'], method=callback['method'], signature=signature))


def contiguous(shape, stride):
    if not shape or len(shape) != len(stride) or not all(positive(n) for n in shape):
        return False
    step = 1
    for size, actual in zip(reversed(shape), reversed(stride)):
        if type(actual) is not int or (size != 1 and actual != step):
            return False
        step *= size
    return True


def public_tensor(meta, event, residual=False):
    prefix = 'residual_' if residual else ''
    address, size = event['residual' if residual else 'output'], event['residual_bytes' if residual else 'output_bytes']
    dims = event[prefix+'dimensions']
    return (positive(meta['object_id']) and positive(meta['storage_identity']) and 0 < dims <= 8
        and meta['storage_offset'] == 0 and meta['storage_addr'] == address and meta['storage_bytes'] == size
        and meta['shape'] == event[prefix+'shape'][:dims]
        and meta['stride'] == event['residual_strides' if residual else 'strides'][:dims]
        and contiguous(meta['shape'], meta['stride']) and math.prod(meta['shape'])*meta['element_size'] == size
        and meta['dtype'] == {0:'torch.float32', 1:'torch.float16', 27:'torch.bfloat16'}[event['dtype']]
        and meta['element_size'] == (4 if event['dtype'] == 0 else 2)
        and meta['actual_format'] == event['format'] and meta['device'] == 'npu:' + str(event['device']))


def allocation(snapshot, call):
    """Rebuild active allocation spans from saved relevant segments."""
    if not isinstance(snapshot, list) or not 0 < len(snapshot) <= 2:
        raise ValueError('RELEVANT_ALLOCATOR_SEGMENTS_REQUIRED')
    blocks = []
    for index, segment in enumerate(snapshot):
        cursor = segment['address']
        if not positive(cursor) or not segment['blocks']:
            raise ValueError('INVALID_ALLOCATOR_SEGMENT')
        segment_blocks = []
        for block in segment['blocks']:
            start, size = block.get('address', cursor), block['size']
            if not positive(start) or not positive(size) or start != cursor:
                raise ValueError('ALLOCATOR_BLOCK_GAP_OR_OVERLAP')
            segment_blocks.append((start, start+size, block['state'], index))
            cursor = start + size
        blocks.extend(segment_blocks)
    ordered = sorted(blocks)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('OVERLAPPING_ALLOCATOR_RECORDS')
    matches = [[b for b in blocks if b[2] == 'active_allocated' and b[0] <= address
                and address + size <= b[1]] for address, size in
               ((call['residual'], call['residual_bytes']), (call['output'], call['output_bytes']))]
    if any(len(v) != 1 for v in matches):
        return None, None
    residual, output = matches[0][0], matches[1][0]
    nonadjacent = [[list(residual[:3])], [list(output[:3])]] if residual != output and residual[1] != output[0] else None
    if residual != output and residual[3] == output[3] and residual[1] == output[0] == call['output']:
        padding = residual[1] - call['residual'] - call['residual_bytes']
        if 0 <= padding and padding + 32 <= 4096:
            return dict(residual_block=list(residual[:2]), output_block=list(output[:2]), padding=padding, shift=padding+32), None
    return None, nonadjacent


def trace(r, row, lifetime, where):
    before, after, events = row['native_before'], row['native_after'], row['native_events']
    for value in (before, after):
        r.check(integer_packet(value) and value['version'] == 1 and value['size'] == 160 and value['flags'] == 7 and value['reserved'] == 0
            and value['capacity'] == 32768 and value['overflow_events'] == value['errors'] == 0
            and value['stored_events'] == value['total_events'] <= value['capacity']
            and value['all_runs'] == value['qmm_runs'] + value['non_target_runs'], 'COMPLETE_NATIVE_COUNTERS', where)
    r.check(all(after[k] >= before[k] for k in before if k not in ('version', 'size', 'flags', 'reserved')),
            'NATIVE_COUNTER_RESET', where)
    r.check(len(events) == after['stored_events'] - before['stored_events'], 'NATIVE_EVENT_DENOMINATOR', where)
    for offset, event in enumerate(events, before['stored_events'] + 1):
        r.check(integer_packet({k:v for k,v in event.items() if k != 'kind_name'},
            arrays=('shape', 'strides', 'residual_shape', 'residual_strides'))
            and all(len(event[k]) == 8 for k in ('shape', 'strides', 'residual_shape', 'residual_strides'))
            and event['version'] == 1 and event['size'] == 416 and event['kind_name'] == KINDS[event['kind']]
            and event['sequence'] == offset and event['thread'] == lifetime['native_thread'], 'NATIVE_EVENT_IDENTITY_OR_GAP', where)
    counts = Counter(e['kind_name'] for e in events)
    r.check(all(counts[k] == after[k] - before[k] for k in KINDS.values()), 'NATIVE_DETAIL_COUNTER_JOIN', where)
    r.check(counts['qmm_enter'] == after['qmm_runs'] - before['qmm_runs'], 'FULL_QMM_DENOMINATOR', where)
    starts = [e for e in events if e['kind_name'] == 'qmm_enter']
    r.check(len({e['call'] for e in starts}) == len(starts) and all(positive(e['call']) for e in starts),
            'NATIVE_CALL_DUPLICATION', where)
    return starts


def correlate(row):
    events, callbacks, producers = row['native_events'], row['callbacks'], row['producers']
    callback_map = {c['independent_call']:c for c in callbacks if c.get('independent_call')}
    result = []
    for native in (e for e in events if e['kind_name'] == 'qmm_enter'):
        callback, reasons = callback_map.get(native['call']), []
        if callback is None:
            reasons.append('NATIVE_WRITER_HAS_NO_OWNED_PUBLIC_CALLBACK')
        else:
            reasons.extend(callback['errors'])
        endings = [e for e in events if e['kind_name'] == 'qmm_exit' and e['call'] == native['call']]
        if len(endings) != 1 or endings[0]['status'] != 0:
            reasons.append('ACTUAL_WRITER_EXECUTION_NOT_COMPLETED')
        public = [p for p in producers if p.get('native_producer') == native['producer']
                  and p.get('output', {}).get('storage_addr') == native['residual']]
        if len(public) != 1:
            reasons.append('ACTUAL_PUBLIC_PRODUCER_LIFETIME_UNPROVEN')
        else:
            reasons.extend(public[0]['errors'])
        consumers = [e for e in events if e['kind_name'] == 'add_rms_exit' and e['sequence'] > native['sequence']
                     and e['residual'] == native['residual'] and e['residual_bytes'] == native['residual_bytes']]
        if not consumers or consumers[0]['status'] or consumers[0]['producer'] != native['producer'] + 1:
            reasons.append('NEXT_ACTUAL_NATIVE_CONSUMER_UNPROVEN')
        result.append(dict(native=native, callback=callback, execution=endings, public_producer=public,
            next_consumer=consumers[:1], reasons=reasons,
            eligibility='UNRESOLVED' if reasons else 'OBSERVED_WRITER_LIFETIME_NOT_LOSSLESS_QUALIFIED',
            native_qualified=False, c2=False, c3=False))
    return result


def classify(candidate):
    native, callback = candidate['native'], candidate['callback']
    if native['residual'] == native['residual_bytes'] == native['producer'] == 0 and native['flags'] in (1, 5):
        return 'UNSUPPORTED', ['NO_ACTUAL_PRECEDING_RESIDUAL_IN_THIS_FRESH_FORWARD']
    reasons = list(candidate['reasons'])
    if callback is None:
        return 'UNVERIFIED', reasons
    allowed = {'LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT', 'PUBLIC_WRITER_OUTPUT_NOT_RETAINED', 'PUBLIC_CONSUMER_LIFETIME_NOT_PROVEN'}
    if callback.get('proven_nonadjacent') and set(reasons).issubset(allowed):
        return 'UNSUPPORTED', ['LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT']
    if not callback.get('writer_output_retained'):
        reasons.append('PUBLIC_WRITER_OUTPUT_NOT_RETAINED')
    if not callback.get('public_consumer_lifetime'):
        reasons.append('PUBLIC_CONSUMER_LIFETIME_NOT_PROVEN')
    return ('UNVERIFIED', sorted(set(reasons))) if reasons else ('OBSERVED_NOT_QUALIFIED', [])


def freeze(left, right):
    counts = [Counter(c['callback'].get('target_key') for c in row if c['callback']) for row in (left, right)]
    result = []
    for index, (a, b) in enumerate(zip(left, right)):
        status, reasons = classify(b)
        previous_status, _ = classify(a)
        callback, previous = b['callback'], a['callback']
        key = callback.get('target_key') if callback else None
        if status != previous_status or (status == 'OBSERVED_NOT_QUALIFIED'
                and (previous is None or previous.get('target_key') != key or counts[1][key] != 1)):
            status, reasons = 'UNVERIFIED', ['HEALTHY_TARGET_LIFETIME_OR_OCCURRENCE_CHANGED']
        if status == 'OBSERVED_NOT_QUALIFIED' and (counts[0][key] != 1 or any(a['native'][k] != b['native'][k] for k in STABLE_NATIVE)):
            status, reasons = 'UNVERIFIED', ['HEALTHY_SINGLE_SITE_OCCURRENCE_OR_NATIVE_FORMAL_CHANGED']
        result.append(dict(index=index, target_key=key, status=status, reasons=reasons,
                           first_native=a['native'], second_native=b['native'], callback=callback))
    return result


def public_producer(r, producer, row, aliases, where):
    r.check(producer['request_id'] == row['request_id'] and producer['original_calls'] == 1
        and (producer['module_path'], producer['module_object_identity']) in aliases, 'ACTUAL_PUBLIC_PRODUCER_SCOPE', where)
    # An unresolved producer remains in the census; it cannot qualify a target.
    events = producer.get('native_events', [])
    r.check(all(e in row['native_events'] for e in events), 'PRODUCER_NATIVE_EVENT_NOT_IN_REQUEST', where)
    if not producer['errors']:
        ends = [e for e in events if e['kind_name'] == 'add_rms_exit']
        r.check(len(ends) == 1 and ends[0]['status'] == 0, 'PUBLIC_PRODUCER_EXECUTION', where)
        if len(ends) == 1:
            e = ends[0]
            r.check(producer['native_producer'] == e['producer'] and public_tensor(producer['output'], e)
                and public_tensor(producer['consumed'], e, True), 'PUBLIC_PRODUCER_FULL_STORAGE_FORMALS', where)


def callback_proof(r, candidate, row, lifetime, aliases, artifacts, where):
    callback, native = candidate['callback'], candidate['native']
    if callback is None:
        return
    call = callback['native_packet']
    r.check(integer_packet(call, arrays=('shape',)) and len(call['shape']) == 4
        and call['version'] == 2 and call['size'] == 144 and call['mechanism'] == 1 and call['reserved'] == 0
        and call['thread'] == lifetime['native_thread'] and positive(call['token']), 'ACTUAL_QMM_CALL_ABI', where)
    r.check(callback['original_continuations'] == 1 and callback['claimed'] in (0, 1), 'ORIGINAL_WRITER_NOT_EXACTLY_ONCE', where)
    r.check(callback['token'] == call['token'] and callback['independent_call'] == call['independent_call'] == native['call']
        and callback['independent_event'] == native and call['producer_sequence'] == native['producer']
        and all(call[k] == native[k] for k in ('output', 'output_bytes', 'residual', 'residual_bytes', 'cache', 'registration', 'device', 'dtype', 'format'))
        and call['dimensions'] == native['dimensions'] <= 4 and call['shape'] == native['shape'][:4], 'NATIVE_TO_PUBLIC_CALLBACK_JOIN', where)
    r.check(callback['actual_request_id'] == row['request_id'] and
        (callback['module_path'], callback['module_object_identity']) in aliases, 'CALLBACK_ACTUAL_REQUEST_MODULE', where)
    method = callback['method']
    r.check(positive(method['callable_identity']) and valid_sha(method['source_sha256']) and valid_sha(method['code_sha256'])
        and PurePosixPath(method['path']).is_absolute() and isinstance(method['module_class'], str)
        and callback['target_key'] == target_key(callback), 'TARGET_KEY_SOURCE_OBJECT_RECOMPUTATION', where)
    if callback.get('allocator_segments'):
        actual, nonadjacent = allocation(artifacts.allocator(callback['allocator_segments']), call)
        if callback.get('allocation') is not None:
            r.check(actual == callback['allocation'], 'ACTUAL_ALLOCATOR_ADJACENCY', where)
        if callback.get('proven_nonadjacent'):
            r.check(actual is None and nonadjacent == callback['proven_nonadjacent'], 'UNSUPPORTED_NONADJACENCY_NOT_PROVEN', where)
    elif callback.get('allocation') or callback.get('proven_nonadjacent'):
        r.check(False, 'ALLOCATOR_ORIGINAL_ARTIFACT_MISSING', where)
    output = callback.get('writer_output_retained')
    if output:
        r.check(positive(output['tensor_object_identity']) and output['storage_address'] == call['output']
            and output['bytes'] == call['output_bytes'] and contiguous(output['shape'], output['stride'])
            and math.prod(output['shape'])*2 == call['output_bytes'] and output['dtype'] == 'torch.bfloat16',
            'PUBLIC_WRITER_OUTPUT_FULL_STORAGE', where)
    if callback.get('public_consumer_lifetime'):
        producers, consumers = candidate['public_producer'], candidate['next_consumer']
        r.check(len(producers) == len(consumers) == 1 and bool(output), 'PUBLIC_CONSUMER_WITNESS_MISSING', where)
        if len(producers) != 1 or len(consumers) != 1 or not output:
            return
        p, end, binding = producers[0], consumers[0], callback['public_consumer_binding']
        later = [c for c in row['producers'] if c.get('native_producer') == end['producer'] and end in c.get('native_events', [])]
        r.check(len(later) == 1, 'NEXT_CONSUMER_PUBLIC_CALL_MISSING', where)
        if len(later) != 1:
            return
        c = later[0]
        r.check(not p['errors'] and not c['errors'] and c['original_calls'] == 1 and p['original_calls'] == 1
            and p['request_id'] == c['request_id'] == row['request_id'], 'PRODUCER_CONSUMER_REQUEST_OR_ERRORS', where)
        r.check(max(e['sequence'] for e in p['native_events']) < native['sequence'] < candidate['execution'][0]['sequence'] < end['sequence']
            and end['producer'] == native['producer'] + 1 and end['status'] == 0, 'PRODUCER_WRITER_CONSUMER_ORDER', where)
        r.check(callback['producer_tensor_object_identity'] == binding['producer_tensor_object_identity'] == p['output']['object_id']
            and binding['consumer_tensor_object_identity'] == c['consumed']['object_id']
            and binding['residual_storage_identity'] == p['output']['storage_identity'] == c['consumed']['storage_identity']
            and binding['writer_tensor_object_identity'] == output['tensor_object_identity']
            and binding['same_residual_storage'] is True and binding['writer_retained_through_actual_consumer'] is True
            and c['consumed_retained_producer'] == call['producer_sequence']
            and public_tensor(p['output'], native, True) and public_tensor(c['consumed'], native, True),
            'PUBLIC_PRODUCER_WRITER_CONSUMER_OBJECT_STORAGE_JOIN', where)
    if classify(candidate)[0] == 'OBSERVED_NOT_QUALIFIED':
        r.check(callback.get('current_candidate_geometry_eligible') is True and bool(callback.get('allocation'))
            and not callback['errors'] and bool(callback.get('public_consumer_binding')), 'RESOLVED_TARGET_COMPLETE_PROOFS', where)


def metadata(r, snapshots, lifetime, build):
    previous, identities, reports = None, None, []
    for label, value in snapshots:
        r.check(value['schema'] == 'demoncat.metadata_arrays.v1' and value['source'] == 'ACTUAL_LOADED_PROXY_COUNTERS'
            and value['pid'] == lifetime['pid'] and value['build_profile_sha256'] == value['build_manifest_sha256'] == build
            and value['scope'] == 'SDK_GETTER_ARRAYS_ONLY' and value['reset_supported'] is False, 'METADATA_ARRAY_REPORT_IDENTITY', label)
        images = {i['name']:i for i in value['images']}
        r.check(len(images) == len(value['images']) == 4 and set(images) == set(SYMBOLS), 'COMPLETE_FOUR_METADATA_DSOS', label)
        current, instance_ids = {}, set()
        for name, image in images.items():
            state = image['state']
            r.check(state['version'] == 1 and state['size'] == 120 and positive(state['instance'])
                and all(type(state[k]) is int and 0 <= state[k] < 2**64 for k in COUNTERS), 'METADATA_ARRAY_ABI', label+':'+name)
            r.check(state['instance'] not in instance_ids, 'METADATA_SHARED_OR_DUPLICATE_INSTANCE', label+':'+name)
            instance_ids.add(state['instance'])
            r.check(all(state[k] == 0 for k in ZERO_COUNTERS) and state['acquired_arrays'] == state['released_arrays']
                and state['peak_live_arrays'] <= state['acquired_arrays'], 'METADATA_ARRAY_LEAK_OR_FAILURE', label+':'+name)
            r.check(image['symbol'] == SYMBOLS[name] and valid_sha(image['sha256']) and bool(image['mapped_paths'])
                and len(set(image['mapped_paths'])) == len(image['mapped_paths'])
                and image['symbol_providers'] and set(image['symbol_providers']).issubset(image['mapped_paths'])
                and PurePosixPath(image['path']).name == name and all(PurePosixPath(p).is_absolute()
                    for p in [image['path'], *image['mapped_paths'], *image['symbol_providers']]), 'METADATA_ACTUAL_IMAGE_PROVIDER_JOIN', label+':'+name)
            current[name] = {k:v for k, v in image.items() if k != 'state'} | {'instance':state['instance']}
            if previous is not None and name in previous:
                r.check(all(state[k] >= previous[name]['state'][k] for k in MONOTONIC), 'METADATA_ARRAY_COUNTER_RESET', label+':'+name)
        if identities is not None:
            r.check(current == identities, 'METADATA_DSO_OR_INSTANCE_CHANGED', label)
        identities, previous = current, images
        reports.append(images)
    initial, latest = reports[0], reports[-1]
    delta = {k:sum(latest[n]['state'][k] - initial[n]['state'][k] for n in SYMBOLS)
             for k in ('capture_calls', 'getter_calls', 'acquired_arrays', 'released_arrays')}
    r.check(all(n > 0 for n in delta.values()) and delta['acquired_arrays'] == delta['released_arrays'], 'METADATA_ACTUAL_INTERVAL_ACTIVITY')
    return dict(images=4, interval_delta=delta, final_live_arrays=sum(i['state']['live_arrays'] for i in latest.values()))


def input_hash(tokens, positions, meta):
    values = []
    for name, numbers in (('input_ids', tokens), ('positions', positions)):
        tensor = meta[name]
        if (tensor['dtype'] not in (('torch.int32', 'torch.int64') if name == 'input_ids' else ('torch.int64',))
                or tensor['shape'] != [len(numbers)] or tensor['stride'] != [1]
                or tensor['storage_offset'] != 0 or not positive(tensor['object_id'])
                or not positive(tensor['storage_identity']) or tensor['device'] != 'npu:0'):
            raise ValueError('FROZEN_ACTUAL_INPUT_INTEGER_VECTOR_REQUIRED')
        h = hashlib.sha256(json.dumps([tensor['dtype'], tensor['shape']]).encode())
        h.update(struct.pack('<' + ('i' if tensor['dtype'] == 'torch.int32' else 'q')*len(numbers), *numbers))
        values.append(h.hexdigest())
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def requests(r, document, inventory, serving):
    rows, phases, observations = inventory['rows'], document['phases'], serving['requests']
    r.check(len(rows) == len(phases) == len(observations) == 8, 'EXACT_EIGHT_REQUESTS')
    r.check(document['sessions']['candidate']['phases'] == phases and serving['native_phases'] == rows,
            'DRIVER_SERVING_INVENTORY_COMPLETE_ROW_JOIN')
    r.check(serving['qualification'] == inventory, 'SERVING_FINAL_INVENTORY_JOIN')
    r.check(set(document['sessions']) == {'candidate'} and document['sessions']['candidate']['trigger'] == 'once', 'SINGLE_SESSION_NO_FALLBACK')
    tokens, internal, external = document['actual_prompt_token_ids'], set(), set()
    r.check(len(tokens) > 1 and all(type(n) is int and n >= 0 for n in tokens), 'ACTUAL_PROMPT_TOKEN_VECTOR')
    baseline = rows[0]['output_sha256']
    for index, (phase, row, observed) in enumerate(zip(phases, rows, observations)):
        label = PHASES[index]
        mapping, snapshot = phase['mapping'], observed['snapshot']
        rid = mapping['internal_request_id']
        r.check(phase['phase'] == row['phase'] == observed['phase'] == label and phase['index'] == index
            and phase['session'] == 'candidate', 'PHASE_ORDER', label)
        r.check(isinstance(rid, str) and rid and rid not in internal and mapping['external_request_id'] not in external
            and positive(mapping['request_state_object_id']) and mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
            and mapping['proof_scope'] == 'DRIVER_MAPPING_ONLY_REQUIRES_REAL_MODEL_PREHOOK_CORROBORATION'
            and rid == row['request_id'] == observed['actual_internal_request_id'] == snapshot['internal_request_id'], 'REAL_DRIVER_SCHEDULER_REQUEST_JOIN', label)
        internal.add(rid); external.add(mapping['external_request_id'])
        r.check(positive(snapshot['request_object_id']) and snapshot['num_requests'] == 1 and snapshot['num_computed_tokens'] == 0
            and snapshot['num_prompt_tokens'] == snapshot['num_scheduled_tokens'] == len(tokens)
            and snapshot['token_ids'] == tokens and snapshot['positions'] == list(range(len(tokens)))
            and snapshot['token_ids_sha256'] == hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
            and snapshot['device_input_bytes_equality'] == 'COMPLETE_DEVICE_READBACK_VERIFIED'
            and snapshot['attention_metadata_keys'], 'ACTUAL_FRESH_PREFILL_INPUT', label)
        r.check(row['input_sha256'] == observed['input_sha256'] == rows[0]['input_sha256']
            == input_hash(tokens, snapshot['positions'], snapshot['input_tensor_metadata']), 'COMPLETE_INPUT_DIGEST_RECOMPUTATION', label)
        r.check(observed['root_returned'] is True and observed['output_shape'] == serving['boundary']['output_contract']
            == [1, serving['boundary']['vocab_size']] and positive(serving['boundary']['vocab_size'])
            and observed['model_output_sha256'] == row['output_sha256'] and valid_sha(row['output_sha256'])
            and observed['model_output_finite'] is True and row['output_finite'] is True, 'COMPLETE_ACTUAL_VOCAB_RETURN', label)
        r.check(phase['generated_tokens'] == [observed['top1']] == document['healthy_before_tokens'] == document['healthy_after_tokens'],
                'ACTUAL_GENERATION_AND_POST_RELEASE_HEALTH', label)
        r.check((row['output_sha256'] != baseline) if MODES[index] == 'fault' else (row['output_sha256'] == baseline),
                'FULL_LOGITS_EFFECT_OR_HEALTHY_OFF_MISMATCH', label)
    return dict(count=len(rows), input_sha256=rows[0]['input_sha256'], healthy_logits_sha256=baseline,
        fault_logits_sha256=rows[5]['output_sha256'], generated_tokens=document['healthy_after_tokens'],
        post_native_remove_full_logits=True, post_python_remove_full_logits=False)


def outcome(r, value, candidate, row, qualification, where):
    call, lease = value['native_call'], value['lease']
    callback, selected = candidate['callback'], qualification['target_key']
    r.check(value['target'] == selected == callback['target_key'] and value['request_id'] == row['request_id']
        and call == callback['native_packet'] and value['mode'] == row['plan']['mode']
        and callback['claimed'] == 1 and callback.get('single_site_current_binding_verified') is True, 'OUTCOME_ACTUAL_SELECTED_CALL', where)
    r.check(integer_packet(lease, arrays=('target',), nested=('expected',)) and len(lease['target']) == 4
        and lease['version'] == 2 and lease['size'] == 248 and lease['reserved'] == 0
        and lease['mode'] == MODE_IDS[value['mode']] and lease['target'] == words(selected)
        and [lease['owner_hi'], lease['owner_lo']] == qualification['native_owner']
        and lease['generation'] == qualification['generation'] and lease['thread'] == call['thread']
        and lease['expected'] == call, 'LEASE_OWNER_GENERATION_ACTUAL_FORMALS', where)
    alloc = callback['allocation']
    r.check(lease['residual_allocation_end'] == alloc['residual_block'][1]
        and lease['output_allocation_begin'] == alloc['output_block'][0] == call['output']
        and lease['shift_bytes'] == alloc['shift'] == alloc['padding'] + 32 <= 4096,
        'LEASE_ACTUAL_ADJACENT_ALLOCATION', where)
    observed, receipt, released = (value[k] for k in ('observed_consumer_receipt', 'consumer_receipt', 'released_receipt'))
    r.check(observed == receipt and released == dict(receipt, released=1) and receipt['released'] == 0,
            'FIRST_NATIVE_RECEIPT_AND_ACK_ONLY_RELEASE', where)
    for label, p in (('consumer', receipt), ('released', released)):
        r.check(integer_packet(p, arrays=('target',)) and len(p['target']) == 4
            and p['version'] == 2 and p['size'] == 296 and p['reserved'] == p['error'] == p['writer_status'] == p['consumer_status'] == 0
            and p['mode'] == lease['mode'] and p['mechanism'] == 1 and p['target'] == lease['target']
            and [p['owner_hi'], p['owner_lo']] == qualification['native_owner'] and p['generation'] == qualification['generation']
            and all(p[k] == call[k] for k in ('token', 'independent_call', 'registration', 'cache', 'producer_sequence',
                'output', 'output_bytes', 'residual', 'residual_bytes')) and positive(p['stream']), 'NATIVE_RECEIPT_IDENTITY', where+':'+label)
        r.check(p['writer_calls'] == p['consumer_calls'] == 1 and all(p[k] == 1 for k in
            ('writer_complete', 'consumer_complete', 'output_readback_exact', 'padding_readback_exact', 'residual_restored_exact')),
            'NATIVE_EXECUTION_CONSUMPTION_RECOVERY', where+':'+label)
    mode = value['mode']
    if mode == 'fault':
        r.check(0 < receipt['changed_residual_bytes'] <= 32 and receipt['residual_unchanged'] == 0
            and receipt['replacement'] == call['output'] - alloc['shift']
            == call['residual'] + call['residual_bytes'] - 32 and receipt['lossless_qualified'] == 1,
            'QMM_ACTUAL_32_BYTE_RESIDUAL_EFFECT', where)
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


def runtime_state(r, state, qualification, thread, callbacks, admissions, *, removed, sealed, where):
    core = state['core']
    r.check(integer_packet(state, nested=('core',)) and integer_packet(core)
        and state['version'] == 2 and state['size'] == 240 and state['reserved'] == 0
        and core['version'] == 2 and core['size'] == 136 and core['mechanism'] == 1
        and core['reserved'] == core['reserved2'] == 0 and core['error'] == 0
        and [core['owner_hi'], core['owner_lo']] == qualification['native_owner']
        and core['generation'] == qualification['generation'] and core['thread'] == thread, 'NATIVE_RUNTIME_OWNER_ABI', where)
    r.check(all(state[k] == 0 for k in ('callback_active', 'completion_active', 'prepared_fia', 'fia_executor',
        'descriptor', 'rejected_calls', 'foreign_calls', 'last_rejection')) and all(core[k] == 0 for k in
        ('pending', 'executing', 'device_bytes', 'descriptor_pending')), 'NATIVE_QUIESCENT_NO_UNCERTAIN_REPLIES', where)
    r.check(state['callback_calls'] == callbacks and state['admitted_calls'] == state['completion_calls'] == core['calls'] == admissions
        and state['passthrough_calls'] == callbacks - admissions, 'NATIVE_CALLBACK_ADMISSION_COUNTS', where)
    r.check(core['installed'] == int(not removed) and core['stopped'] == int(removed)
        and core['sealed'] == int(sealed) and core['targets'] == int(admissions > 0 and not removed), 'NATIVE_LIFECYCLE_ORDER', where)
    if removed or not admissions:
        r.check(core['host_bytes'] == 0, 'NATIVE_HOST_RESOURCES_NOT_ZERO', where)
    else:
        r.check(positive(core['host_bytes']), 'NATIVE_QUALIFIED_REFERENCE_BYTES_MISSING', where)


def review(document, inventory, serving, artifacts, *, expected_count=135, expected_aliases=728,
           expected_selection=(SELECTED_KEY, 83), expected_resolved=2):
    r, details = Review(), {}
    rows, lifetime = inventory['rows'], inventory['lifetime']
    aliases = inventory['alias_inventory']
    alias_pairs = {(a['path'], a['object_identity']) for a in aliases}
    r.check(inventory['schema'] == 'demoncat.writer26.candidate_experiment.v2' and inventory['stage'] == document['stage'] == 'c1_single_effects'
        and inventory['mechanism'] == document['mechanism'] == '26.residual_stomp'
        and inventory['scope'] == document['scope'] == SCOPE, 'EXACT_SINGLE_QMM_SCOPE')
    r.check(len(aliases) == expected_aliases and len({a['path'] for a in aliases}) == len(aliases)
        and all(positive(a['object_identity']) and isinstance(a['path'], str) and a['path'] for a in aliases)
        and ('<root>', lifetime['model_object_identity']) in alias_pairs, 'COMPLETE_ALIAS_INVENTORY')
    worker = lifetime['worker']
    r.check(worker == document['identity']['worker'] == document['sessions']['candidate']['setup']['binding']['worker']
        and worker['pid'] == lifetime['pid'] and worker['process_start_identity'] == lifetime['process_start_identity']
        and positive(lifetime['pid']) and positive(lifetime['native_thread']) and worker['rank'] == 0
        and worker['logical_device'] == worker['configured_logical_device'] == worker['observed_logical_device']
        and worker['worker_start_identity'] and worker['model_instance_id'], 'ONE_ACTUAL_WORKER_PROCESS_LIFETIME')
    r.check(inventory['build'] == document['sessions']['candidate']['setup']['binding']['build'] == BUILD_SHA,
            'SINGLE_NATIVE_BUILD_JOIN')
    r.check(inventory['failure'] is inventory['first_failure'] is None and inventory['failure_events'] == []
        and inventory['blockers'] == [] and not any(document.get(k) for k in ('error', 'cleanup_error', 'quarantine_required')),
        'FIRST_FAILURE_OR_INCOMPLETE_RECOVERY')
    details['requests'] = r.section('requests', requests, r, document, inventory, serving)
    rebuilt, callback_total, producer_total = [], 0, 0
    all_calls, all_tokens, outside = set(), set(), []
    for index, row in enumerate(rows):
        label = row['phase']
        r.section(label+':trace', trace, r, row, lifetime, label)
        if index:
            previous, current = rows[index-1]['native_after'], row['native_before']
            delta = current['all_runs'] - previous['all_runs']
            r.check(delta >= 0 and delta == current['non_target_runs'] - previous['non_target_runs']
                and all(current[k] == previous[k] for k in current if k not in ('all_runs', 'non_target_runs')),
                'BETWEEN_REQUEST_TARGETED_TRACE_ACTIVITY', label)
            outside.append(dict(before_phase=label, non_target_runs=delta))
        r.check(row['callback_start'] == callback_total and row['producer_start'] == producer_total, 'PUBLIC_LEDGER_OFFSET', label)
        callback_total += len(row['callbacks']); producer_total += len(row['producers'])
        reconstructed = r.section(label+':candidates', correlate, row)
        rebuilt.append(reconstructed)
        r.check(reconstructed == row['candidates'] and len(row['candidates']) == expected_count, 'COMPLETE_CANDIDATE_RECONSTRUCTION', label)
        calls = [e['call'] for e in row['native_events'] if e['kind_name'] == 'qmm_enter']
        r.check(not all_calls.intersection(calls), 'NATIVE_CALL_REUSED_ACROSS_REQUESTS', label); all_calls.update(calls)
        callbacks = row['callbacks']
        callback_calls = [c.get('independent_call') for c in callbacks]
        tokens = [c['native_packet']['token'] for c in callbacks]
        r.check(len(set(callback_calls)) == len(callback_calls) and set(callback_calls).issubset(calls)
            and len(set(tokens)) == len(tokens) and not all_tokens.intersection(tokens), 'CALLBACK_TOKEN_UNIQUE_ACTUAL_CALL', label)
        all_tokens.update(tokens)
        r.check((not callbacks) if index == 7 else bool(callbacks), 'NATIVE_CALLBACK_REMOVED_BEFORE_HEALTH', label)
        for p in row['producers']:
            r.section(label+':producer', public_producer, r, p, row, alias_pairs, label)
        if reconstructed is not None:
            for ordinal, candidate in enumerate(reconstructed):
                r.section(label+':'+str(ordinal), callback_proof, r, candidate, row, lifetime, alias_pairs, artifacts, label+':'+str(ordinal))
        complete = row['complete_inventory']
        r.check(complete['native_writer_count'] == len(calls) == expected_count
            and complete['native_events_sha256'] == digest(row['native_events'])
            and complete['candidates_sha256'] == digest(row['candidates'])
            and complete['actual_aliases_sha256'] == digest(aliases), 'FULL_INVENTORY_HASHES', label)
    frozen = r.section('healthy_freeze', freeze, rebuilt[0], rebuilt[1]) if all(rebuilt[:2]) else None
    r.check(frozen == inventory['candidates'], 'TWICE_HEALTHY_CLASSIFICATION_RECOMPUTATION')
    selected = inventory['single_site_c1']['selection']
    qualification = inventory['single_site_c1']['reference_lossless_off']
    if frozen is not None:
        resolved = {c['target_key']:c for c in frozen if c['status'] == 'OBSERVED_NOT_QUALIFIED'}
        r.check(len(resolved) == expected_resolved == inventory['resolved_candidate_count'] and bool(resolved), 'COMPLETE_RESOLVED_SET')
        if resolved:
            key = min(resolved)
            r.check(selected['target_key'] == key and selected['inventory_index'] == resolved[key]['index']
                and (expected_selection is None or (key, selected['inventory_index']) == expected_selection), 'MIN_COMPLETE_HEALTHY_KEY_SELECTION')
        r.check(selected['policy'] == 'MIN_ACTUAL_HEALTHY_RESOLVED_TARGET_KEY_NO_FALLBACK'
            and selected['complete_native_count'] == len(frozen) and selected['complete_resolved_count'] == len(resolved)
            and selected['inventory_sha256'] == digest(frozen) and selected['aliases_sha256'] == digest(aliases)
            and selected['planned_target_count'] == 1 and selected['max_targets_is_selector'] is False,
            'COMPLETE_SELECTION_RECEIPT')
        details['inventory'] = dict(per_request_counts=[len(row['candidates']) for row in rows], natural_calls=len(all_calls),
            healthy_classification=dict(Counter(c['status'] for c in frozen)), classification_reasons=dict(Counter(
                reason for c in frozen for reason in c['reasons'])), resolved_count=len(resolved), aliases=len(aliases),
            selected_key=selected['target_key'], selected_index=selected['inventory_index'],
            outside_request_non_target_runs=outside)
    outcomes = inventory['outcomes']
    r.check(len(outcomes) == 5 and [v['mode'] for v in outcomes] == ['reference', 'lossless', 'off', 'fault', 'off'], 'EXACT_FIVE_OUTCOMES_NO_FALLBACK')
    r.check(qualification['target_key'] == selected['target_key'] and qualification['request_ids'] == [o['request_id'] for o in outcomes[:3]]
        and qualification['outcomes_sha256'] == digest(outcomes[:3]) and qualification['lifetime_sha256'] == digest(lifetime)
        and qualification['build'] == inventory['build'] and qualification['baseline_output_sha256'] == rows[0]['output_sha256']
        and qualification['scope'] == SCOPE and qualification['native_owner'] == words(hashlib.sha256(qualification['owner'].encode()).hexdigest())[:2]
        and positive(qualification['generation']), 'SOURCE_BOUND_REFERENCE_LOSSLESS_OFF_ISSUANCE')
    admissions = callbacks = 0
    for index, row in enumerate(rows):
        mode = MODES[index]
        r.check(row['plan'] == dict(phase=PHASES[index], mode=mode, target=selected['target_key'] if mode else None)
            and row['admissions'] == row['completed'] == int(mode is not None), 'ONE_FRESH_PLAN_ADMISSION_PER_REQUEST', row['phase'])
        actual = [c for c in row['callbacks'] if c['claimed'] == 1]
        r.check(len(actual) == int(mode is not None) and all(c['target_key'] == selected['target_key'] for c in actual),
                'ONLY_SELECTED_CALL_WAS_CLAIMED', row['phase'])
        if mode:
            candidate = row['candidates'][selected['inventory_index']]
            r.check(classify(candidate) == ('OBSERVED_NOT_QUALIFIED', [])
                and candidate['callback']['target_key'] == selected['target_key']
                and all(candidate['native'][k] == rows[1]['candidates'][selected['inventory_index']]['native'][k] for k in STABLE_NATIVE),
                'SELECTED_OCCURRENCE_STILL_VALID_BEFORE_EACH_ADMISSION', row['phase'])
            r.section(row['phase']+':outcome', outcome, r, outcomes[index-2], candidate, row, qualification, row['phase'])
        phase = document['phases'][index]
        r.section(row['phase']+':advance', runtime_state, r, phase['advance']['native_state'], qualification,
            lifetime['native_thread'], callbacks, admissions, removed=False, sealed=index >= 5, where=row['phase']+':advance')
        callbacks += len(row['callbacks']); admissions += int(mode is not None)
        r.section(row['phase']+':result', runtime_state, r, phase['result']['native_state'], qualification,
            lifetime['native_thread'], callbacks, admissions, removed=index == 7, sealed=index >= 4, where=row['phase']+':result')
    r.check(inventory['qualified_candidate_count'] == 1 and len({o['target'] for o in outcomes}) == 1,
            'SINGLE_NATIVE_LOSSLESS_TARGET')
    r.check(len({o['writer_output_sha256'] for o in outcomes}) == len({o['residual_before_sha256'] for o in outcomes}) == 1,
            'FULL_WRITER_OUTPUT_AND_RESIDUAL_BASELINES')
    r.check(inventory['last_native_receipt'] == outcomes[-1]['observed_consumer_receipt'], 'LAST_RECEIPT_NOT_OVERWRITTEN_BY_SUCCESS')
    final = document['sessions']['candidate']['remove']['report']
    r.section('final_native', runtime_state, r, inventory['native_state'], qualification, lifetime['native_thread'],
        callbacks, admissions, removed=True, sealed=True, where='final')
    r.check(final['native_state'] == inventory['native_state'] and final['completed_requests'] == 8
        and final['full_native_writer_counts'] == [expected_count]*8 and final['phase'] == inventory['phase'] == 'complete'
        and final['owned_root_logits_hooks'] == final['owned_module_hooks'] == final['owned_public_api_wrappers'] == 0
        and inventory['owned_hooks'] == inventory['owned_public_api_wrappers'] == 0
        and inventory['single_site_c1']['owned_root_logits_hooks'] == 0
        and final['installed'] is inventory['installed'] is False and inventory['native_removed'] is True,
        'FINAL_ALL_OWNED_NATIVE_AND_PYTHON_RESOURCES_ZERO')
    r.check(all(inventory[k] == 0 for k in ('pending_output_tensors', 'pending_residual_tensors', 'pending_public_storage_owners'))
        and inventory['pending_per_call_leases'] == inventory['pending_outcomes'] == []
        and inventory['public_output_observer'] is None, 'PENDING_PUBLIC_OWNERS_OR_LEASES_RETAINED')
    r.check(final['single_site_c1'] == inventory['single_site_c1'] == document['single_site_c1'], 'SINGLE_SITE_FINAL_REPORT_JOIN')
    snapshots = [('initial', inventory['metadata_arrays_initial'])] + [(row['phase'], row['metadata_arrays']) for row in rows]
    snapshots.append(('final', inventory['metadata_arrays_latest']))
    details['metadata_arrays'] = r.section('metadata_arrays', metadata, r, snapshots, lifetime, inventory['build'])
    details['effect'] = dict(native_outcomes=len(outcomes), fault_invocations=sum(o['mode'] == 'fault' for o in outcomes),
        fault_changed_residual_bytes=[o['consumer_receipt']['changed_residual_bytes'] for o in outcomes if o['mode'] == 'fault'],
        writer_output_sha256=outcomes[0]['writer_output_sha256'], full_residual_sha256=outcomes[0]['residual_before_sha256'],
        allocator_artifacts_rehashed=len(artifacts.cache), final_host_bytes=inventory['native_state']['core']['host_bytes'])
    return dict(**r.summary(), **details, scope=SCOPE, single_site_receipt_chain_reconciled=not r.errors,
        public_qualification_granted=False, c2=False, c3=False, e5=False, complete_tensor_bytes_rehashed=False,
        evidence_limits=[
            'Full inputs/logits/residuals/writer outputs were hashed by frozen runtime code; no complete tensor bins are available for independent device-byte rehash.',
            'Saved relevant allocator segments are rehashed and adjacency rebuilt; complete_snapshot_sha256 names an unsaved whole allocator snapshot.',
            'Object/storage identity, strong-reference duration, original-call counts and exact readbacks are source-bound runtime/Native receipts, not new live observations.',
            'Native owner fields may remain as retired identity after removal; active resources and callbacks are required to be zero.',
            'Health request checks complete logits after Native removal; the later ordinary generation checks tokens after Python cleanup.',
            'This fixed fresh-prefill QMM target is not the complete resolved set, C2, C3, FIA or E5.',
            'Separate E5 review reported an earlier KV-cache input difference and incomparable preceding rows; this reader does not establish fault-only output attribution.'])


def provenance(document, inventory, source_root, native_root):
    r = Review()
    manifest_path = resolve_under(source_root, 'source_manifest.json')
    manifest = read(manifest_path)
    r.check(file_sha(manifest_path) == SOURCE_MANIFEST and manifest['base_commit'] == SOURCE_COMMIT, 'EXACT_SOURCE54_MANIFEST')
    files = {v['path']:v for v in manifest['files']}
    r.check(len(files) == len(manifest['files']), 'SOURCE_MANIFEST_DUPLICATE_PATH')
    hashes = {}
    for name in SOURCES:
        path = resolve_under(source_root, name)
        hashes[name] = file_sha(path)
        r.check(files[name]['sha256'] == hashes[name] and files[name]['bytes'] == path.stat().st_size, 'FROZEN_SOURCE_BYTES', name)
    loaded = document['sessions']['candidate']['setup']['sources']
    r.check(len(loaded) == len(SETUP_SOURCES), 'EXACT_INSTALLED_SOURCE_SET')
    for row, relative in zip(loaded, SETUP_SOURCES):
        installed = PurePosixPath('/installed') / (relative[4:] if relative.startswith('src/') else relative)
        r.check(PurePosixPath(row['path']) == installed and row['sha256'] == hashes[relative], 'ACTUAL_INSTALLED_SOURCE_JOIN', relative)
    path = resolve_under(native_root, 'build.json'); build = read(path)
    r.check(file_sha(path) == BUILD_SHA == inventory['build'] and build['schema'] == 'demoncat.writer26.group_deployment.v2'
        and build['legacy_writers_absent'] is True and build['actual_npu_calls'] == 0, 'EXACT_NATIVE_BUILD')
    for name, expected in build['files'].items():
        r.check(file_sha(resolve_under(native_root, name)) == expected and valid_sha(expected), 'NATIVE_BUILD_FILE_BYTES', name)
    receipt_path = resolve_under(native_root, 'build-receipt.json')
    receipt = read(receipt_path)
    r.check(file_sha(receipt_path) == build['build_receipt_sha256']
        and receipt['schema'] == 'demoncat.writer26.group_build.v2' and receipt['actual_npu_calls'] == 0
        and receipt['device_mounts'] == [] and receipt['sources'] == build['sources'], 'NATIVE_BUILD_RECEIPT_BYTES_AND_SOURCES')
    r.check(set((SOURCES[8], SOURCES[9])).issubset(build['sources']), 'NATIVE_CORE_COMPILE_INPUTS_REQUIRED')
    for name, expected in build['sources'].items():
        actual = file_sha(resolve_under(source_root, name))
        r.check(valid_sha(expected) and expected == actual == files[name]['sha256'], 'ACTUAL_NATIVE_COMPILE_SOURCE_BYTES', name)
    r.check(all(not (Path(native_root)/name).exists() for name in ('libfault26_qmm_output_underflow_abi0.so',
        'libfault26_fia_writer_proxy_abi0.so')), 'LEGACY_WRITER_NOT_IN_V2_BUNDLE')
    spec = build['metadata_array_ownership']
    r.check(spec['schema'] == 'demoncat.metadata_arrays.v1' and spec['state_version'] == 1 and spec['images'] == SYMBOLS
        and spec['reset_supported'] is False and spec['runtime_quiescent_readback_required'] is True
        and spec['parent_receipt_sha256'] == build['files']['metadata-array-ownership.json'], 'NATIVE_METADATA_OWNERSHIP_BUILD')
    r.check(build['files']['libfault26_readonly_trace_abi0.so'] == inventory['lifetime']['native_inventory_sha256'] == TRACE_SHA
        and build['files']['libdcat26_group_v2.so'] == inventory['lifetime']['native_writer_sha256'] == WRITER_SHA, 'ACTUAL_NATIVE_TRACE_WRITER_IMAGES')
    for label, report in [('initial', inventory['metadata_arrays_initial']), ('final', inventory['metadata_arrays_latest'])]:
        for image in report['images']:
            r.check(image['sha256'] == build['files'][image['name']] and image['path'] == '/native/'+image['name'],
                    'LOADED_METADATA_IMAGE_BYTES', label+':'+image['name'])
    startup = document['metadata_startup']
    r.check(startup['schema'] == 'demoncat.writer26.startup_dependencies.v1' and startup['pid'] == inventory['lifetime']['pid']
        and startup['manifest_sha256'] == BUILD_SHA and startup['implementation_sha256'] == hashes[SOURCES[5]]
        and startup['model_runtime_imports_before'] == [] and startup['callback_installed'] is False
        and startup['device_api_calls'] == 0 and startup['global_symbol_scope_changed'] is False, 'PRE_MODEL_LOCAL_DEPENDENCY_BOOTSTRAP')
    r.section('startup_metadata', metadata, r, [('startup', startup['metadata_arrays']),
        ('initial', inventory['metadata_arrays_initial']), ('final', inventory['metadata_arrays_latest'])], inventory['lifetime'], BUILD_SHA)
    return dict(**r.summary(), source_files_rehashed=len(hashes), native_files_rehashed=len(build['files'])+1,
        native_compile_sources_rehashed=len(build['sources']),
        source_manifest_sha256=file_sha(manifest_path), native_build_sha256=file_sha(path),
        external_model_method_source_files_rehashed=False,
        source_limit='Saved external vLLM/model method source and code hashes bind target identity; their external files and loaded code objects are not independently re-read here.')


def forbid_runtime(event, args):
    if event in ('ctypes.dlopen', 'subprocess.Popen', 'os.system', 'socket.connect'):
        raise RuntimeError('CPU_READ_ONLY_REVIEW:' + event)
    if event == 'import' and args[0].split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
        raise RuntimeError('NO_FRAMEWORK_IMPORT_IN_REVIEW')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True, help='Directory containing original result.json and writer-v2/')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    args = parser.parse_args(argv)
    sys.addaudithook(forbid_runtime)
    report = dict(schema='demoncat.writer26.single_site_source54.independent_review.v1',
        reviewer_sha256=file_sha(__file__), read_only=True, public_qualification_granted=False, c2=False, c3=False, e5=False)
    originals = {}
    try:
        for name, relative, expected in (('result', 'result.json', RESULT_SHA),
            ('inventory', 'writer-v2/candidate-experiment/writer_inventory.json', INVENTORY_SHA),
            ('serving', 'writer-v2/runtime/serving_receipt.json', SERVING_SHA)):
            path = resolve_under(args.evidence_root, relative)
            actual = file_sha(path)
            originals[name] = dict(path=str(path), sha256=actual, bytes=path.stat().st_size)
            if actual != expected:
                raise ValueError('ORIGINAL_SOURCE54_SHA_MISMATCH:' + name)
        report['originals'] = originals
        document, inventory, serving = (read(originals[n]['path']) for n in ('result', 'inventory', 'serving'))
        report['provenance'] = provenance(document, inventory, args.source_root, args.native_root)
        report['evidence'] = review(document, inventory, serving, Artifacts(args.evidence_root))
        report['single_site_source_bound_evidence_reconciled'] = report['provenance']['consistent'] and report['evidence']['consistent']
        report['status'] = ('SOURCE_BOUND_SINGLE_QMM_RECEIPTS_RECONCILED_NOT_C2_C3_E5'
            if report['single_site_source_bound_evidence_reconciled'] else 'EVIDENCE_GAP_OR_CONTRADICTION')
    except (KeyError, TypeError, ValueError, IndexError, OSError, OverflowError) as error:
        report.update(status='EVIDENCE_GAP_OR_CONTRADICTION', originals=originals,
            single_site_source_bound_evidence_reconciled=False, error=type(error).__name__ + ':' + str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['single_site_source_bound_evidence_reconciled'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
