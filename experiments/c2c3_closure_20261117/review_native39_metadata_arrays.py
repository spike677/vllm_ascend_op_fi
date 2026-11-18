"""Independently reconcile recorded SDK array ownership, without runtime calls.

This checks recorded bytes, not the runtime admission result. Build/import
identity, numerical effects and statistics retain their separate reviewers.
"""
import argparse
import hashlib
import json
from pathlib import Path


COUNTERS = ('capture_calls', 'active_captures', 'getter_calls', 'active_getters',
            'getter_failures', 'getter_exceptions', 'capture_exceptions',
            'acquired_arrays', 'released_arrays', 'live_arrays', 'peak_live_arrays',
            'ownership_errors', 'revision')
MONOTONIC = tuple(k for k in COUNTERS if not k.startswith('active_') and k != 'live_arrays')
ZERO = ('active_captures', 'active_getters', 'getter_failures', 'getter_exceptions',
        'capture_exceptions', 'live_arrays', 'ownership_errors')
DELTAS = dict(capture_calls=3, getter_calls=9, acquired_arrays=9, released_arrays=9)


class Refusal(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Refusal(reason)


def uint(value, positive=False):
    return type(value) is int and (0 < value if positive else 0 <= value) and value < 2**64


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def state(value, label):
    require(isinstance(value, dict) and type(value.get('version')) is int and value['version'] == 1
            and type(value.get('size')) is int and value['size'] == 120
            and uint(value.get('instance'), True)
            and all(uint(value.get(k)) for k in COUNTERS), label + ':COUNTER_ABI')
    require(all(value[k] == 0 for k in ZERO)
            and value['acquired_arrays'] == value['released_arrays'], label + ':UNRELEASED_OR_ERROR')
    return value


def monotonic(before, after, label):
    require(before['instance'] == after['instance'], label + ':COUNTER_INSTANCE_CHANGED')
    require(all(after[k] >= before[k] for k in MONOTONIC), label + ':COUNTER_RESET_OR_REORDER')


def arrays(value, label):
    require(isinstance(value, dict) and value.get('schema') == 'demoncat.metadata_arrays.v1'
            and value.get('source') == 'ACTUAL_LOADED_PROXY_COUNTERS'
            and value.get('scope') == 'SDK_GETTER_ARRAYS_ONLY' and value.get('reset_supported') is False
            and uint(value.get('pid'), True) and sha(value.get('build_profile_sha256'))
            and sha(value.get('build_manifest_sha256')), label + ':LOADED_REPORT_IDENTITY')
    rows, identities, instances = {}, {}, set()
    for row in value['images']:
        name = row['name']; current = state(row['state'], label + ':' + name)
        require(name in ('libopapi.so', 'libopapi_nn.so') and name not in rows
                and current['instance'] not in instances, label + ':PROXY_OR_INSTANCE_SET')
        require(sha(row.get('sha256')) and isinstance(row.get('path'), str) and row['path']
                and row.get('symbol') == 'fault39_metadata_arrays_v1'
                and isinstance(row.get('mapped_paths'), list) and row['mapped_paths']
                and isinstance(row.get('symbol_providers'), list) and row['symbol_providers']
                and all(isinstance(p, str) and p for p in row['mapped_paths'])
                and len(set(row['mapped_paths'])) == len(row['mapped_paths'])
                and set(row['symbol_providers']).issubset(row['mapped_paths']), label + ':IMAGE_MAPPING_IDENTITY')
        rows[name] = current; instances.add(current['instance'])
        identities[name] = (row['path'], row['sha256'], row['symbol'], current['instance'],
                            tuple(sorted(row['mapped_paths'])), tuple(sorted(row['symbol_providers'])))
    require('libopapi.so' in rows, label + ':MAIN_PROXY_MISSING')
    return dict(identity=(value['pid'], value['build_profile_sha256'], value['build_manifest_sha256'], identities), rows=rows)


def cache_key(value, label):
    require(isinstance(value, (list, tuple)) and len(value) == 3 and isinstance(value[0], str)
            and value[0].startswith('f39v1:aclnnMatmul:') and all(uint(v, True) for v in value[1:]),
            label + ':CACHE_KEY_OR_EPOCH')
    return tuple(value)


class Reviewer:
    def __init__(self):
        self.snapshots = 0; self.previous = self.first = None
        self.witnesses = {}; self.phases = []; self.native_snapshots = 0

    def observe(self, value, label):
        current = arrays(value, label)
        if self.previous is not None:
            require(current['identity'] == self.previous['identity'], label + ':LOADED_INSTANCE_OR_BUILD_CHANGED')
            for name, row in current['rows'].items():
                monotonic(self.previous['rows'][name], row, label + ':' + name)
        if self.first is None:
            self.first = current
        self.previous = current; self.snapshots += 1
        return current

    def activity(self, evidence, current, label, *, bind=False):
        require(isinstance(evidence, dict) and evidence.get('schema') == 'demoncat.metadata_array_activity.v1'
                and evidence.get('pid') == current['identity'][0]
                and evidence.get('build_profile_sha256') == current['identity'][1]
                and evidence.get('process_lifetime_nonzero_alone_accepted') is False,
                label + ':ACTIVITY_IDENTITY')
        expected = {k: sum(row[k] - self.first['rows'][name][k] for name, row in current['rows'].items())
                    for k in DELTAS}
        require(evidence.get('group_observed_delta') == expected, label + ':GROUP_DELTA_MISMATCH')
        rows = evidence.get('cached_registration_witnesses')
        require(isinstance(rows, list) and rows, label + ':CACHE_WITNESSES_MISSING')
        indexed = {}
        for row in rows:
            key = cache_key(row['cache_key'], label)
            require(key not in indexed and row.get('registration_live') is True, label + ':DUPLICATE_OR_DEAD_CACHE')
            indexed[key] = row
        if bind:
            self.witnesses = indexed
        else:
            require(indexed == self.witnesses, label + ':CACHE_WITNESS_BYTES_CHANGED')

    def validate_witnesses(self, current):
        loaded = {row['instance']: row for row in current['rows'].values()}
        intervals = {}
        for key, row in self.witnesses.items():
            witness = row['metadata_arrays']; label = 'cache:' + str(key[2])
            require(witness.get('source') == 'actual_typed_gws_capture', label + ':ACTUAL_CAPTURE_REQUIRED')
            before = state(witness['before'], label + ':before'); after = state(witness['after'], label + ':after')
            monotonic(before, after, label)
            require(after['instance'] in loaded and after['revision'] > before['revision']
                    and all(after[k] - before[k] == count for k, count in DELTAS.items()),
                    label + ':EXACT_THREE_TENSOR_CAPTURE_REQUIRED')
            monotonic(after, loaded[after['instance']], label + ':loaded')
            intervals.setdefault(after['instance'], []).append((before['acquired_arrays'], after['acquired_arrays']))
        for spans in intervals.values():
            spans.sort()
            require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), 'COPIED_OR_OVERLAPPING_CAPTURE_WITNESS')

    def registry(self, native, label):
        found = {}
        for candidate in native['independent_candidates']:
            for cache in candidate['cache_identities']:
                key = cache_key([candidate['signature'], cache['address'], cache['epoch']], label)
                if key not in self.witnesses:
                    continue
                require(key not in found, label + ':AMBIGUOUS_LIVE_REGISTRATION')
                found[key] = dict(cache_key=list(key), registration_live=cache.get('registration_live'),
                                  metadata_arrays=cache.get('metadata_arrays'))
        require(found == self.witnesses, label + ':LIVE_CACHE_REGISTRY_OR_CAPTURE_CHANGED')

    def run(self, initial, bound, health, receipt):
        self.observe(initial, 'initial')
        phases = receipt['native_phases']; qualification = receipt['qualification']
        trigger = qualification['trigger']
        require(trigger in ('once', 'every_match'), 'UNKNOWN_TRIGGER')
        expected = ['discovery_1', 'discovery_2', 'reference', 'lossless_self', 'lossless_mat2', 'lossless_dual', 'fault']
        if trigger == 'every_match':
            expected.append('fault_repeat')
        expected += ['off_observed', 'off']
        require([p['phase'] for p in phases] == expected
                and qualification['completed_phases'] == expected, 'INCOMPLETE_OR_REORDERED_PHASES')
        request_ids = [p['request_id'] for p in phases]
        require(len(set(request_ids)) == len(phases) and qualification['actual_request_ids'] == request_ids,
                'ACTUAL_REQUEST_SEQUENCE_CHANGED')
        require(len(receipt['requests']) == len(phases), 'REQUEST_RECEIPTS_INCOMPLETE')
        universe = None
        for phase_index, phase in enumerate(phases):
            name = phase['phase']; sites = set(); keys = set(); native_count = 0
            require(receipt['requests'][phase_index]['snapshot']['internal_request_id'] == phase['request_id'],
                    name + ':REQUEST_MAPPING_CHANGED')
            for index, call in enumerate(phase['calls']):
                label = name + ':call' + str(index); delta = call['native_delta']
                require(len(delta) <= 1 and call['original_calls'] == 1, label + ':ONE_ORIGINAL_CALL_REQUIRED')
                if not delta:
                    require(call.get('native') is None, label + ':NON_TARGET_NATIVE_STATE')
                    continue
                require(type(delta[0]['count']) is int and delta[0]['count'] == 1, label + ':NATIVE_DELTA_COUNT')
                key = cache_key(delta[0]['key'], label)
                site = (json.dumps(call['site'], sort_keys=True), key)
                require(site not in sites, label + ':DUPLICATE_ACTUAL_SITE_CACHE')
                sites.add(site); keys.add(key); native_count += 1
                native = call.get('native')
                if name not in ('discovery_1', 'discovery_2', 'off'):
                    require(isinstance(native, dict), label + ':NATIVE_SNAPSHOT_MISSING')
                if native is not None:
                    require((native['target_signature'], native['selected_cache_address'], native['selected_cache_epoch']) == key,
                            label + ':ACTUAL_SELECTED_CACHE_MISMATCH')
                    current = self.observe(native['metadata_arrays'], label)
                    require(native['inflight'] == native['leases'] == native['last_error'] == 0,
                            label + ':NATIVE_RESOURCES_OR_ERROR')
                    self.activity(native['metadata_array_activity'], current, label)
                    self.registry(native, label); self.native_snapshots += 1
            require(native_count > 0 and native_count == phase['native_denominator']
                    and phase['independent_denominator_complete'] is True, name + ':NATIVE_DENOMINATOR')
            if universe is None:
                universe = sites
            require(sites == universe, name + ':ACTUAL_TARGET_UNIVERSE_CHANGED')
            current = self.observe(phase['metadata_arrays'], name + ':end')
            if name == 'discovery_2':
                self.activity(bound, current, 'registration_binding', bind=True)
                require(set(self.witnesses) == keys, 'ACTUAL_KEY_WITNESS_COVERAGE_MISMATCH')
                self.validate_witnesses(current)
            if self.witnesses:
                require(keys == set(self.witnesses), name + ':ACTUAL_KEY_WITNESS_COVERAGE_MISMATCH')
            self.phases.append(dict(phase=name, request_id=phase['request_id'], native_calls=native_count,
                                    distinct_cache_keys=len(keys)))
        native = health['native']; current = self.observe(native['metadata_arrays'], 'final_health')
        self.activity(native['metadata_array_activity'], current, 'final_health')
        self.activity(qualification['metadata_array_activity'], current, 'final_qualification')
        self.registry(native, 'final_health'); self.validate_witnesses(current)
        require(not native['owner'] and native['enabled'] is False
                and native['inflight'] == native['leases'] == native['last_error'] == health['owned_hooks'] == 0,
                'FINAL_NATIVE_RESOURCE_OR_ERROR')
        require(health['worker'][0] == current['identity'][0]
                and health['baseline'] == health['off'] == phases[0]['model_output_sha256']
                and qualification['failure'] is None and qualification['recovery_verified'] is True,
                'FINAL_HEALTH_IDENTITY_OR_OFF_MISMATCH')
        return dict(trigger=trigger, distinct_cache_keys=len(self.witnesses), actual_sites=len(universe),
            counter_snapshots=self.snapshots, per_call_native_snapshots=self.native_snapshots,
            phases=self.phases, group_observed_delta=qualification['metadata_array_activity']['group_observed_delta'],
            caches=[dict(cache_key=list(key), registration_live=True,
                         capture_delta={k: row['metadata_arrays']['after'][k]-row['metadata_arrays']['before'][k] for k in DELTAS},
                         witness_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                         instance=row['metadata_arrays']['after']['instance']) for key, row in sorted(self.witnesses.items())],
            worker_identity=list(health['worker']), loaded_counter_identity=current['identity'],
            initial_images=self.first['rows'], final_images=current['rows'])


def review(initial, bound, health, receipt):
    result = dict(schema='demoncat.native39_metadata_array_review.v1', qualifications_granted=False,
                  scope='RECORDED_SDK_GETTER_ARRAYS_ONLY', checks_passed=False, errors=[])
    try:
        result.update(Reviewer().run(initial, bound, health, receipt)); result['checks_passed'] = True
    except (Refusal, KeyError, TypeError, IndexError, ValueError) as error:
        result['errors'].append(str(error) if isinstance(error, Refusal) else 'MALFORMED_EVIDENCE:' + repr(error))
    return result


def read_stable(path):
    before = path.stat(); content = path.read_bytes(); after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
            'EVIDENCE_FILE_CHANGED_DURING_READ:' + str(path))
    return json.loads(content), dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def review_stage(stage):
    files = {}
    for name, suffix in (('initial', 'metadata-arrays-initial'), ('bound', 'metadata-cached-registration-witnesses'),
                         ('health', 'worker-health')):
        matches = list((stage/'native').glob('*-' + suffix + '.json'))
        require(len(matches) == 1, 'ONE_COMPLETED_ARTIFACT_REQUIRED:' + suffix)
        files[name] = matches[0]
    files['receipt'] = stage/'runtime/serving_receipt.json'
    values, records = {}, {}
    for name, path in files.items():
        values[name], records[name] = read_stable(path)
    result = review(**values); result['sources'] = records
    result['reviewer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def link_reviews(previous, current):
    """Check continuity of recorded C2/C3 counters; never issue a live C2 proof."""
    result=dict(checks_passed=False, errors=[], qualifications_granted=False)
    try:
        require(previous['checks_passed'] and current['checks_passed'], 'BOTH_COMPLETED_STAGES_REQUIRED')
        require(previous['trigger']=='once' and current['trigger']=='every_match', 'C2_THEN_C3_REQUIRED')
        require(previous['worker_identity']==current['worker_identity']
                and previous['loaded_counter_identity']==current['loaded_counter_identity'],
                'CROSS_GROUP_WORKER_OR_LOADED_INSTANCE_CHANGED')
        require(previous['caches']==current['caches'], 'CROSS_GROUP_CACHE_EPOCH_OR_WITNESS_CHANGED')
        for name,before in previous['final_images'].items():
            monotonic(before,current['initial_images'][name],'cross_group:'+name)
        result['checks_passed']=True
    except (Refusal,KeyError,TypeError) as error:
        result['errors']=[str(error) if isinstance(error,Refusal) else 'MALFORMED_STAGE_REVIEW:'+repr(error)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--previous-stage', type=Path,
                        help='Optional completed C2 stage; check actual same-Worker counter continuity into C3.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = review_stage(args.stage)
        if args.previous_stage is not None:
            previous=review_stage(args.previous_stage)
            result['previous_stage']=previous
            result['cross_group']=link_reviews(previous,result)
            result['checks_passed']=result['checks_passed'] and result['cross_group']['checks_passed']
            result['errors'].extend(result['cross_group']['errors'])
    except (Refusal, OSError, ValueError) as error:
        result = dict(checks_passed=False, errors=[str(error)], qualifications_granted=False)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('checks_passed', 'errors', 'distinct_cache_keys',
          'actual_sites', 'counter_snapshots', 'qualifications_granted') if k in result}))
    return 0 if result['checks_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
