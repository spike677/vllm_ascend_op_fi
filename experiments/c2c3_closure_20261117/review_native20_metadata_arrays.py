"""Offline phase-boundary SDK ownership audit; independent of coverage/E5."""
import argparse
import hashlib
import json
from pathlib import Path

from review_native39_metadata_arrays import state, monotonic, require, sha


def inspect_snapshot(value, label):
    require(value.get('schema') == 'demoncat.metadata_arrays.v1'
            and value.get('source') == 'ACTUAL_LOADED_PROXY_COUNTERS'
            and value.get('scope') == 'SDK_GETTER_ARRAYS_ONLY'
            and value.get('reset_supported') is False
            and type(value.get('pid')) is int and value['pid'] > 0
            and sha(value.get('build_profile_sha256'))
            and sha(value.get('build_manifest_sha256')), label + ':ARRAY_IDENTITY')
    rows, identities = {}, {}
    for item in value['images']:
        name = item['name']
        require(name in ('libopapi.so', 'libopapi_nn.so') and name not in rows
                and item.get('symbol') == 'dcat20_metadata_arrays_v1'
                and sha(item.get('sha256')) and item.get('path')
                and item.get('mapped_paths') and item.get('symbol_providers')
                and set(item['symbol_providers']).issubset(item['mapped_paths']), label + ':LOADED_IMAGE')
        rows[name] = state(item['state'], label + ':' + name)
        identities[name] = (item['path'], item['sha256'], item['symbol'], rows[name]['instance'],
                            sorted(item['mapped_paths']), sorted(item['symbol_providers']))
    require(set(rows) == {'libopapi.so', 'libopapi_nn.so'}
            and len({v['instance'] for v in rows.values()}) == 2, label + ':DISTINCT_PROXY_INSTANCES')
    return (value['pid'], value['build_profile_sha256'], value['build_manifest_sha256'], identities), rows


def review_group(folder, previous=None):
    receipt = json.loads((folder / 'runtime/serving_receipt.json').read_bytes())
    phases = receipt['native_phases']; qualification = receipt['qualification']
    expected = ['discovery_1', 'discovery_2', 'reference', 'lossless', 'qualification_off', 'baseline', 'fault']
    if qualification['trigger'] == 'every_match':
        expected.append('fault_repeat')
    else:
        require(qualification['trigger'] == 'once', 'UNKNOWN_TRIGGER')
    expected += ['off_observed', 'off']
    require([p['phase'] for p in phases] == expected and qualification['completed_phases'] == expected,
            'INCOMPLETE_PHASES')
    delta_fields = ('capture_calls', 'getter_calls', 'acquired_arrays', 'released_arrays')
    snapshots, positive_intervals, previous_activity = [], 0, None
    for phase in phases:
        label = phase['phase']; identity, rows = inspect_snapshot(phase['metadata_arrays'], label)
        activity = phase['metadata_array_activity']
        require(activity.get('process_lifetime_nonzero_alone_accepted') is False
                and activity['pid'] == identity[0] and activity['build_profile_sha256'] == identity[1],
                label + ':ACTIVITY_SCOPE')
        if previous is not None:
            require(identity == previous[0], label + ':LOADED_BUILD_OR_INSTANCE_CHANGED')
            for name in rows:
                monotonic(previous[1][name], rows[name], label + ':' + name)
            actual = {key: sum(rows[name][key] - previous[1][name][key] for name in rows) for key in delta_fields}
            if previous_activity is not None:
                declared = {key: activity['group_observed_delta'][key] - previous_activity[key] for key in delta_fields}
                require(actual == declared, label + ':RECORDED_ACTIVITY_DELTA_MISMATCH')
                require(all(value > 0 for value in actual.values()), label + ':NO_CURRENT_PHASE_GETTER_ACTIVITY')
                positive_intervals += 1
        snapshots.append(dict(phase=label, totals={key: sum(v[key] for v in rows.values()) for key in delta_fields}))
        previous = identity, rows
        previous_activity = activity['group_observed_delta']
    health_paths = list((folder / 'group').glob('*-worker-health.json'))
    require(len(health_paths) == 1, 'UNIQUE_FINAL_HEALTH_REQUIRED')
    health = json.loads(health_paths[0].read_bytes())
    identity, rows = inspect_snapshot(health['metadata_arrays'], 'health')
    require(identity == previous[0], 'HEALTH_LOADED_IDENTITY_CHANGED')
    for name in rows:
        monotonic(previous[1][name], rows[name], 'health:' + name)
    require(health['owned_hooks'] == 0 and health['baseline'] == health['off']
            and health['baseline'] == phases[0]['model_output_sha256']
            and qualification['recovery_verified'] is True and qualification['failure'] is None,
            'FINAL_HEALTH_OR_OWNERSHIP_FAILED')
    return dict(checks_passed=True, phase_boundaries=len(phases),
                positive_within_group_intervals=positive_intervals, phases=snapshots,
                loaded_counter_identity=identity, final_images=rows,
                qualifications_granted=False,
                scope='Recorded phase/health boundaries; per-call effects and complete coverage require their separate audit.'), (identity, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = dict(schema='demoncat.native20_metadata_review.v1', qualifications_granted=False)
    previous = None
    for name in ('c2', 'c3'):
        row, previous = review_group(args.attempt_case / name, previous)
        path = args.attempt_case / name / 'runtime/serving_receipt.json'
        row['source_receipt_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[name] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({name: {key: result[name][key] for key in
                      ('checks_passed', 'phase_boundaries', 'positive_within_group_intervals')} for name in ('c2', 'c3')}))


if __name__ == '__main__':
    main()
