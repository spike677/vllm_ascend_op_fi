"""All-hit CPU integer-index oracle only; no Torch, NPU, views or injector code."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def reference():
    size, stride, offset = (8, 8), (16, 1), 17
    coordinates = list(itertools.product(*(range(n) for n in size)))
    indices = [offset + sum(i * s for i, s in zip(coord, stride)) for coord in coordinates]
    initial = list(range(256))
    assert len(indices) == len(set(indices)) == 64
    assert min(indices) == 17 and max(indices) == 136
    assert all(0 <= index < len(initial) for index in indices)
    untouched = [index for index in range(256) if index not in set(indices)]
    assert len(untouched) == 192
    def scatter(before, values):
        result = before.copy()
        for index, value in zip(indices, values):
            result[index] = value
        assert all(result[i] == before[i] for i in untouched)
        return result

    def exact(*vectors):
        assert all(type(value) is int and -256 <= value <= 256 for vector in vectors for value in vector)

    baseline = initial.copy()
    calls = []
    for call in range(3):
        values = [-64 * (call + 1) - index for index in range(64)]
        before = baseline.copy()
        baseline = scatter(before, values)
        temporary_after = scatter(initial, values)
        healthy_output = [value + 1 for value in baseline]
        stale_output = [value + 1 for value in initial]
        changed = [i for i in range(256) if baseline[i] != before[i]]
        consumer_changed = [i for i in range(256) if healthy_output[i] != stale_output[i]]
        assert changed == consumer_changed == indices  # Every write is non-idempotent.
        exact(before, values, baseline, temporary_after, healthy_output, stale_output)
        calls.append({
            'call': call,
            'baseline_pre_base_flat': before,
            'baseline_post_base_flat': baseline.copy(),
            'fault_pre_base_flat': initial.copy(),
            'fault_post_base_flat': initial.copy(),
            'baseline_pre_view_flat': [before[i] for i in indices],
            'fault_pre_view_flat': [initial[i] for i in indices],
            'write_values_flat': values,
            'baseline_consumer_flat': healthy_output,
            'fault_consumer_flat': stale_output,
            'expected_fault_temporary_post_base_flat': temporary_after,
            'baseline_changed_from_pre_indices': changed,
            'fault_changed_from_pre_indices': [],
            'consumer_pair_changed_indices': consumer_changed,
        })
    guard_values = [-16 - index for index in range(64)]
    guard_updated = scatter(initial, guard_values)
    guard_consumer = [value + 1 for value in guard_updated]
    exact(initial, guard_values, guard_updated, guard_consumer)
    assert [i for i in range(256) if guard_updated[i] != initial[i]] == indices
    return {
        'schema': 'fault15.alias_detach.allhit.logical_reference.v1',
        'status': 'CPU_LOGICAL_REFERENCE_ONLY_NOT_NPU_VALIDATION',
        'forward_count': 2, 'calls_per_forward': 3, 'registered_base_count': 2,
        'matched_calls_per_arm': 6,
        'base_shape': [16, 16], 'view_size': list(size), 'view_stride': list(stride), 'offset': offset,
        'indices': indices, 'untouched_indices': untouched,
        'initial_base_flat': initial, 'initial_view_flat': [initial[index] for index in indices],
        'calls': calls,
        'guards': {
            'other_callsite': {
                'operation': 'Read-only native view of the current registered base from a different callsite',
                'expected_alias': 'original input base',
                'expected_values_rule': 'current input base values at indices; do not assume reset values after writes',
            },
            'unregistered_base': {
                'operation': 'Same selected entry, fresh independent unregistered NZ29 base, real copy_ then base+1',
                'fresh_base_per_forward': True,
                'initial_base_flat': initial.copy(),
                'initial_view_flat': [initial[index] for index in indices],
                'write_values_flat': guard_values,
                'expected_updated_base_flat': guard_updated,
                'expected_updated_consumer_flat': guard_consumer,
            },
        },
        'expected_counts_per_arm': {
            'matched_original_calls': 6, 'unmatched_calls': 4, 'unmatched_original_calls': 4,
            'guard_calls_per_forward': 2, 'fault_injection_hits': {'B': 0, 'F': 6},
        },
        'target_count': 64, 'untouched_count': 192,
        'exact_bfloat16_reason': 'All values are integers in [-256,256]; no rounding tolerance is needed.',
        'sequence_semantics': 'Each forward starts with a different fresh registered base, both retained alive. Its three calls reuse that base without resetting. The same per-call reference applies independently to both forwards.',
        'scope': 'Pure logical oracle, not NPU evidence. Later same-base B/F pre-values differ naturally after the first write; local transitions and each new storage relation must be checked separately from inherited differences.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = reference()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'target_count': 64, 'untouched_count': 192,
                      'forward_count': 2, 'calls_per_forward': 3, 'matched_calls_per_arm': 6,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
