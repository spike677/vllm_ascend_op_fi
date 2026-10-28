"""CPU integer-index oracle preparation only: no Torch, NPU or injector code."""
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
    values = [-64 - index for index in range(64)]
    assert len(indices) == len(set(indices)) == 64
    assert min(indices) == 17 and max(indices) == 136
    assert all(0 <= index < len(initial) for index in indices)
    updated = initial.copy()
    for index, value in zip(indices, values):
        updated[index] = value
    untouched = [index for index in range(256) if index not in set(indices)]
    assert len(untouched) == 192
    assert all(updated[index] == initial[index] for index in untouched)
    assert all(updated[index] != initial[index] for index in indices)
    updated_output = [value + 1 for value in updated]
    stale_output = [value + 1 for value in initial]
    assert [index for index in range(256) if updated_output[index] != stale_output[index]] == indices
    assert all(type(value) is int and -256 <= value <= 256
               for vector in (initial, values, updated, updated_output, stale_output) for value in vector)
    return {
        'status': 'CPU_LOGICAL_REFERENCE_ONLY_NOT_NPU_VALIDATION',
        'base_shape': [16, 16], 'view_size': list(size), 'view_stride': list(stride), 'offset': offset,
        'indices': indices, 'untouched_indices': untouched,
        'initial_base_flat': initial, 'initial_view_flat': [initial[index] for index in indices],
        'write_values_flat': values, 'expected_updated_base_flat': updated,
        'expected_stale_base_flat': initial, 'expected_updated_consumer_flat': updated_output,
        'expected_stale_consumer_flat': stale_output,
        'target_count': 64, 'untouched_count': 192,
        'exact_bfloat16_reason': 'All values are integers in [-256,256]; no rounding tolerance is needed.',
        'scope': 'Defines expected logical effects only. It does not demonstrate actual NPU format, aliasing, conversion allocation, writeback or injection.'}


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
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
