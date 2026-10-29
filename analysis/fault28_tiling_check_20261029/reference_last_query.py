"""FP64 reference from represented FP16 native HF inputs; no NPU or injection.

Evaluate the prespecified request0 final prefill query, with every valid key
included once. A smaller reference error does not establish absence of a
batch-consistency fault; error direction is reported without a pass threshold.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def native_tensor(case, row):
    import numpy as np
    assert row['dtype'] == 'torch.float16' and row['byteorder'] == 'little'
    assert row['raw_order'] == 'C-contiguous logical elements'
    assert row['hashed_slice'] == 'entire logical tensor'
    path = (case / row['private_raw_file']).resolve()
    assert case.resolve() in path.parents and path.is_file()
    assert path.stat().st_size == math.prod(row['shape']) * 2 == row['logical_bytes']
    assert digest(path) == row['logical_sha256']
    return np.memmap(path, dtype='<f2', mode='r', shape=tuple(row['shape']))


def reference(case, call):
    import numpy as np
    assert call['actual_layout'] == 'BNSD' and call['target_mapping']['batch_slot'] == 0
    values = call['selected_layer_private_inputs']
    q, k, v = [native_tensor(case, values[name]) for name in ('query', 'key', 'value')]
    batch, heads, length, dimension = q.shape
    assert k.shape == v.shape and k.shape[0] == batch and k.shape[2:] == (length, dimension)
    assert call['target_mapping']['last_token_position'] == length - 1
    assert heads % k.shape[1] == 0
    args = call['scalar_arguments']
    assert args['dropout_p'] == 0.0
    assert heads == k.shape[1] or args['enable_gqa'] is True
    # For a causal, equal-length full prefill the final query sees all keys.
    # Reject other masks rather than silently changing their semantics.
    if values.get('attn_mask') is None:
        assert args['is_causal'] is True
    else:
        raise ValueError('Explicit masks require a separately verified mask reader')
    scale = args['scale'] if args['scale'] is not None else 1.0 / math.sqrt(dimension)
    output = []
    for head in range(heads):
        kv_head = head // (heads // k.shape[1])
        scores = k[0, kv_head].astype(np.float64) @ q[0, head, -1].astype(np.float64)
        scores *= scale
        weights = np.exp(scores - scores.max())
        weights /= weights.sum(dtype=np.float64)
        output.append(weights @ v[0, kv_head].astype(np.float64))
    return np.asarray(output, dtype=np.float64), {
        'batch': batch, 'query_heads': heads, 'kv_heads': k.shape[1], 'length': length,
        'head_dimension': dimension, 'scale': scale,
        'all_represented_fp16_inputs': {name: values[name]['logical_sha256'] for name in ('query', 'key', 'value')},
        'valid_key_indices': [0, length - 1], 'valid_key_count_per_head': length,
        'query_index': length - 1, 'request_index': 0,
        'arithmetic': 'FP64 QK, exp, normalization and PV, from exact represented FP16 bytes'}


def metrics(actual, expected):
    import numpy as np
    error = actual.astype(np.float64) - expected
    return {'max_abs': float(np.abs(error).max()), 'mean_abs': float(np.abs(error).mean()),
            'rms': float(np.sqrt(np.mean(error * error))),
            'nonzero_elements': int(np.count_nonzero(error)), 'elements': int(error.size)}


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-case', type=Path, required=True)
    parser.add_argument('--fault-case', type=Path, required=True)
    parser.add_argument('--call-file', default='native_calls/call_001.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    calls = [json.loads((case / args.call_file).read_bytes()) for case in (args.baseline_case, args.fault_case)]
    refs = [reference(case, call) for case, call in zip((args.baseline_case, args.fault_case), calls)]
    assert refs[0][1] == refs[1][1] and np.array_equal(refs[0][0], refs[1][0])
    outputs = []
    for case, call in zip((args.baseline_case, args.fault_case), calls):
        output = native_tensor(case, call['selected_layer_private_output'])
        assert list(output.shape) == call['selected_layer_private_inputs']['query']['shape']
        outputs.append(output[0, :, -1].astype(np.float64))
    report = {'schema': 'fault28.last_query_reference.v1', 'status': 'REFERENCE_EVALUATED_NOT_ACCEPTANCE',
              'inputs_identical': True, 'target': refs[0][1],
              'baseline_vs_reference': metrics(outputs[0], refs[0][0]),
              'fault_vs_reference': metrics(outputs[1], refs[0][0]),
              'fault_vs_baseline': metrics(outputs[1], outputs[0]),
              'call_record_sha256': [digest(case / args.call_file) for case in (args.baseline_case, args.fault_case)],
              'analyzer_sha256': digest(Path(__file__)), 'numpy_version': np.__version__,
              'scope': 'One prespecified final query; does not establish tiling legality, model propagation or historical reproduction.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
