"""No-device import/input/schema/reference checks for native replay preparation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.dont_write_bytecode = True
    import torch
    import torch_npu
    import replay
    manifest = json.loads(a.input_manifest.read_bytes())
    checks = []
    reference_hashes = []
    for batch in (1, 8):
        case = {'input_manifest_file': str(a.input_manifest),
                'input_manifest_sha256': hashlib.sha256(a.input_manifest.read_bytes()).hexdigest(), 'batch_size': batch}
        _, values = replay.load_frozen_inputs(torch, case)
        for name, value in values.items():
            raw = replay.raw_bytes(torch, value)
            assert replay.digest(raw) == manifest['inputs'][name]['expected_full_sha256'][str(batch)]
            size = len(raw) // batch
            assert all(replay.digest(raw[i * size:(i + 1) * size]) == manifest['inputs'][name]['sha256'] for i in range(batch))
            assert value.device.type == 'cpu'
            checks.append({'batch': batch, 'input': name, 'shape': list(value.shape), 'stride': list(value.stride()),
                           'all_rows_equal_frozen_original_batch0': True, 'full_sha256': replay.digest(raw)})
        reference, scale = replay.reference_queries(torch, values, manifest['native_kwargs']['scale'])
        assert reference.shape == (10, 16, 128) and reference.dtype == torch.float64 and bool(torch.isfinite(reference).all())
        # Query0 permits exactly one key: softmax is exactly one and output equals represented V[0].
        expected0 = torch.stack([values['value'][0, head // 2, 0].double() for head in range(16)])
        assert torch.equal(reference[0], expected0)
        reference_hashes.append(replay.digest(replay.raw_bytes(torch, reference)))
    assert len(set(reference_hashes)) == 1
    schema = str(torch.ops.npu.npu_fusion_attention_v3.default._schema)
    for item in ('head_num', 'input_layout', 'pre_tockens', 'next_tockens', 'inner_precise', 'sparse_mode', 'sink'):
        assert item in schema
    assert callable(torch_npu.npu_fusion_attention_v3)
    kwargs = manifest['native_kwargs']
    assert kwargs == {'head_num': 16, 'input_layout': 'BNSD', 'scale': 0.08838834764831845,
                     'keep_prob': 1.0, 'pre_tockens': 2147483647, 'next_tockens': 0,
                     'inner_precise': 0, 'sparse_mode': 2, 'gen_mask_parallel': True, 'sync': False, 'softmax_layout': ''}
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool), diagonal=1)
    assert mask.dtype == torch.bool and int(mask.sum()) == 2048 * 2047 // 2
    assert not bool(mask[-1].any())
    row = {'schema': 'fault28.causal.cpu_prepare.v1', 'status': 'CPU_PREPARATION_PASSED',
           'npu_calls': 0, 'native_api_schema': schema, 'input_checks': checks,
           'reference_queries': replay.REFERENCE_QUERIES, 'reference_fp64_sha256': reference_hashes[0],
           'reference_B1_B8_exact_equal': True, 'query0_reference_equals_original_value': True,
           'effective_scale_float32': scale, 'mask_sha256': replay.digest(replay.raw_bytes(torch, mask)),
           'replay_source_sha256': replay.sha(Path(replay.__file__)), 'cpu_script_sha256': replay.sha(__file__),
           'input_manifest_sha256': replay.sha(a.input_manifest),
           'limits': 'No NPU dispatch, private-vendor execution, strategy legality, or fault mechanism acceptance.'}
    a.output.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': row['status'], 'npu_calls': 0, 'native_api_schema': schema,
                      'reference_B1_B8_exact_equal': True, 'input_checks': len(checks)}))


if __name__ == '__main__':
    main()
