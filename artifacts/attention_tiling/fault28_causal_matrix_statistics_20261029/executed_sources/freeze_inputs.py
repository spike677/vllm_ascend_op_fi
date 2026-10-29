"""Freeze batch0 logical FP16 inputs from an existing, immutable model capture. CPU only."""
import argparse
import hashlib
import json
from pathlib import Path

CALL_SHA = '458a683a616da397bd1eeba47b7348d0e2bf78c253ea43c158f7f393bb67c0c8'
TARGET_SHAS = {'query': '45755102b0f5ac710aa9ff30ab3bfe2ad69b66041281a1671eaccb144557dc00',
               'key': '73d1dec27ffe3873678a9daff7d422d51346997427fbc64279dbc0bfa8156fde',
               'value': '339fdb463b0ef9ba0931c739b2ee3db1fc705e1b6a5c28f0fd5daf6fea915947'}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def freeze(source, output):
    source = source.resolve(strict=True)
    call_file = source / 'native_calls/call_001.json'
    require(sha(call_file) == CALL_SHA, 'Wrong original capture metadata')
    call = json.loads(call_file.read_bytes())
    require(call['case_id'] == 'fa_off_b8' and call['forward_index'] == 0 and call['layer_index'] == 1,
            'Expected original OFF model layer1 forward0')
    require(call['scalar_arguments'] == {'dropout_p': 0.0, 'is_causal': True,
            'scale': 0.08838834764831845, 'enable_gqa': True} and call['attention_mask'] is None,
            'Original native lowering contract changed')
    require(not output.exists(), 'Frozen input directory must be new')
    rows, payloads = {}, {}
    for name in ('query', 'key', 'value'):
        record = call['selected_layer_private_inputs'][name]
        expected_shape = [8, 16 if name == 'query' else 8, 2048, 128]
        require(record['shape'] == expected_shape and record['dtype'] == 'torch.float16'
                and record['byteorder'] == 'little' and record['raw_order'] == 'C-contiguous logical elements',
                'Unexpected original input representation')
        file = source / record['private_raw_file']
        require(file.resolve(strict=True).is_relative_to(source), 'Original input path escapes source')
        raw = file.read_bytes()
        require(len(raw) == record['logical_bytes'] and sha_bytes(raw) == record['logical_sha256'],
                'Original complete input file differs')
        target = raw[:len(raw) // 8]
        require(sha_bytes(target) == TARGET_SHAS[name] == call['target_inputs_before'][name]['logical_sha256']
                == call['target_inputs_after'][name]['logical_sha256'], 'Target batch0 source differs')
        payloads[name] = target
        rows[name] = {'file': name + '.fp16.bin', 'shape': [1, *expected_shape[1:]], 'bytes': len(target),
                      'sha256': sha_bytes(target), 'dtype': 'torch.float16', 'byteorder': 'little',
                      'raw_order': 'C-contiguous logical elements', 'source_file': str(file),
                      'source_bytes': len(raw), 'source_sha256': sha_bytes(raw), 'source_shape': expected_shape,
                      'source_stride': record['stride'], 'source_actual_format': record['actual_format'],
                      'batch_rule': 'B1=batch0; B8=8 byte-identical copies of original batch0 in logical BNSD order',
                      'expected_full_sha256': {'1': sha_bytes(target), '8': sha_bytes(target * 8)}}
    output.mkdir(parents=True)
    for name, data in payloads.items():
        (output / rows[name]['file']).write_bytes(data)
    (output / 'source_call.json').write_bytes(call_file.read_bytes())
    manifest = {'schema': 'fault28.causal.inputs.v1', 'source_call_sha256': CALL_SHA,
                'source_case_directory': str(source), 'source_call_file': 'source_call.json',
                'source': 'Actual Qwen3 model layer1 OFF capture; no synthetic values or outlier perturbations.',
                'target_batch_slot': 0, 'sequence_length': 2048, 'query_heads': 16, 'kv_heads': 8,
                'head_dim': 128, 'inputs': rows,
                'source_sdpa_scalar_arguments': call['scalar_arguments'],
                'native_kwargs': {'head_num': 16, 'input_layout': 'BNSD', 'scale': call['scalar_arguments']['scale'],
                    'keep_prob': 1.0, 'pre_tockens': 2147483647, 'next_tockens': 0,
                    'inner_precise': 0, 'sparse_mode': 2, 'gen_mask_parallel': True, 'sync': False,
                    'softmax_layout': ''},
                'native_mask': {'dtype': 'torch.bool', 'shape': [2048, 2048],
                                'rule': 'strict upper triangle true; same compressed causal mask as native SDPA lowering'},
                'publication': 'PRIVATE_INPUT_BINARY_DO_NOT_PUBLISH'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    print(json.dumps({'status': 'INPUTS_FROZEN_CPU_ONLY', 'manifest_sha256': sha(output / 'manifest.json'),
                      'target_sha256': TARGET_SHAS}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    freeze(a.source_case, a.output)
