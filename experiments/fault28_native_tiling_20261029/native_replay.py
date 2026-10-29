"""Stock native FIA calibration; synthetic input is never model-path evidence.

This initial entry checks the CANN logging/tiling capture path. Actual captured
model inputs require a separately validated reader before mechanism acceptance.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def marker(event, **fields):
    print(json.dumps({'event': event, 'pid': os.getpid(), 'epoch_ns': time.time_ns(), **fields}), flush=True)


def payload(tensor):
    cpu = tensor.detach().contiguous().cpu()
    data = cpu.view(__import__('torch').uint8).numpy().tobytes()
    return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype), 'sha256': hashlib.sha256(data).hexdigest()}, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case_file.read_text(encoding='utf-8'))
    assert case['mode'] == 'synthetic_native_calibration' and case['operator_variant'] == 'stock'
    assert case['batch_size'] in (1, 8) and case['prompt_length'] in (256, 512, 1024, 2048)
    assert os.environ.get('ACLNN_CACHE_LIMIT') == '0'
    assert not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'case_id': case['case_id'], 'status': 'RUNNING', 'source_case_sha256': hashlib.sha256(args.case_file.read_bytes()).hexdigest(),
              'mode': case['mode'], 'synthetic_not_model_capture': True, 'mechanism_accepted': False,
              'input_unchanged': None, 'calls': [], 'native_calls': [], 'start_epoch': time.time()}
    try:
        import torch
        import torch_npu
        torch_npu.npu.set_device(0)
        b, s, nq, nkv, d = case['batch_size'], case['prompt_length'], 16, 8, 128
        generator = torch.Generator(device='cpu').manual_seed(280029)
        # Fixed CPU seed and target row data shared by B1/B8. No perturbation.
        cpu_inputs = [torch.randn((s, n, d), generator=generator, dtype=torch.float32).half() for n in (nq, nkv, nkv)]
        inputs = [t.repeat((b, 1, 1)).to('npu:0') for t in cpu_inputs]
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool), diagonal=1).to('npu:0')
        lengths = [s * (i + 1) for i in range(b)]
        kwargs = {'num_heads': nq, 'num_key_value_heads': nkv, 'input_layout': 'TND',
                  'actual_seq_lengths': lengths, 'actual_seq_lengths_kv': lengths,
                  'atten_mask': mask, 'scale': 1 / math.sqrt(d), 'pre_tokens': 2147483647,
                  'next_tokens': 0, 'sparse_mode': 3, 'inner_precise': 0, 'block_size': 0}
        result['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torch-npu')}
        result['shape_configuration'] = {'batch': b, 'length': s, 'query_heads': nq, 'kv_heads': nkv, 'head_dim': d,
                                         'layout': 'TND', 'causal': True, 'actual_lengths': lengths, 'runtime_dtype': 'float16'}
        before = []
        for name, tensor in zip(('query', 'key', 'value'), inputs):
            info, _ = payload(tensor[:s])
            info.update(stride=list(tensor.stride()), storage_offset=tensor.storage_offset(), device=str(tensor.device))
            before.append(info)
        result['target_input'] = before
        for call in range(2):
            torch_npu.npu.synchronize()
            context = {'case_id': case['case_id'], 'thread_id': threading.get_native_id(),
                       'rank': 0, 'native_index': call, 'call_id': f'pid{os.getpid()}:r0:n{call}',
                       'layer_index': -1, 'layer_name': 'synthetic_not_model',
                       'api': 'torch_npu.npu_fused_infer_attention_score'}
            marker('native_api_enter', **context, batch=b, sequence_length=s)
            start = time.time_ns()
            native = torch_npu.npu_fused_infer_attention_score(*inputs, **kwargs)
            torch_npu.npu.synchronize()
            end = time.time_ns()
            marker('native_api_exit', **context, status='returned', start_epoch_ns=start, end_epoch_ns=end)
            result['native_calls'].append({'pid': os.getpid(), **context, 'status': 'returned',
                                           'start_epoch_ns': start, 'end_epoch_ns': end})
            output = native[0]
            assert output.dtype == torch.float16 and output.shape[0] == b * s
            info, data = payload(output[:s])
            private = args.output / 'private_replay'
            private.mkdir(exist_ok=True)
            filename = 'call' + str(call) + '_target_attention.bin'
            (private / filename).write_bytes(data)
            info['private_file'] = 'private_replay/' + filename
            result['calls'].append({'call': call, 'start_epoch_ns': start, 'end_epoch_ns': end,
                                    'api': 'torch_npu.npu_fused_infer_attention_score', 'target_output': info})
        after = [payload(t[:s])[0] for t in inputs]
        result['input_unchanged'] = all(a['sha256'] == z['sha256'] for a, z in zip(before, after))
        assert result['input_unchanged']
        result['repeat_outputs_equal'] = result['calls'][0]['target_output']['sha256'] == result['calls'][1]['target_output']['sha256']
        result['status'] = 'CAPTURE_COMPLETED'
        result['end_epoch'] = time.time()
        save(args.output / 'result.json', result)
        print(json.dumps({'case_id': case['case_id'], 'status': result['status'], 'repeat_outputs_equal': result['repeat_outputs_equal'],
                          'mechanism_accepted': False, 'synthetic_not_model_capture': True}), flush=True)
        return 0
    except BaseException:
        result.update(status='FAIL', traceback=traceback.format_exc(), end_epoch=time.time())
        save(args.output / 'result.json', result)
        print(result['traceback'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
