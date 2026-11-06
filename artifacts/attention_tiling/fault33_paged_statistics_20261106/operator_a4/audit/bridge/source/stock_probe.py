"""Finite stock paged-IFA discovery with fixed logical inputs and private outputs.

This is discovery, not fault acceptance. Every bracket contains one original API
call. Different cache geometries are verified through explicit block mappings;
their native msprobe metadata must not be silently treated as identical.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def payload(tensor):
    import torch
    return tensor.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()


def record(tensor):
    b = payload(tensor)
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'stride': list(tensor.stride()), 'storage_offset': tensor.storage_offset(),
            'logical_bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(), 'byteorder': sys.byteorder}


def physical_cache(logical, block_size):
    import torch
    batch, heads, length, dim = logical.shape
    count = math.ceil(length / block_size)
    padded = torch.zeros((batch, heads, count * block_size, dim), dtype=logical.dtype)
    padded[:, :, :length, :] = logical
    blocks = padded.view(batch, heads, count, block_size, dim).permute(0, 2, 1, 3, 4).contiguous()
    blocks = blocks.view(batch * count, heads, block_size, dim)
    table = torch.arange(batch * count - 1, -1, -1, dtype=torch.int32).view(batch, count)
    physical = blocks.flip(0).contiguous()
    recovered = physical[table.long()].permute(0, 2, 1, 3, 4).contiguous().view(batch, heads, count * block_size, dim)
    require(torch.equal(recovered[:, :, :length, :], logical), 'Block table changes logical data')
    return physical, table


def make_logical(case):
    import torch
    batch, qheads, kvheads, length, dim = [case[k] for k in ('batch', 'q_heads', 'kv_heads', 'kv_length', 'head_dim')]
    qs, ks, vs = [], [], []
    for request in range(batch):
        generator = torch.Generator(device='cpu').manual_seed(case['seed'] + 104729 * request)
        qs.append(torch.randn((qheads, 1, dim), generator=generator, dtype=torch.float32).half())
        ks.append(torch.randn((kvheads, length, dim), generator=generator, dtype=torch.float32).half())
        vs.append(torch.randn((kvheads, length, dim), generator=generator, dtype=torch.float32).half())
    return torch.stack(qs), torch.stack(ks), torch.stack(vs)


def compare(a, b):
    import torch
    require(a.shape == b.shape, 'Comparison requires full aligned output')
    bitwise_equal = payload(a) == payload(b) if a.dtype == b.dtype else None
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    idx = torch.nonzero(a != b, as_tuple=False)
    return {'shape': list(a.shape), 'elements': a.numel(), 'different_elements': len(idx),
            'bitwise_equal': bitwise_equal,
            'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(),
            'l2': torch.linalg.vector_norm(a - b).item(),
            'first_indices': idx[:16].tolist(),
            'all_finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            'request_different_elements': [(a[i] != b[i]).sum().item() for i in range(a.shape[0])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    case = json.loads(Path(args.case_file).read_bytes())
    require(case['schema'] == 'fault33.stock_paged_discovery.case.v1', 'Unexpected case')
    require(case['dtype'] == 'float16' and case['inner_precise'] == 0, 'Pinned high precision FP16 case required')
    require(case['batch'] in (1, 4, 8, 16) and case['q_heads'] % case['kv_heads'] == 0,
            'Invalid finite batch/GQA geometry')
    require(case['kv_length'] in (2048, 4096, 8192, 16384), 'Unregistered length')
    require(case['block_sizes'] and len(case['block_sizes']) <= 4
            and all(type(v) is int and v in (64, 80, 112, 128, 256, 512) for v in case['block_sizes']), 'Unregistered block sizes')
    require(case['repeats'] in (2, 3), 'Finite repeat plan required')
    out = Path(args.output)
    out.mkdir(exist_ok=False, parents=True)
    (out / 'private_output').mkdir()
    write(out / 'case.json', case)
    import torch
    import torch_npu
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(0)
    torch.npu.synchronize()
    q_cpu, k_cpu, v_cpu = make_logical(case)
    logical = {'q': record(q_cpu), 'k': record(k_cpu), 'v': record(v_cpu)}
    logical['target_request_00'] = {'q': record(q_cpu[:1]), 'k': record(k_cpu[:1]), 'v': record(v_cpu[:1])}
    write(out / 'logical_inputs.json', logical)
    q = q_cpu.to('npu')
    qh = case['q_heads']
    kh = case['kv_heads']
    reference_k = k_cpu.repeat_interleave(qh // kh, dim=1).double()
    reference_v = v_cpu.repeat_interleave(qh // kh, dim=1).double()
    reference = ((q_cpu.double() @ reference_k.transpose(-1, -2)) / math.sqrt(case['head_dim'])).softmax(-1) @ reference_v
    write(out / 'reference.json', {'dtype': 'float64', 'formula': 'softmax(Q @ K.T / sqrt(D)) @ V', **record(reference)})
    all_outputs, calls = {}, []
    original_api = torch_npu.npu_incre_flash_attention
    for block_size in case['block_sizes']:
        kc, table = physical_cache(k_cpu, block_size)
        vc, table_v = physical_cache(v_cpu, block_size)
        require(torch.equal(table, table_v), 'K/V block maps differ')
        kn, vn, tn = [t.to('npu') for t in (kc, vc, table)]
        entry = {'block_size': block_size, 'block_table': table.tolist(),
                 'physical_key': record(kn), 'physical_value': record(vn), 'table': record(tn),
                 'logical_key_sha256': logical['k']['sha256'], 'logical_value_sha256': logical['v']['sha256']}
        write(out / ('cache_' + str(block_size) + '.json'), entry)
        for repeat in range(case['repeats']):
            call_id = 'bs' + str(block_size) + '_r' + str(repeat)
            before = {name: record(t) for name, t in [('query', q), ('key', kn), ('value', vn), ('table', tn)]}
            marker = {'fault33': 'enter', 'call_id': call_id, 'pid': os.getpid(),
                      'tid': threading.get_native_id(), 'block_size': block_size,
                      'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
            torch.npu.synchronize()
            print(json.dumps(marker), flush=True)
            start = time.monotonic()
            output = original_api(q, kn, vn, block_table=tn,
                actual_seq_lengths=[case['kv_length']] * case['batch'],
                num_heads=qh, num_key_value_heads=kh, input_layout='BNSD',
                scale_value=1.0 / math.sqrt(case['head_dim']), block_size=block_size, inner_precise=0)
            torch.npu.synchronize()
            print(json.dumps({**marker, 'fault33': 'exit', 'elapsed_seconds': time.monotonic() - start}), flush=True)
            after = {name: record(t) for name, t in [('query', q), ('key', kn), ('value', vn), ('table', tn)]}
            require(before == after, 'Original API changed input logical bytes/geometry')
            output_cpu = output.detach().cpu().contiguous()
            require(bool(torch.isfinite(output_cpu).all()), 'Non-finite native output')
            b = payload(output_cpu)
            private_file = out / 'private_output' / (call_id + '.bin')
            private_file.write_bytes(b)
            call = {'call_id': call_id, 'api': 'torch_npu.npu_incre_flash_attention', 'original_api_calls': 1,
                    'pid': marker['pid'], 'tid': marker['tid'], 'block_size': block_size,
                    'deterministic_algorithms': marker['deterministic_algorithms'],
                    'before': before, 'after': after, 'output': record(output_cpu),
                    'private_full_output': str(private_file.relative_to(out)),
                    'reference_comparison': compare(reference, output_cpu)}
            write(out / (call_id + '.json'), call)
            calls.append(call)
            all_outputs[call_id] = output_cpu
        del kn, vn, tn
    comparisons = []
    names = list(all_outputs)
    for name in names[1:]:
        comparisons.append({'a': names[0], 'b': name, **compare(all_outputs[names[0]], all_outputs[name])})
    stability = {str(size): all(payload(all_outputs['bs' + str(size) + '_r0']) ==
                               payload(all_outputs['bs' + str(size) + '_r' + str(i)])
                               for i in range(1, case['repeats'])) for size in case['block_sizes']}
    write(out / 'result.json', {'schema': 'fault33.stock_paged_discovery.result.v1',
          'status': 'CAPTURE_COMPLETED', 'case_id': case['case_id'], 'calls': calls,
          'comparisons': comparisons, 'logical_inputs': logical,
          'mechanism_accepted': False, 'software': {'torch': torch.__version__, 'torch_npu': torch_npu.__version__},
          'same_plan_repeat_stability': stability, 'same_plan_repeats_stable': all(stability.values()),
          'scope': 'Stock discovery only; different physical layouts require logical mapping and actual tiling verification.'})
    print(json.dumps({'status': 'CAPTURE_COMPLETED', 'case_id': case['case_id'], 'calls': len(calls),
                      'different_counts': [x['different_elements'] for x in comparisons]}), flush=True)


if __name__ == '__main__':
    main()
