"""CPU evidence contracts for the source87 TND2048 normal candidate.

No tensor conversion, Native admission, kernel implementation or certificate.
Public storage views and Native prepared descriptors are different boundaries.
These checks deliberately do not fill missing Native fields from public tensors.
"""
from copy import deepcopy
import hashlib
import json
import math


NORMAL_KEY = 103000000030000003
S1 = S2 = 2048
GROUP = 2
M = S2_BASE = 512
SUPPLIED = frozenset(('query', 'key', 'value', 'atten_mask', 'block_table',
    'input_layout', 'block_size', 'num_heads', 'num_key_value_heads', 'scale',
    'sparse_mode', 'actual_seq_lengths', 'actual_seq_lengths_kv'))
DEFAULTS = dict(actual_shared_prefix_len=None, antiquant_mode=0,
    antiquant_offset=None, antiquant_scale=None, dequant_scale1=None,
    dequant_scale2=None, inner_precise=0, key_antiquant_mode=0,
    key_antiquant_offset=None, key_antiquant_scale=None, key_rope=None,
    key_rope_antiquant_scale=None, key_shared_prefix=None, kv_padding_size=None,
    next_tokens=2147483647, pre_tokens=2147483647, pse_shift=None,
    quant_offset2=None, quant_scale1=None, quant_scale2=None, query_padding_size=None,
    query_rope=None, softmax_lse_flag=False, value_antiquant_mode=0,
    value_antiquant_offset=None, value_antiquant_scale=None, value_shared_prefix=None)
PUBLIC_VIEWS = {
    'query': ((2048, 16, 128), (2048, 128, 1), 0, 8388608, 'torch.float16', 2),
    'key': ((2048, 8, 128), (1024, 128, 1), 0, 4194304, 'torch.float16', 2),
    'value': ((2048, 8, 128), (4096, 128, 1), 3072, 16777216, 'torch.float16', 2),
    'atten_mask': ((2048, 2048), (2048, 1), 0, 4194304, 'torch.int8', 1),
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _same(actual, expected):
    # bool/int equality is unsuitable for actual argument evidence.
    return type(actual) is type(expected) and actual == expected


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def public_candidate(captured, row):
    """Validate an already recorded real public call; return a detached snapshot.

    The version may be null in inference mode. A null version is preserved;
    it does not prove immutability, allocation liveness or the prepared pointer.
    """
    require(row.get('actual_operator') == 'torch_npu.npu_fused_infer_attention_score'
        and type(row.get('original_calls')) is int and row['original_calls'] == 1
        and row.get('original_returned') is True, 'TND28_ACTUAL_PUBLIC_CALL_REQUIRED')
    require(set(captured['inputs']) == SUPPLIED
        and captured['supplied_argument_names'] == sorted(SUPPLIED)
        and captured['original_callable']['type'] == 'torch._ops.OpOverloadPacket'
        and type(captured['original_callable'].get('object_id')) is int
        and captured['original_callable']['object_id'] > 0,
        'TND28_PUBLIC_ARGUMENT_OR_CALLABLE_DIFFERS')
    require(isinstance(captured.get('public_default_schema'), str)
        and bool(captured['public_default_schema'])
        and set(captured['omitted_argument_defaults']) == set(DEFAULTS)
        and all(_same(captured['omitted_argument_defaults'][k], v) for k, v in DEFAULTS.items()),
        'TND28_DECLARED_DEFAULT_SCHEMA_DIFFERS')
    values = captured['inputs']
    expected = dict(input_layout='TND', block_size=128, block_table=None,
        num_heads=16, num_key_value_heads=8, sparse_mode=3,
        actual_seq_lengths=[2048], actual_seq_lengths_kv=[2048])
    for name, value in expected.items():
        require(values[name].get('kind') == 'scalar' and _same(values[name].get('value'), value),
                'TND28_PUBLIC_ARGUMENT_DIFFERS:' + name)
    scale = values['scale'].get('value')
    require(values['scale'].get('kind') == 'scalar' and type(scale) is float
        and math.isfinite(scale) and abs(scale - 128**-.5) <= 1e-12,
        'TND28_PUBLIC_SCALE_DIFFERS')
    scheduler, live = row['scheduler'], captured['live_metadata']
    require(scheduler.get('status') == 'READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE'
        and scheduler.get('internal_request_id') == row.get('request_id')
        and isinstance(row.get('request_id'), str) and bool(row['request_id'])
        and all(_same(scheduler.get(k), v) for k, v in dict(num_requests=1,
            num_computed_tokens=0, num_prompt_tokens=2048, num_scheduled_tokens=2048).items()),
        'TND28_FRESH_SINGLE_REQUEST_REQUIRED')
    require(all(_same(live.get(k), v) for k, v in dict(num_prefills=1, num_decodes=0,
        num_actual_tokens=2048, causal=True, seq_lens_list=[2048],
        actual_seq_lengths_q=[2048], mask_is_actual_metadata_object=True).items()),
        'TND28_CURRENT_CAUSAL_METADATA_REQUIRED')
    tensors, devices = {}, set()
    for name, (shape, stride, offset, storage_bytes, dtype, width) in PUBLIC_VIEWS.items():
        item = values[name]
        require(item.get('kind') == 'tensor', 'TND28_TENSOR_REQUIRED:' + name)
        snap, meta = item['snapshot'], item['snapshot']['metadata']
        require(meta.get('shape') == list(shape) and meta.get('stride') == list(stride)
            and all(type(n) is int for n in meta['shape'] + meta['stride'])
            and _same(meta.get('storage_offset'), offset)
            and _same(meta.get('storage_nbytes'), storage_bytes) and meta.get('dtype') == dtype,
            'TND28_ACTUAL_PUBLIC_VIEW_DIFFERS:' + name)
        require(all(type(meta.get(k)) is int and meta[k] > 0 for k in ('object_id', 'storage_identity'))
            and snap.get('finite') is True and _same(snap.get('bytes'), math.prod(shape)*width)
            and _sha(snap.get('sha256')) and 'version' in snap
            and (snap['version'] is None or type(snap['version']) is int and snap['version'] >= 0),
            'TND28_PUBLIC_OBJECT_OR_BYTES_UNOBSERVED:' + name)
        end = (offset + 1 + sum((n-1)*s for n, s in zip(shape, stride))) * width
        require(end <= storage_bytes, 'TND28_PUBLIC_EXTENT_OUTSIDE_STORAGE:' + name)
        device = meta.get('device')
        require(isinstance(device, str) and device.startswith('npu:')
            and device[4:].isdigit() and str(int(device[4:])) == device[4:],
            'TND28_ACTUAL_LOGICAL_DEVICE_REQUIRED')
        devices.add(device)
        tensors[name] = dict(snapshot=deepcopy(snap), first_byte=offset*width, span_end_byte=end)
    require(len(devices) == 1, 'TND28_MIXED_LOGICAL_DEVICE')
    require(len({tensors[n]['snapshot']['metadata']['object_id'] for n in tensors}) == 4,
            'TND28_PUBLIC_OPERAND_OBJECT_ALIAS')
    return dict(schema='demoncat.attention28.tnd_prefill_public_candidate.v1',
        status='PUBLIC_GEOMETRY_ONLY_NOT_NATIVE_ADMISSION', request_id=row['request_id'],
        capture_sha256=digest(captured), tensors=tensors,
        supplied_scalars={n: deepcopy(values[n]['value']) for n in sorted(expected)},
        declared_defaults=deepcopy(captured['omitted_argument_defaults']),
        native_attributes_observed=False, public_prepared_pointer_stride_join=False,
        mask_values_verified=False, stock_private_exact_proven=False,
        fault_execution_ready=False, c1=False, c2=False, c3=False)


def tnd_coordinate(flat_row, *, query_tokens=S1, group=GROUP):
    """c77 GetGS1Idx TND/SG coordinates, independent of kernel execution."""
    require(type(query_tokens) is int and 0 < query_tokens <= S1
        and type(group) is int and 0 < group <= GROUP
        and type(flat_row) is int and 0 <= flat_row < query_tokens*group,
        'TND28_INVALID_FLATTENED_ROW')
    return divmod(flat_row, group)


def normal_coordinate_matrix(*, query_tokens=S1, key_tokens=S2, group=GROUP,
                             m=M, s2_base=S2_BASE):
    """Unpruned M512/S2=512 tile plan; no denominator/replay or tensor math.

    Enumerates every flat (S1,G) row exactly once per S2 block. Causal intervals
    are analytical test data only; they are never sent to the kernel.
    """
    require(all(type(n) is int for n in (query_tokens, key_tokens, group, m, s2_base))
        and (query_tokens, key_tokens, group, m, s2_base) == (S1, S2, GROUP, M, S2_BASE),
        'TND28_NORMAL_ONLY_EXACT_GEOMETRY_REQUIRED')
    tiles = []
    for flat_start in range(0, query_tokens*group, m):
        rows = min(m, query_tokens*group-flat_start)
        first_q, first_g = tnd_coordinate(flat_start)
        last_q, last_g = tnd_coordinate(flat_start+rows-1)
        for key_start in range(0, key_tokens, s2_base):
            key_end = min(key_tokens, key_start+s2_base)
            allowed = sum(max(0, min(key_end, tnd_coordinate(r)[0]+1)-key_start)
                          for r in range(flat_start, flat_start+rows))
            tiles.append(dict(flat_start=flat_start, rows=rows, first_query=first_q,
                first_group=first_g, last_query=last_q, last_group=last_g,
                key_start=key_start, key_end=key_end, unmasked_pairs=allowed))
    return dict(schema='demoncat.attention28.tnd_normal_coordinate_contract.v1',
        tiling_key=str(NORMAL_KEY), m=m, s2_base=s2_base, tiles=tiles,
        unmasked_pairs_per_kv_head=sum(t['unmasked_pairs'] for t in tiles),
        actual_split_core_traversal_verified=False, kernel_executed=False)
