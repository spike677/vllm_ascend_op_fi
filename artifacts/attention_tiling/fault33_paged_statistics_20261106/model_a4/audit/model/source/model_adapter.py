"""Bridge the actual Qwen decode SDPA arguments to one paged IFA per layer."""
from contextlib import contextmanager
import functools
import hashlib
import json
import math
import os
import sys
import threading
import time

from replay import host_scope
from stock_probe import require


def bind_decode(args, kwargs):
    names = ('query', 'key', 'value', 'attn_mask', 'dropout_p', 'is_causal')
    require(len(args) <= len(names) and not (set(names[:len(args)]) & set(kwargs)),
            'Ambiguous SDPA arguments')
    require(set(kwargs) <= set(names) | {'scale', 'enable_gqa'}, 'Unknown SDPA argument')
    fields = dict(attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)
    fields.update(zip(names, args))
    fields.update(kwargs)
    require(fields['attn_mask'] is None and fields['is_causal'] is False
            and fields['dropout_p'] == 0 and fields['enable_gqa'] is True,
            'Only an unmasked, final-position GQA decode is registered')
    scale = fields['scale'] if fields['scale'] is not None else 128 ** -0.5
    require(math.isclose(scale, 128 ** -0.5, rel_tol=0, abs_tol=1e-12), 'Unexpected SDPA scale')
    return fields['query'], fields['key'], fields['value'], scale


def cpu_array(tensor):
    # Read one request or a small output. This fast path never copies an entire KV batch to host.
    require(tensor.numel() * tensor.element_size() <= 32 * 1024**2, 'Unbounded diagnostic transfer')
    return tensor.detach().cpu().contiguous().numpy()


def array_digest(array):
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def tensor_report(tensor, *, destination=None, finite=False):
    import numpy as np
    current = hashlib.sha256()
    rows = []
    size = 0
    stream = destination.open('xb') if destination is not None else None
    try:
        for i in range(tensor.shape[0]):
            array = cpu_array(tensor[i])
            if finite:
                require(np.isfinite(array).all(), 'Nonfinite model output')
            chunk = memoryview(array).cast('B')
            current.update(chunk)
            rows.append(array_digest(array))
            size += len(chunk)
            if stream is not None:
                stream.write(chunk)
    finally:
        if stream is not None:
            stream.close()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'bytes': size,
            'sha256': current.hexdigest(), 'request_sha256': rows, 'byteorder': sys.byteorder}


def pack_pages(logical, block_size):
    b, h, length, d = logical.shape
    require(length % block_size == 0, 'Exact whole pages required')
    return logical.reshape(b, h, length // block_size, block_size, d).permute(
        0, 2, 1, 3, 4).contiguous().view(b * (length // block_size), h, block_size, d)


def physical_report(pages, table, logical, prefix_rows):
    """Inverse-map each actual table row; compare every represented FP16 element by digest."""
    import numpy as np
    full = hashlib.sha256()
    physical = hashlib.sha256()
    logical_rows = []
    page_rows = []
    for i, ids in enumerate(table):
        count = len(ids)
        require(ids == list(range(i * count, (i + 1) * count)), 'Unexpected physical block map')
        # The range above is the table row, not an assumption about private cache storage.
        mapped = cpu_array(pages.narrow(0, ids[0], count))
        page_rows.append(array_digest(mapped))
        physical.update(memoryview(mapped).cast('B'))
        back = np.ascontiguousarray(mapped.transpose(1, 0, 2, 3)).reshape(
            logical.shape[1], logical.shape[2], logical.shape[3])
        original = cpu_array(logical[i])
        logical_digest = array_digest(original)
        require(array_digest(back) == logical_digest, 'Native pages differ from actual attention KV')
        prefix = np.ascontiguousarray(original[:, :-1, :])
        require(array_digest(prefix) == prefix_rows[i], 'Cache prefix differs from real prefill')
        logical_rows.append(logical_digest)
        full.update(memoryview(original).cast('B'))
    return {'logical_sha256': full.hexdigest(), 'physical_sha256': physical.hexdigest(),
            'request_sha256': logical_rows, 'physical_request_sha256': page_rows,
            'logical_shape': list(logical.shape), 'physical_shape': list(pages.shape),
            'dtype': str(logical.dtype), 'prefill_prefix_verified': True, 'inverse_verified': True}


def physical_rows_after(pages, table):
    return [array_digest(cpu_array(pages.narrow(0, row[0], len(row)))) for row in table]


def event(kind, **fields):
    row = {'fault33': kind, 'pid': os.getpid(), 'tid': threading.get_native_id(),
           'monotonic_ns': time.monotonic_ns(), **fields}
    print(json.dumps(row, allow_nan=False), flush=True)
    return row


def cache_layer(cache, index):
    item = cache.layers[index]
    return item.keys, item.values


@contextmanager
def observe_prefill(model, functional, torch):
    """Count all original prefill calls, independently of decode injection or msprobe."""
    require(not any(k.startswith('FAULT33_PA_') for k in os.environ), 'Prefill inherited host scope')
    original = functional.scaled_dot_product_attention
    calls, handles = [], []
    active = None

    def enter(index):
        def hook(module, args):
            nonlocal active
            require(active is None and len(calls) == index, 'Unregistered prefill attention order')
            active = index
        return hook

    def exit_layer(index):
        def hook(module, args, output):
            nonlocal active
            require(active == index and len(calls) == index + 1, 'Prefill layer missed its original SDPA')
            active = None
        return hook

    @functools.wraps(original)
    def measured(*args, **kwargs):
        require(active is not None and active == len(calls), 'Unbound/duplicate prefill SDPA')
        fields = dict(zip(('query', 'key', 'value', 'attn_mask', 'dropout_p', 'is_causal'), args))
        fields.update(kwargs)
        q, k, v = (fields[key] for key in ('query', 'key', 'value'))
        require(list(q.shape) == [8, 16, 16383, 128] and list(k.shape) == list(v.shape) == [8, 8, 16383, 128]
                and all(x.dtype == torch.float16 and x.device.type == 'npu' for x in (q, k, v))
                and fields.get('attn_mask') is None and fields.get('is_causal') is True
                and fields.get('enable_gqa') is True and fields.get('dropout_p', 0) == 0,
                'Unexpected real prefill attention descriptors')
        result = original(*args, **kwargs)
        require(result.shape == q.shape and result.dtype == q.dtype, 'Unexpected original SDPA output')
        calls.append({'layer_index': active, 'api': 'torch.nn.functional.scaled_dot_product_attention',
                      'original_api_calls': 1, 'scope_active': False, 'query_shape': list(q.shape),
                      'key_shape': list(k.shape), 'value_shape': list(v.shape), 'output_shape': list(result.shape),
                      'dtype': str(q.dtype), 'is_causal': True, 'enable_gqa': True,
                      'attn_mask': None, 'dropout_p': 0, 'scale': fields.get('scale')})
        return result

    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(enter(i)))
            handles.append(layer.self_attn.register_forward_hook(exit_layer(i)))
        functional.scaled_dot_product_attention = measured
        yield calls
        require(active is None and [c['layer_index'] for c in calls] == list(range(28)), 'Incomplete prefill')
    finally:
        for handle in handles:
            handle.remove()
        require(functional.scaled_dot_product_attention is measured, 'Unexpected replacement during prefill')
        functional.scaled_dot_product_attention = original


class PagedDecode:
    """Installed only for one actual model decode. Prefill uses the original HF route."""
    def __init__(self, torch, npu, model, cache, prefix, table, case, output, request_ids):
        self.torch, self.npu, self.model, self.cache = torch, npu, model, cache
        self.prefix, self.table = prefix, table
        self.mapping = table.cpu().tolist()
        self.case, self.output, self.request_ids = case, output, request_ids
        self.calls, self.visits, self.exits = [], [], []
        self.active = None

    def enter(self, index):
        def hook(module, args):
            require(self.active is None and index == len(self.visits), 'Unexpected Qwen attention visit')
            self.active = index
            self.visits.append(index)
            event('module_enter', case_id=self.case['case_id'], layer_index=index, request_ids=self.request_ids)
        return hook

    def leave(self, index):
        def hook(module, args, output):
            require(self.active == index and len(self.calls) == index + 1
                    and 'o_proj' in self.calls[index], 'Real attention left without IFA/o_proj evidence')
            self.active = None
            self.exits.append(index)
            event('module_exit', case_id=self.case['case_id'], layer_index=index)
        return hook

    def projection(self, index):
        def hook(module, args, output):
            require(self.active == index and len(self.calls) == index + 1
                    and 'o_proj' not in self.calls[index], 'Unbound or duplicate output projection')
            row = self.calls[index]
            require(len(args) == 1 and list(args[0].shape) == [8, 1, 2048], 'Projection input shape')
            consumed = tensor_report(args[0], finite=True)
            require(consumed['request_sha256'] == row['attention']['request_sha256'],
                    'Projection consumed different attention output bytes')
            require(list(output.shape) == [8, 1, 1024], 'Projection output shape')
            row['o_proj_input'] = consumed
            row['o_proj'] = tensor_report(output, finite=True)
        return hook

    def call(self, *args, **kwargs):
        idx = self.active
        require(idx is not None and len(self.calls) == idx, 'Outside or duplicate SDPA during decode')
        q, k, v, scale = bind_decode(args, kwargs)
        require(list(q.shape) == [8, 16, 1, 128] and list(k.shape) == list(v.shape) == [8, 8, 16384, 128]
                and all(t.dtype == self.torch.float16 and t.device.type == 'npu' for t in (q, k, v)),
                'Unregistered actual model attention descriptors')
        real_cache = cache_layer(self.cache, idx)
        for tensor, stored in zip((k, v), real_cache):
            require(tensor.shape == stored.shape and tensor.dtype == stored.dtype and tensor.device == stored.device,
                    'Model cache descriptors differ')
            if tensor.data_ptr() != stored.data_ptr() or tensor.stride() != stored.stride():
                require(tensor_report(tensor) == tensor_report(stored), 'SDPA KV differs from actual updated cache')
        pk, pv = pack_pages(k, 128), pack_pages(v, 128)
        q_before = tensor_report(q)
        inputs = {name: physical_report(pages, self.mapping, logical, self.prefix[idx][name]['request_sha256'])
                  for name, pages, logical in [('key', pk, k), ('value', pv, v)]}
        call_id = f'decode_layer_{idx:02d}'
        with host_scope(self.case, self.output, call_id) as scope:
            marker = dict(call_id=call_id, case_id=self.case['case_id'], layer_index=idx,
                          scope_token=scope['FAULT33_PA_SCOPE_TOKEN'], block_size=128,
                          deterministic_algorithms=self.torch.are_deterministic_algorithms_enabled())
            self.npu.npu.synchronize()
            entered = event('enter', **marker)
            attended = self.npu.npu_incre_flash_attention(
                q, pk, pv, block_table=self.table, actual_seq_lengths=[16384] * 8, num_heads=16,
                num_key_value_heads=8, input_layout='BNSD', scale_value=scale, block_size=128, inner_precise=0)
            self.npu.npu.synchronize()
            exited = event('exit', **marker)
        require(self.table.cpu().tolist() == self.mapping and tensor_report(q) == q_before, 'IFA changed Q/table')
        for name, pages in [('key', pk), ('value', pv)]:
            require(physical_rows_after(pages, self.mapping) == inputs[name]['physical_request_sha256'],
                    'IFA changed physical KV pages')
        del pk, pv
        require(attended.shape == q.shape and attended.dtype == q.dtype and attended.device == q.device,
                'Unexpected native IFA return geometry')
        self.calls.append(dict(call_id=call_id, layer_index=idx, request_ids=self.request_ids,
                               pid=entered['pid'], tid=entered['tid'], scope_token=marker['scope_token'],
                               enter=entered, exit=exited, query=q_before, kv=inputs,
                               actual_seq_lengths=[16384] * 8, original_api_calls=1,
                               native_inputs_unchanged=True, attention=tensor_report(attended, finite=True)))
        return attended

    @contextmanager
    def install(self, functional):
        original = functional.scaled_dot_product_attention
        wrapper = functools.wraps(original)(lambda *args, **kwargs: self.call(*args, **kwargs))
        handles = []
        try:
            for idx, layer in enumerate(self.model.model.layers):
                attn = layer.self_attn
                require(type(attn).__name__ == 'Qwen3Attention' and attn.layer_idx == idx, 'Unknown module mapping')
                handles.append(attn.register_forward_pre_hook(self.enter(idx)))
                handles.append(attn.register_forward_hook(self.leave(idx)))
                handles.append(attn.o_proj.register_forward_hook(self.projection(idx)))
            functional.scaled_dot_product_attention = wrapper
            yield
        finally:
            for handle in handles:
                handle.remove()
            require(functional.scaled_dot_product_attention is wrapper, 'Unexpected SDPA replacement')
            functional.scaled_dot_product_attention = original
