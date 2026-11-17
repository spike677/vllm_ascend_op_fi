"""Read-only, source-bound audit of natural paged decode calls.

This is deliberately not a Fault33 injection Provider. The reviewed 8.5.1
Worker calls TND FIA; the existing Fault33 native qualification is for a
different 9.1 BNSD IFA bundle. This observer cannot confer that qualification.
It reads actual page/slot metadata and observes the real cache writer and
consumer, without repacking KV, replacing a runner method, or changing an
operator argument. Diagnostic device-to-host copies contain only indices.
"""
import hashlib
import inspect
import threading

from .serving_context import require, tensor_metadata, require_serial_eager


ATTENTION_SOURCE_SHA256 = 'bbe69c3015ee15789dea0d6ec9795243458b68494294c4c023ed5a5d665c550b'
LAYER_SOURCE_SHA256 = '990ead07c828e26192ca46b4b8abae7a6792f67c27510705ad7b483035f3631c'
APIS = ('npu_fused_infer_attention_score', 'npu_fused_infer_attention_score_v2',
        '_npu_paged_attention', 'npu_incre_flash_attention')
_ACTIVE = None


def _meta(tensor):
    result = tensor_metadata(tensor)
    result['storage_nbytes'] = int(tensor.untyped_storage().nbytes())
    return result


def _same_storage(left, right):
    a, b = _meta(left), _meta(right)
    return a['device'] == b['device'] and a['storage_identity'] == b['storage_identity']


def _flat_alias(actual, cache):
    a, b = _meta(actual), _meta(cache)
    require(_same_storage(actual, cache) and a['storage_offset'] == b['storage_offset']
            and actual.numel() == cache.numel() and actual.is_contiguous()
            and cache.is_contiguous(), 'LIVE_CACHE_ALIAS_MISMATCH')
    return a


def _disjoint(left, right):
    if not _same_storage(left, right):return True
    require(left.is_contiguous() and right.is_contiguous(), 'CACHE_STRIDING_UNADAPTED')
    a, b = left.storage_offset() * left.element_size(), right.storage_offset() * right.element_size()
    return a + left.numel() * left.element_size() <= b or b + right.numel() * right.element_size() <= a


def _indices(tensor, label):
    import torch
    require(isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.int32, torch.int64),
            'INTEGER_METADATA_REQUIRED:' + label)
    require(tensor.numel() <= 1048576, 'METADATA_BOUND_EXCEEDED:' + label)
    # Actual indices are checked; tensor values for Q/K/V are never transferred.
    return tensor.detach().cpu().tolist()


def _native_route(operator, layout):
    if operator == 'npu_fused_infer_attention_score' and layout == 'TND':
        return 'OBSERVED_TND_FIA_REQUIRES_OWN_NATIVE_ADAPTER'
    if operator == '_npu_paged_attention':
        return 'OBSERVED_PA_REQUIRES_OWN_NATIVE_ADAPTER'
    if operator == 'npu_incre_flash_attention' and layout == 'BNSD':
        return 'BNSD_IFA_PRESENT_BUT_VENDOR_LAUNCH_AND_WORKER_QUALIFICATION_MISSING'
    return 'OBSERVED_API_LAYOUT_REQUIRES_OWN_NATIVE_ADAPTER'


def audit_paged_call(operator, actual, *, impl, metadata, scheduler, writer, allow_cpu_test=False):
    """Cross-check actual consumer, live cache, actual writer and row mapping.

    CPU fixtures exercise invalid-page/slot rejection only. Success here proves
    neither cache contents nor native strategy selection/launch.
    """
    require(operator in APIS, 'UNKNOWN_ATTENTION_API')
    query = actual['query']
    key = actual.get('key', actual.get('key_cache'))
    value = actual.get('value', actual.get('value_cache'))
    cache_k, cache_v = impl.key_cache, impl.value_cache
    require(cache_k is not None and cache_v is not None, 'LIVE_PAGED_CACHE_MISSING')
    require(len(cache_k.shape) == 4 and tuple(cache_k.shape) == tuple(cache_v.shape),
            'PAGED_CACHE_GEOMETRY_UNADAPTED')
    blocks, block_size, kv_heads, dim = map(int, cache_k.shape)
    require(blocks > 0 and block_size > 0 and kv_heads == impl.num_kv_heads and dim == impl.head_size,
            'IMPL_CACHE_GEOMETRY_MISMATCH')
    require(_disjoint(cache_k, cache_v), 'KEY_VALUE_CACHE_STORAGE_COLLISION')
    device = str(query.device).split(':')[0]
    require(device == 'npu' or (allow_cpu_test and device == 'cpu'), 'NPU_QUERY_REQUIRED')
    require(all(str(x.device) == str(query.device) and x.dtype == query.dtype
                for x in (cache_k, cache_v, key, value)), 'CACHE_DEVICE_DTYPE_MISMATCH')
    key_meta, value_meta = _flat_alias(key, cache_k), _flat_alias(value, cache_v)
    table = actual.get('block_table')
    require(table is not None and _same_storage(table, metadata.block_tables)
            and table.storage_offset() == metadata.block_tables.storage_offset(),
            'ACTUAL_BLOCK_TABLE_NOT_METADATA')
    count = scheduler['request_count']
    require(scheduler['counts'] == [1] * count and scheduler['token_row_count'] == count,
            'PURE_SINGLE_TOKEN_DECODE_REQUIRED')
    require(getattr(metadata.attn_state, 'name', None) == 'DecodeOnly'
            and metadata.num_actual_tokens == count, 'METADATA_NOT_PURE_DECODE')
    require(getattr(impl, 'sliding_window', None) is None and getattr(impl, 'sinks', None) is None,
            'SLIDING_OR_SINK_UNADAPTED')
    positions = scheduler['row_token_positions']
    lengths = list(metadata.seq_lens_list)
    require(lengths == [position + 1 for position in positions], 'ACTUAL_SEQUENCE_LENGTH_MISMATCH')
    q_lengths = list(metadata.actual_seq_lengths_q)
    require(q_lengths == list(range(1, count + 1)), 'ACTUAL_QUERY_BOUNDARIES_MISMATCH')
    table_values = _indices(table, 'block_table')
    require(len(table_values) == count, 'PADDED_PAGE_TABLE_UNADAPTED')
    slots = _indices(metadata.slot_mapping, 'slot_mapping')
    require(len(slots) == count and len(set(slots)) == count, 'SLOT_COUNT_OR_COLLISION')
    expected_slots, used_pages = [], []
    for row, position in enumerate(positions):
        require(type(position) is int and position >= 0, 'INVALID_LOGICAL_POSITION')
        needed = (position + block_size) // block_size
        pages = table_values[row][:needed]
        require(len(pages) == needed and all(type(page) is int and 0 <= page < blocks for page in pages),
                'PAGE_TABLE_OUT_OF_RANGE')
        require(len(set(pages)) == len(pages), 'DUPLICATE_PAGE_WITHIN_SEQUENCE')
        used_pages.append(pages)
        expected_slots.append(pages[position // block_size] * block_size + position % block_size)
    # Prefix sharing is disabled by the surrounding scheduler adapter.
    flattened = [page for pages in used_pages for page in pages]
    require(len(set(flattened)) == len(flattened), 'CROSS_REQUEST_PAGE_ALIAS_UNADAPTED')
    require(slots == expected_slots, 'CURRENT_TOKEN_SLOT_DOES_NOT_MATCH_PAGE_TABLE')
    require(writer is not None and writer['returned'] is True, 'ACTUAL_CACHE_WRITER_NOT_OBSERVED')
    for name, cache in (('key_cache', cache_k), ('value_cache', cache_v)):
        current = _meta(cache)
        require(all(writer[name][field] == current[field]
                    for field in ('storage_identity', 'storage_offset', 'shape', 'stride', 'dtype', 'device')),
                'WRITER_CONSUMER_CACHE_OR_SLOT_MISMATCH')
    require(writer['slots'] == slots, 'WRITER_CONSUMER_CACHE_OR_SLOT_MISMATCH')
    layout = actual.get('input_layout')
    if operator == 'npu_fused_infer_attention_score' and layout == 'TND':
        require(list(query.shape) == [count, impl.num_heads, dim], 'TND_QUERY_GEOMETRY_MISMATCH')
        require(actual.get('block_size') == block_size and actual.get('sparse_mode') == 3,
                'ACTUAL_FIA_PARAMETERS_UNADAPTED')
        require(list(actual['actual_seq_lengths']) == q_lengths
                and list(actual['actual_seq_lengths_kv']) == lengths, 'ACTUAL_API_LENGTHS_MISMATCH')
    return dict(schema='demoncat.paged33.boundary_audit.v1',
                diagnostic_only=True, public_worker_qualified=False, native_launch_verified=False,
                read_only=True, metadata_binding='VERIFIED',
                cache_contents_verified=False, actual_operator='torch_npu.' + operator,
                layout=layout, query=_meta(query), key=key_meta, value=value_meta,
                live_key_cache=_meta(cache_k), live_value_cache=_meta(cache_v),
                page_table=_meta(table), used_pages=used_pages, slots=slots,
                expected_slots=expected_slots, sequence_lengths=lengths,
                scheduler=scheduler, writer=writer,
                status='OBSERVED_UNADAPTED_API', refusal=_native_route(operator, layout),
                qualification_scope='CPU_CONTRACT_ONLY' if allow_cpu_test else 'NATURAL_CALL_METADATA_ONLY')


class Paged33Audit:
    """Owned observers around real registered modules and real Python APIs."""
    def __init__(self, worker, *, max_seqs=1):
        from pathlib import Path
        import torch_npu
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm_ascend.attention import attention_v1
        from .call_binding import ForwardBinding
        self.worker, self.runner = worker, worker.model_runner
        self.model, self.max_seqs = self.runner.model, max_seqs
        require_serial_eager(self.runner, expected_max_seqs=max_seqs)
        source = Path(inspect.getsourcefile(attention_v1.AscendAttentionBackendImpl)).resolve(strict=True)
        require(hashlib.sha256(source.read_bytes()).hexdigest() == ATTENTION_SOURCE_SHA256,
                'ATTENTION_SOURCE_ADAPTER_REQUIRED')
        layer_source = Path(inspect.getsourcefile(Attention)).resolve(strict=True)
        require(hashlib.sha256(layer_source.read_bytes()).hexdigest() == LAYER_SOURCE_SHA256,
                'ATTENTION_LAYER_SOURCE_ADAPTER_REQUIRED')
        self.source = dict(backend=dict(path=str(source), sha256=ATTENTION_SOURCE_SHA256),
                           layer=dict(path=str(layer_source), sha256=LAYER_SOURCE_SHA256))
        self.api_module, self.device_operator = torch_npu, attention_v1.DeviceOperator
        self.candidates = [(path, module) for path, module in self.model.named_modules()
                           if type(module) is Attention
                           and type(getattr(module, 'impl', None)) is attention_v1.AscendAttentionBackendImpl]
        require(bool(self.candidates), 'NO_REVIEWED_ATTENTION_IMPL_IN_LOADED_MODEL')
        self.binding = ForwardBinding(self.model)
        self.thread, self.handles, self.patches = threading.get_ident(), [], []
        self.expected, self.current, self.stack = None, None, []
        self.rows, self.errors, self.writers = [], [], {}
        self.prefill_forwards = self.decode_forwards = 0
        self.installed = False

    def _inside(self):
        return threading.get_ident() == self.thread and self.current is not None

    def install(self):
        global _ACTIVE
        require(_ACTIVE is None and not self.installed, 'PAGED_AUDIT_OWNER_ALREADY_PRESENT')
        _ACTIVE = self
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._root_pre, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._root_post, with_kwargs=True, always_call=True))
            for path, module in self.candidates:
                def pre(m, args, kwargs, path=path):
                    if self._inside():self.stack.append((path, m))
                def post(m, args, kwargs, output):
                    if self._inside():
                        require(self.stack and self.stack[-1][1] is m, 'ATTENTION_OBSERVER_STACK_MISMATCH')
                        self.stack.pop()
                self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
                self.handles.append(module.register_forward_hook(post, with_kwargs=True, always_call=True))
            for name in APIS:
                if not hasattr(self.api_module, name):continue
                original = getattr(self.api_module, name)
                def wrapper(*args, _name=name, _original=original, **kwargs):
                    record = None
                    if self._inside():record = self._consumer(_name, args, kwargs)
                    result = self._invoke_api(_name, _original, args, kwargs, record)
                    if record is not None:record['consumer_returned'] = True
                    return result
                setattr(self.api_module, name, wrapper)
                self.patches.append((self.api_module, name, original, wrapper))
            descriptor = inspect.getattr_static(self.device_operator, 'reshape_and_cache')
            original = getattr(self.device_operator, 'reshape_and_cache')
            def writer(*args, **kwargs):
                result = original(*args, **kwargs)
                if self._inside():self._writer(args, kwargs)
                return result
            installed = staticmethod(writer)
            setattr(self.device_operator, 'reshape_and_cache', installed)
            self.patches.append((self.device_operator, 'reshape_and_cache', descriptor, installed))
            self.installed = True
        except BaseException:
            self.close()
            raise

    def _invoke_api(self, name, original, args, kwargs, record):
        """Read-only default; a separate experimental adapter owns native scope."""
        return original(*args, **kwargs)

    def expect(self, expected):
        require(self.installed and self.expected is None and self.current is None, 'AUDIT_REQUEST_ALREADY_BOUND')
        self.expected = dict(expected)

    def _root_pre(self, module, args, kwargs):
        if threading.get_ident() != self.thread:return
        require(self.expected is not None and self.current is None, 'AUDIT_EXPECTED_REQUEST_REQUIRED')
        if self.runner.with_prefill:
            self.prefill_forwards += 1
            return
        from vllm.forward_context import get_forward_context
        from .batch_context import snapshot_first_decode_rows
        context = get_forward_context()
        actual = self.binding.bind(module, args, kwargs)
        self.current = snapshot_first_decode_rows(self.runner, self.model, actual, context,
                                                  self.expected, expected_max_seqs=self.max_seqs)
        self.decode_forwards += 1
        self.writers = {}

    def _root_post(self, module, args, kwargs, output):
        if self._inside():
            require(not self.stack, 'ATTENTION_OBSERVER_STACK_LEAK')
            self.current = None

    def _writer(self, args, kwargs):
        try:
            require(not args and bool(self.stack), 'CACHE_WRITER_OUTSIDE_BOUND_ATTENTION')
            path, module = self.stack[-1]
            self.writers[id(module)] = dict(returned=True, module_path=path,
                key_cache=_meta(kwargs['key_cache']), value_cache=_meta(kwargs['value_cache']),
                slot_mapping=_meta(kwargs['slot_mapping']), slots=_indices(kwargs['slot_mapping'], 'writer_slots'))
        except BaseException as error:self.errors.append(dict(stage='cache_writer', error=repr(error)))

    def _consumer(self, name, args, kwargs):
        row = dict(actual_operator='torch_npu.' + name, consumer_returned=False,
                   metadata_binding='UNOBSERVABLE', status='OBSERVED_UNADAPTED_API')
        self.rows.append(row)
        try:
            require(bool(self.stack), 'ATTENTION_API_OUTSIDE_BOUND_MODULE')
            path, module = self.stack[-1]
            actual = dict(kwargs)
            require(len(args) <= 3, 'POSITIONAL_OPTIONALS_UNADAPTED')
            for key, value in zip(('query', 'key', 'value'), args):
                require(key not in actual, 'DUPLICATE_API_ARGUMENT')
                actual[key] = value
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            pool = context.attn_metadata
            require(isinstance(pool, dict) and context.no_compile_layers.get(module.layer_name) is module,
                    'ACTUAL_ATTENTION_CONTEXT_OWNER_MISMATCH')
            metadata = pool[module.layer_name]
            live_cache = module.kv_cache[context.virtual_engine]
            require(len(live_cache) == 2, 'LIVE_CACHE_PAIR_REQUIRED')
            _flat_alias(module.impl.key_cache, live_cache[0])
            _flat_alias(module.impl.value_cache, live_cache[1])
            row.update(audit_paged_call(name, actual, impl=module.impl,
                       metadata=metadata, scheduler=self.current,
                       writer=self.writers.get(id(module))))
            row.update(module_path=path, module_object_id=id(module), impl_object_id=id(module.impl),
                       context_layer_name=module.layer_name, virtual_engine=context.virtual_engine)
        except BaseException as error:
            row['refusal'] = repr(error)
        return row

    def report(self):
        return dict(schema='demoncat.paged33.worker_audit.v1', diagnostic_only=True,
            public_worker_qualified=False, injection_enabled=False, core_methods_replaced=False,
            source=self.source, candidate_count=len(self.candidates),
            prefill_forwards=self.prefill_forwards, decode_forwards=self.decode_forwards,
            installed=self.installed, rows=self.rows, errors=self.errors,
            status='OBSERVED_UNADAPTED_API' if self.rows else 'NO_ACTUAL_ATTENTION_OBSERVED',
            limitation='No native adapter or tiling-launch qualification is granted by this audit.')

    def close(self):
        global _ACTIVE
        require(threading.get_ident() == self.thread and self.current is None, 'ACTIVE_OR_FOREIGN_AUDIT_CLOSE')
        # If another owner changed a wrapper, do not overwrite that owner.
        for obj, name, original, installed in reversed(self.patches):
            require(inspect.getattr_static(obj, name) is installed, 'AUDIT_WRAPPER_OWNERSHIP_CHANGED:' + name)
            setattr(obj, name, original)
        self.patches.clear()
        for handle in reversed(self.handles):handle.remove()
        self.handles.clear()
        self.installed = False
        if _ACTIVE is self:_ACTIVE = None
