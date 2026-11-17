"""Fault22: legal source-row output proxy at a live RoPE table lookup.

Rebuilt from the published mechanism contract, not the missing historical
extension. This cannot reproduce or diagnose uint32 address overflow.
"""
from ._scoped_dispatch import ScopedDispatch
from .core import UnsupportedCall
from .tensor_observer import tensors


class WrongRowSession(ScopedDispatch):
    mechanism = '22.wrong_row'

    def __init__(self, model, *, selected_calls=None, admission=None, **kwargs):
        super().__init__(model, **kwargs)
        self.records = []
        self.selected_calls = None if selected_calls is None else frozenset(selected_calls)
        self.admission = admission

    def is_selected(self, event):
        if self.admission is not None:
            return bool(self.admission(event))
        return ((event['path'], event['ordinal']) in self.selected_calls
                if self.selected_calls is not None else super().is_selected(event))

    def dispatch(self, func, args, kwargs):
        import torch
        schema = getattr(func, '_schema', None)
        name = getattr(schema, 'name', '')
        is_lookup = ((name == 'aten::index_select' and not getattr(schema, 'overload_name', ''))
                     or (name == 'aten::index' and getattr(schema, 'overload_name', '') == 'Tensor'))
        if not is_lookup:
            # Track aliases produced by split/view without guessing equality
            # from a data_ptr that may include a different storage offset.
            inputs = list(tensors((args, kwargs)))
            readers = []
            linked = []
            for record in self.records:
                if record['forward'] != len(self.forwards)-1: continue
                source_ids = {x.untyped_storage()._cdata for x in record['lineage']}
                if any(x.untyped_storage()._cdata in source_ids for x in inputs):
                    linked.append(record)
            if name in ('aten::mul','aten::add','npu::npu_apply_rotary_pos_emb',
                        'npu::npu_rotary_mul', '_C_ascend::rotary_embedding'):
                readers = linked
            output = func(*args, **kwargs)
            for record in readers:
                record['event']['consumers'].append(dict(operator=str(func),
                    returned=True, same_lookup_storage=any(x.untyped_storage()._cdata ==
                        record['result'].untyped_storage()._cdata for x in inputs),
                    actual_lookup_lineage=True, module_path=self.stack[-1][0]))
            if name in ('aten::view', 'aten::reshape', 'aten::_unsafe_view', 'aten::slice',
                        'aten::select', 'aten::split', 'aten::split_with_sizes', 'aten::expand',
                        'aten::unsqueeze', 'aten::squeeze', 'aten::clone', 'aten::_to_copy',
                        'aten::repeat', 'aten::cat', 'aten::transpose', 'aten::permute'):
                for record in linked:
                    record['lineage'].extend(tensors(output))
                    record['event']['lineage_operators'].append(str(func))
            return output
        if name == 'aten::index_select':
            values = dict(zip(('input','dim','index'), args)); values.update(kwargs)
            source, dim, index = (values.get(k) for k in ('input','dim','index'))
        else:
            values = dict(zip(('self', 'indices'), args)); values.update(kwargs)
            source, indices = values.get('self'), values.get('indices')
            if not isinstance(indices, (tuple, list)) or len(indices) != 1:
                return func(*args, **kwargs)
            index, dim = indices[0], 0
        owners = [(path, module) for path, module in self.stack
                  if getattr(module, 'cos_sin_cache', None) is source]
        if len(owners) != 1:
            return func(*args, **kwargs)
        event = self.event(str(func))
        event['cache_owner_path'] = owners[0][0]
        with self.diagnostics():
            if (not isinstance(source, torch.Tensor) or not isinstance(index, torch.Tensor)
                or dim != 0 or source.ndim != 2 or index.ndim != 1 or index.numel() < 2
                or not source.is_floating_point() or index.dtype not in (torch.int32,torch.int64)
                or not source.is_contiguous() or source.device != index.device
                or not self.simulation and source.device.type != 'npu'):
                event.update(status='unsupported', reason='LIVE_CONTIGUOUS_ROPE_TABLE_AND_MULTI_TOKEN_INDICES_REQUIRED')
                return func(*args, **kwargs)
            before_source, before_index = self.cpu(source), self.cpu(index)
            if (bool((before_index < 0).any()) or bool((before_index >= source.shape[0]).any())
                    or int(before_index[0]) == int(before_index[-1])):
                event.update(status='unsupported', reason='DISTINCT_LEGAL_FIRST_LAST_SOURCE_ROWS_REQUIRED')
                return func(*args, **kwargs)
        # The original lookup executes exactly once, before the one-row proxy.
        result = func(*args, **kwargs)
        with self.diagnostics():
            native = self.cpu(result)
            expected = before_source.index_select(0, before_index.long())
            if not torch.equal(native, expected) or result.untyped_storage()._cdata == source.untyped_storage()._cdata:
                raise UnsupportedCall('Original lookup or independent result storage was not verified')
            injecting = self.arm == 'fault' and self.is_selected(event)
            if injecting:
                replacement = source[int(before_index[0])].clone()
                self.temps.append(replacement)
                result[-1].copy_(replacement)
                self.synchronize()
            after = self.cpu(result)
            desired = native.clone()
            if injecting: desired[-1] = before_source[int(before_index[0])]
            if (not torch.equal(after, desired) or not torch.equal(self.cpu(source), before_source)
                    or not torch.equal(self.cpu(index), before_index)):
                raise UnsupportedCall('Wrong-row proxy modified unowned data or failed readback')
            event.update(status='applied' if injecting else 'baseline', applied=injecting,
                original_execution_verified=True, native_calls=1, source_shape=list(source.shape),
                output_shape=list(result.shape), dtype=str(result.dtype), target_row=index.numel()-1,
                correct_source_row=int(before_index[-1]), replacement_source_row=int(before_index[0]),
                source_unchanged=True, index_unchanged=True, non_target_rows_unchanged=True,
                replacement_readback_exact=True, output_changed=not torch.equal(native,after),
                cache_object_id=id(source), lookup_storage_id=int(result.untyped_storage()._cdata),
                consumers=[], lineage_operators=[], boundary=str(func)+'.return',
                semantics='WRONG_ROW_OUTPUT_PROXY_NOT_UINT32_OVERFLOW')
            self.records.append(dict(result=result, source=source, event=event,
                                     lineage=[result], forward=len(self.forwards)-1))
        return result

    def close(self):
        super().close()
        self.records.clear()
