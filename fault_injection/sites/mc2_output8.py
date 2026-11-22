"""Unregistered MC2 output-row proxy; no collective metadata is modified.

Trusted adapters must supply the current call's row/request map. This helper
does not establish the M4 scheduler join or grant a deployment qualification.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from .mla28 import require, fingerprint
from .projection39 import tensors


def shifted_rows(output, request_rows):
    require(isinstance(output, torch.Tensor) and output.ndim == 2
            and output.is_contiguous() and output.shape[0] == len(request_rows),
            'MC2_OUTPUT_ROW_MAPPING_REQUIRED')
    groups = {}
    for index, request in enumerate(request_rows):
        require(request is None or type(request) is str and request,
                'MC2_REQUEST_ID_REQUIRED')
        if request is not None:
            groups.setdefault(request, []).append(index)
    indices = list(range(len(request_rows)))
    for rows in groups.values():
        for i, target in enumerate(rows):
            indices[target] = rows[(i + 1) % len(rows)]
    selected = torch.tensor(indices, dtype=torch.int64, device=output.device)
    return output.index_select(0, selected), indices


class MC2OutputConsumer(TorchDispatchMode):
    """Whole-output read only. Aliases/subregions and in-place writes refuse.

    RegionLeases supplies real same-stream completion before byte restoration.
    A Python return alone is never counted as consumption.
    """
    def __init__(self, leases):
        super().__init__()
        self.leases, self.pending, self.events = leases, None, []

    def watch(self, output, request_rows, *, site, fault):
        require(self.pending is None, 'MC2_OUTPUT_CONSUMER_OVERLAP')
        with _disable_current_modes():
            value, indices = shifted_rows(output, request_rows)
            eligible = indices != list(range(len(indices)))
            e = dict(site=site, operator='npu_moe_distribute_combine_v2',
                     request_rows=list(request_rows), observed=True,
                     eligible=eligible, applied=False, consumed=False,
                     restored=False, changed=0, failed=False,
                     reason=None if eligible else 'NO_MULTI_ROW_REQUEST_GROUP')
            self.events.append(e)
            if not eligible:
                return output
            token = None
            if fault:
                e['before_sha256'] = fingerprint(output)
                token = self.leases.acquire(output)
                e['changed'] = int((output != value).sum().item())
                self.leases.write(token, value)
                e.update(applied=True, after_sha256=fingerprint(output))
            self.pending = (output, token, e)
        return output

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        matched = False
        if self.pending is not None:
            out, token, event = self.pending
            schema = func._schema
            for index, argument in enumerate(schema.arguments):
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                for tensor in tensors(value):
                    if tensor.untyped_storage()._cdata != out.untyped_storage()._cdata:
                        continue
                    require(tensor is out, 'MC2_ALIAS_CONSUMER_UNREVIEWED')
                    require(argument.alias_info is None or not argument.alias_info.is_write,
                            'MC2_CONSUMER_WRITES_PROTECTED_OUTPUT')
                    matched = True
            if matched:
                require(not any(r.alias_info is not None for r in schema.returns),
                        'MC2_VIEW_IS_NOT_VALUE_CONSUMER')
                require(schema.name not in ('aten::empty_like', 'aten::zeros_like',
                        'aten::ones_like', 'aten::full_like', 'aten::new_empty',
                        'aten::new_empty_strided', 'aten::new_zeros', 'aten::new_ones',
                        'aten::new_full', 'aten::is_pinned'),
                        'MC2_METADATA_IS_NOT_CONSUMPTION')
        result = func(*args, **kwargs)
        if matched:
            with _disable_current_modes():
                if token is not None:
                    self.leases.consumed(token)
                    self.leases.restore(token)
                    event.update(restored=True, restored_sha256=fingerprint(out))
                    require(event['before_sha256'] == event['restored_sha256'],
                            'MC2_OUTPUT_RESTORE_MISMATCH')
                event.update(consumed=True, consumer_operator=func._schema.name)
                self.pending = None
        return result

    def close(self):
        require(self.pending is None and self.leases.count() == 0,
                'MC2_UNCONSUMED_OUTPUT_RETAINS_LEASE')


def observe_combine_return(original, args, kwargs, *, mode, context):
    """Preserve the complete original MC2 call, then alter its actual result.

    Context is supplied by a trusted runtime adapter, never by public payload.
    No output is fabricated when this EP rank has no actual combine call.
    """
    require(context.get('actual_scheduler_join') is True
            and context.get('operator') == 'npu_moe_distribute_combine_v2',
            'MC2_ACTUAL_CALL_JOIN_UNPROVEN')
    output = original(*args, **kwargs)
    return mode.watch(output, context['request_rows'], site=context['site'],
                      fault=context['fault'])
