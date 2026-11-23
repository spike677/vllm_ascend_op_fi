"""Read-only dispatch census. No injection, admission, tensor copies, or value dump.

Reports dispatch-visible alias/write/read candidates. Raw-pointer/Triton writes
and implicit kernel format conversions are not observable through this tool.
"""
import collections
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def metadata(t, format_reader):
    shape, stride = list(t.shape), list(t.stride())
    try:
        fmt = format_reader(t)
    except Exception:
        fmt = None
    return dict(shape=shape, stride=stride, offset=t.storage_offset(), dtype=str(t.dtype),
                device=str(t.device), format=fmt, contiguous=t.is_contiguous())


def span(t):
    """Exact byte interval only for contiguous tensors; holes are not inferred."""
    if not t.is_contiguous() or not t.numel():
        return None
    begin=t.storage_offset()*t.element_size()
    return (begin,begin+t.numel()*t.element_size())


def root_base(t):
    seen=set()
    while getattr(t,'_base',None) is not None and id(t) not in seen:
        seen.add(id(t));t=t._base
    return t


class AliasChainCensus(TorchDispatchMode):
    def __init__(self, format_reader=lambda t: None, *, max_samples=128, module=lambda: ''):
        super().__init__()
        self.format_reader, self.max_samples, self.module = format_reader, max_samples, module
        self.views = {}
        self.pending = {}
        self.counts = collections.Counter()
        self.groups = collections.Counter()
        self.view_formats = collections.Counter()
        self.samples = []
        self.seq = 0

    def _storage(self, t):
        return (str(t.device), t.untyped_storage()._cdata)

    def _view(self, t):
        row = self.views.get(id(t))
        if row is not None and row['view']() is t and row['base']() is not None:
            return row
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.seq += 1
        name = str(func)
        schema = func._schema
        alias_result = any(r.alias_info is not None for r in schema.returns)
        writes, reads = [], []
        for i, arg in enumerate(schema.arguments):
            value = args[i] if i < len(args) else kwargs.get(arg.name)
            (writes if arg.alias_info is not None and arg.alias_info.is_write else reads).extend(tensors(value))
        self.counts['dispatch_calls'] += 1
        # Observe value consumers, excluding aliases/metadata and in-place writers.
        if not alias_result and not writes:
            for t in reads:
                key = self._storage(t)
                for p in self.pending.get(key, ()):
                    base = p['base']()
                    if base is None or p['consumed']:
                        continue
                    read_span=span(t)
                    write_span=p['write_span']
                    if read_span is None or write_span is None or max(read_span[0],write_span[0])>=min(read_span[1],write_span[1]):
                        continue
                    p['consumed'] = True
                    self.counts['base_read_after_view_write'] += 1
                    group = (p['view_op'], p['writer'], name, str(p['base_meta']['format']))
                    self.groups[group] += 1
                    if len(self.samples) < self.max_samples:
                        self.samples.append(dict(view_op=p['view_op'], writer=p['writer'], consumer=name,
                            base=p['base_meta'], view=p['view_meta'], writer_module=p['module'],
                            consumer_module=self.module(), write_sequence=p['sequence'], read_sequence=self.seq,
                            base_object_same=base is t, shared_storage=True,
                            consumer_tensor=metadata(t,self.format_reader),
                            contiguous_write_span=list(write_span),contiguous_read_span=list(read_span),
                            proof='HOST_DISPATCH_CANDIDATE_NOT_DEVICE_VALUE_OR_IMPLICIT_COPY_PROOF'))
        write_rows = []
        for t in writes:
            row = self._view(t)
            if row:
                base = row['base']()
                if base is not None and self._storage(base) == self._storage(t):
                    write_rows.append((t, base, row))
        result = func(*args, **kwargs)
        for t, base, row in write_rows:
            self.counts['view_writes'] += 1
            item = dict(base=weakref.ref(base), view_op=row['op'], writer=name,
                        sequence=self.seq, consumed=False, module=self.module(),
                        base_meta=metadata(base, self.format_reader), view_meta=metadata(t, self.format_reader),write_span=span(t))
            key = self._storage(base)
            live = [p for p in self.pending.get(key, ()) if p['base']() is not None and not p['consumed']]
            # Keep every current-base write in counters; bounded examples are explicit.
            if len(live) >= 64:
                self.counts['pending_evidence_dropped'] += 1
                live.pop(0)
            live.append(item)
            self.pending[key] = live
        if alias_result and not writes:
            inputs = list(tensors((args, kwargs)))
            for t in tensors(result):
                for source in inputs:
                    if t is source or self._storage(t) != self._storage(source):
                        continue
                    parent = self._view(source)
                    base = parent['base']() if parent else root_base(source)
                    if base is None:
                        continue
                    self.views[id(t)] = dict(view=weakref.ref(t), base=weakref.ref(base), op=name)
                    self.counts['shared_storage_views'] += 1
                    self.view_formats[str(metadata(base,self.format_reader)['format'])] += 1
                    break
        # Remove dead weak references periodically; never retain model tensors.
        if self.seq % 256 == 0:
            self.views = {k:v for k,v in self.views.items() if v['view']() is not None and v['base']() is not None}
            self.pending = {k:[p for p in rows if p['base']() is not None and not p['consumed']]
                            for k,rows in self.pending.items()}
        return result

    def report(self):
        return dict(schema='dcat.alias.census.v1', counts=dict(self.counts), samples=self.samples, view_base_formats=dict(self.view_formats),
                    groups=[dict(view_op=k[0],writer=k[1],consumer=k[2],base_format=k[3],count=v)
                            for k,v in sorted(self.groups.items())],
                    sample_limit=self.max_samples, sampling_truncated=self.counts['base_read_after_view_write']>len(self.samples),
                    observation_only=True, injection_enabled=False, qualification_granted=False,
                    limitations=['Dispatch-visible calls only; fused/Triton/raw-pointer kernel internals unobserved',
                                 'Format is runtime metadata; an implicit format-copy trigger is not proven',
                                 'Consumer means a later value op reads an overlapping contiguous interval of the same live base storage; no output value proof',
                                 'No model-family or private-format absence qualification is inferred from zero observed candidates'])
