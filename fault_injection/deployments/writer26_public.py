"""Retain actual FIA API returns before Attention copies into its own output.

Only exact loaded Torch packets and the existing source-pinned msprobe wrappers
are accepted. The original outer callable executes once; no tensor view, copy,
address reconstruction, or device initialization is performed here.
"""
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType

from .writer26_inventory import require, tensor_matches_native


APIS = ('npu_fused_infer_attention_score', 'npu_fused_infer_attention_score_v2')


def _packet(value, name):
    from torch._ops import OpOverloadPacket
    from demoncat.runtime.vllm_ascend.worker_extension import collector_dispatch as collector

    wrappers = []
    for _ in range(8):
        if isinstance(value, OpOverloadPacket):
            require(str(value) == 'npu.' + name, 'FIA_PUBLIC_DISPATCHER_IDENTITY_CHANGED')
            return value, wrappers
        module = sys.modules.get(collector.REGISTRY_MODULE)
        require(isinstance(value, FunctionType) and module is not None,
                'UNREVIEWED_FIA_PUBLIC_API_WRAPPER')
        path = Path(module.__file__).resolve(strict=True)
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == collector.APPROVED_REGISTRY_SHA256,
                'UNREVIEWED_FIA_COLLECTOR_SOURCE')
        factories = (
            (module.ApiRegistry.register_custom_api, ('ApiRegistry', 'register_custom_api'),
             ('wrap_api_func', 'api_function')),
            (module.ApiWrapper.wrap_api_func, ('ApiWrapper', 'wrap_api_func'), ('api_function',)),
        )
        matched = []
        for factory, location, inner in factories:
            require(isinstance(factory, FunctionType) and factory.__globals__ is vars(module)
                    and Path(factory.__code__.co_filename).resolve() == path,
                    'FIA_COLLECTOR_FACTORY_OWNER_CHANGED')
            compiled = compile(raw, factory.__code__.co_filename, 'exec', dont_inherit=True)
            require(factory.__code__ == collector.nested(compiled, location),
                    'FIA_COLLECTOR_FACTORY_CODE_CHANGED')
            if value.__code__ is collector.nested(factory.__code__, inner):
                matched.append('.'.join(location))
        require(len(matched) == 1 and value.__globals__ is vars(module),
                'FIA_COLLECTOR_WRAPPER_IDENTITY_CHANGED')
        closure = inspect.getclosurevars(value).nonlocals
        require(closure.get('api_name') in (name, 'npu.' + name) and 'api_func' in closure,
                'FIA_COLLECTOR_PACKET_OWNER_CHANGED')
        wrappers.append(dict(callable_identity=id(value), factory=matched[0],
                             source_sha256=collector.APPROVED_REGISTRY_SHA256))
        value = closure['api_func']
    raise RuntimeError('FIA_COLLECTOR_WRAPPER_DEPTH_EXCEEDED')


class _ObservedAPI:
    def __init__(self, observer, binding):
        self.observer, self.binding = observer, binding

    def __call__(self, *args, **kwargs):
        return self.observer.call(self.binding, args, kwargs)

    def __getattr__(self, name):
        # Preserve the original overload surface, including .out. We do not
        # claim an observation of overloads called directly through this proxy.
        return getattr(self.binding['original'], name)


class FiaPublicOutputObserver:
    def __init__(self, experiment):
        self.experiment = experiment
        self.owned, self.frames = [], []
        self.calls, self.refusals = [], []

    def install(self):
        import torch
        require(not self.owned and not self.frames, 'FIA_PUBLIC_OBSERVER_ALREADY_OWNED')
        module = sys.modules.get('torch_npu')
        if module is None:
            self.refusals.append(dict(reason='LOADED_TORCH_NPU_PUBLIC_API_REQUIRED'))
            return
        for name in APIS:
            original = getattr(module, name, None)
            if original is None:
                continue
            try:
                packet, wrappers = _packet(original, name)
                namespace_packet, namespace_wrappers = _packet(getattr(torch.ops.npu, name), name)
                require(packet is namespace_packet, 'FIA_PUBLIC_ALIAS_NOT_ACTUAL_TORCH_PACKET')
            except (AttributeError, OSError, ValueError, RuntimeError, TypeError) as error:
                self.refusals.append(dict(api=name, reason=str(error)))
                continue
            binding = dict(namespace=module, name=name, original=original, packet=packet,
                           wrappers=wrappers, namespace_wrappers=namespace_wrappers,
                           installed=False, restore_attempted=False)
            binding['observer'] = _ObservedAPI(self, binding)
            self.owned.append(binding)  # Own before changing the public alias.
            setattr(module, name, binding['observer'])
            require(getattr(module, name) is binding['observer'], 'FIA_PUBLIC_ALIAS_INSTALL_UNCONFIRMED')
            binding['installed'] = True

    def matches_callback(self, row):
        if len(self.frames) != 1:
            return False
        frame = self.frames[-1]
        return (frame['valid'] and self.experiment.current is frame['request']
                and row.get('module_object_identity') == id(frame['module'])
                and row.get('actual_request_id') == frame['request']['request_id'])

    def call(self, binding, args, kwargs):
        experiment = self.experiment
        if experiment.current is None:
            return binding['original'](*args, **kwargs)
        experiment._thread()
        record = dict(api=binding['name'], ordinal=len(self.calls), original_calls=0,
                      request_id=experiment.current['request_id'], errors=[], retained=False)
        self.calls.append(record)
        frame = dict(request=experiment.current, module=experiment.stack[-1], valid=False,
                     callback_start=len(experiment.callbacks), record=record)
        self.frames.append(frame)
        try:
            try:
                require(len(self.frames) == 1 and getattr(binding['namespace'], binding['name']) is binding['observer'],
                        'FIA_PUBLIC_SCOPE_OR_ALIAS_OWNER_CHANGED')
                packet, wrappers = _packet(binding['original'], binding['name'])
                require(packet is binding['packet'] and wrappers == binding['wrappers'],
                        'FIA_PUBLIC_CALLABLE_CHAIN_CHANGED')
                before = experiment.reader.state()
                frame['valid'] = True
            except (AttributeError, OSError, ValueError, RuntimeError, TypeError) as error:
                record['errors'].append(str(error))
            record['original_calls'] = 1
            result = binding['original'](*args, **kwargs)
            if frame['valid']:
                try:
                    _, events = experiment.reader.delta(before)
                    self._returned(binding, frame, events, result)
                except (AttributeError, KeyError, ValueError, RuntimeError, TypeError) as error:
                    record['errors'].append(str(error))
                    for row in experiment.callbacks[frame['callback_start']:]:
                        row.setdefault('public_output_refusals', []).append(str(error))
            return result
        finally:
            require(self.frames and self.frames[-1] is frame, 'FIA_PUBLIC_OBSERVER_STACK_CHANGED')
            self.frames.pop()

    def _returned(self, binding, frame, events, result):
        from .writer26_experiment import tensors
        experiment = self.experiment
        require(experiment.current is frame['request'] and experiment.stack[-1] is frame['module'],
                'FIA_PUBLIC_RETURN_REQUEST_OR_MODULE_CHANGED')
        rows = experiment.callbacks[frame['callback_start']:]
        starts = [event for event in events if event['kind_name'] == 'fia_before']
        registrations = [event for event in events if event['kind_name'] == 'fia_gws']
        executions = [event for event in events if event['kind_name'] == 'fia_enter']
        ends = [event for event in events if event['kind_name'] == 'fia_exit']
        require(len(rows) == len(starts) == len(registrations) == len(executions) == len(ends) == 1,
                'ONE_NATIVE_FIA_CALL_PER_PUBLIC_RETURN_REQUIRED')
        row, start, gws, enter, end = rows[0], starts[0], registrations[0], executions[0], ends[0]
        call = row['native_packet']
        require(row.get('module_object_identity') == id(frame['module']) and row.get('independent_event') == start
                and row.get('actual_request_id') == frame['request']['request_id']
                and all(event['call'] == call['independent_call']
                        and event['thread'] == call['thread'] == experiment.native_thread
                        and event['status'] == 0 for event in (start, gws, enter, end))
                and start['sequence'] < gws['sequence'] < enter['sequence'] < end['sequence']
                and gws['executor'] == enter['executor'] == end['executor'] != 0
                and enter['registration'] != 0
                and all(start[key] == call[key] for key in
                        ('output', 'output_bytes', 'residual', 'residual_bytes', 'device', 'dtype', 'format', 'dimensions'))
                and 0 < call['dimensions'] <= len(call['shape'])
                and start['shape'][:call['dimensions']] == call['shape'][:call['dimensions']]
                and all(gws[key] == enter[key] for key in
                        ('output', 'output_bytes', 'device', 'dtype', 'format', 'dimensions', 'shape', 'strides')),
                'PUBLIC_FIA_RETURN_NOT_BOUND_TO_COMPLETED_NATIVE_CALL')
        # The real trace's exit contains token/thread/executor/status only.
        # GWS/enter describe the effective destination, which Native may replace
        # during a qualified lease; the literal public return binds to before.
        dtype = {0: 'torch.float32', 1: 'torch.float16', 27: 'torch.bfloat16'}.get(call['dtype'])
        found = {id(tensor): tensor for tensor in tensors(result)
                 if tensor_matches_native(tensor, start) and str(tensor.dtype) == dtype}
        require(len(found) == 1, 'PUBLIC_FIA_RETURN_FULL_NATIVE_FORMAL_REQUIRED')
        tensor = next(iter(found.values()))
        public_format = binding['namespace'].get_npu_format(tensor)
        # Preserve the actual public/native discrepancy before refusing. A
        # format failure is not a same-format observation or proof that two
        # differently labelled layouts can safely share a native descriptor.
        frame['record']['format_observation'] = dict(
            public_format=int(public_format), native_format=int(call['format']),
            shape=list(tensor.shape), stride=list(tensor.stride()),
            dtype=str(tensor.dtype), storage_offset=tensor.storage_offset(),
            contiguous=tensor.is_contiguous(), full_native_formal_matched=True,
            exact_format_match=public_format == call['format'])
        require(public_format == call['format'], 'PUBLIC_FIA_RETURN_NATIVE_FORMAT_CHANGED')
        experiment.output_refs[call['token']] = tensor
        row['writer_output_retained'] = dict(evidence='ACTUAL_PUBLIC_FIA_RETURN',
            tensor_object_identity=id(tensor), storage_identity=int(tensor.untyped_storage()._cdata),
            storage_address=tensor.untyped_storage().data_ptr(), bytes=tensor.untyped_storage().nbytes(),
            shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype), format=call['format'],
            api=binding['name'], api_callable_identity=id(binding['original']),
            dispatcher_packet_identity=id(binding['packet']), collector_wrappers=binding['wrappers'],
            independent_call=call['independent_call'], entry_sequence=start['sequence'], exit_sequence=end['sequence'])
        frame['record'].update(retained=True, independent_call=call['independent_call'],
                               tensor_object_identity=id(tensor))

    def close(self):
        self.experiment._thread()
        require(not self.frames, 'FIA_PUBLIC_CALL_STILL_INFLIGHT')
        for binding in self.owned:
            actual = getattr(binding['namespace'], binding['name'])
            require(actual is binding['observer'] or (actual is binding['original']
                    and (binding['restore_attempted'] or not binding['installed'])),
                    'FIA_PUBLIC_ALIAS_OWNER_CHANGED')
        while self.owned:
            binding = self.owned[-1]
            binding['restore_attempted'] = True
            try:
                if getattr(binding['namespace'], binding['name']) is binding['observer']:
                    setattr(binding['namespace'], binding['name'], binding['original'])
                require(getattr(binding['namespace'], binding['name']) is binding['original'],
                        'FIA_PUBLIC_ALIAS_RESTORE_UNCONFIRMED')
            finally:
                # A setter can raise after restoration. Confirm the real alias
                # before releasing its handle, while propagating the failure.
                if getattr(binding['namespace'], binding['name']) is binding['original']:
                    self.owned.pop()

    def report(self):
        return dict(owned_api_wrappers=len(self.owned), inflight_public_calls=len(self.frames),
                    calls=self.calls, refusals=self.refusals, source_scope='ACTUAL_FIA_PUBLIC_RETURN_ONLY',
                    tensor_views_created=0, native_qualified=False)
