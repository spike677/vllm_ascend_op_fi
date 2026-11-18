"""Bounded experimental Worker evidence around the real TND native adapter.

Retain actual return/cache references during inference. All full tensor reads,
hashes and comparisons occur after the collector stops. No runner replacement,
synthetic cache packing or output modification is performed.
An explicitly separate no-statistics boundary diagnostic additionally reads
the first decode API's inputs and immediate returns in both stock/private arms.
"""
import hashlib
import inspect
import json
from pathlib import Path

from .paged33_audit import Paged33Audit
from .paged33_tnd import TNDNativeAdapter, Paged33TNDProvider
from .serving_context import discover_logits_boundary, require
from .call_binding import ForwardBinding


def class_forward_binding(module):
    """Bind the registered class method through retained collector wrappers.

    msprobe stop disables collection but retains wrapped_forward. A later OFF
    session must not mistake that variadic wrapper for the original signature.
    ForwardBinding.bind still verifies every actual wrapper and its owner.
    """
    binding = ForwardBinding(module)
    binding.original = type(module).forward.__get__(module, type(module))
    binding.signature = inspect.signature(binding.original)
    return binding


def tensor_file(value, path):
    import torch
    require(isinstance(value, torch.Tensor) and value.layout == torch.strided
            and 0 < value.numel()*value.element_size() <= 128*1024**2, 'TND33_COMPLETE_TENSOR_BOUND')
    cpu = value.detach().contiguous().cpu()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return dict(path=str(path), shape=list(cpu.shape), dtype=str(cpu.dtype), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), finite=bool(torch.isfinite(cpu).all()))


def logical_cache_file(cache, used_pages, length, path):
    """Read the valid natural page order; this never feeds data to an operator."""
    import torch
    require(cache.ndim == 4 and len(used_pages) == 1 and type(length) is int and length > 0,
            'TND33_SINGLE_REQUEST_CACHE_READBACK_REQUIRED')
    page_size = int(cache.shape[1])
    pages = used_pages[0]
    require(len(pages) == (length+page_size-1)//page_size and len(set(pages)) == len(pages)
            and all(type(page) is int and 0 <= page < cache.shape[0] for page in pages),
            'TND33_CACHE_READBACK_PAGE_MAP_INVALID')
    require(length*cache.shape[2]*cache.shape[3]*cache.element_size() <= 64*1024**2,
            'TND33_CACHE_READBACK_BOUND')
    digest, size, finite, left = hashlib.sha256(), 0, True, length
    with Path(path).open('xb') as stream:
        for page in pages:
            valid = min(left, page_size)
            cpu = cache[page, :valid].detach().contiguous().cpu()
            raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(raw);stream.write(raw);size += len(raw)
            finite = finite and bool(torch.isfinite(cpu).all())
            left -= valid
    require(left == 0, 'TND33_CACHE_READBACK_INCOMPLETE')
    return dict(path=str(path), shape=[length, int(cache.shape[2]), int(cache.shape[3])],
                dtype=str(cache.dtype), bytes=size, sha256=digest.hexdigest(), finite=finite,
                physical_pages=list(pages), readback='VALID_LOGICAL_PAGE_ORDER_AFTER_CONSUMER',
                operator_input_repacked=False)


def payload_identity(report):
    """Exact represented bytes, with dtype/shape; no statistical tolerance."""
    keys = ('shape', 'dtype', 'bytes', 'sha256')
    def ident(row):
        return {key: row[key] for key in keys}
    return dict(hidden=ident(report['hidden']), logits_hidden=ident(report['logits_hidden']),
                logits=ident(report['logits']),
                attention={name: {kind: ident(value) for kind, value in row.items()}
                           for name, row in report['attention'].items()})


class TNDWorkerProbeSession(Paged33TNDProvider):
    def __init__(self, worker, *, native_args, evidence_dir, statistics=False, boundary_args=None):
        require(boundary_args is None or (not statistics and (native_args is None or native_args['arm'] in ('baseline', 'off'))),
                'TND33_BOUNDARY_DIAGNOSTIC_REQUIRES_NO_FAULT_NO_STATISTICS')
        import torch
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.folder = Path(evidence_dir)
        require(str(self.folder).startswith('/evidence/') and not self.folder.exists(),
                'TND33_FRESH_PROBE_DIRECTORY_REQUIRED')
        self.folder.mkdir(parents=True)
        self.collector, self.packet_attributes = None, None
        self.capture_active = False
        self.tensor_refs, self.pending_decode, self.root_hidden, self.logits_hidden, self.logits = {}, False, None, None, None
        model = worker.model_runner.model
        if statistics:
            from msprobe.pytorch import PrecisionDebugger
            import msprobe
            from .collector_dispatch import CollectorPacketAttributes
            package = Path(msprobe.__file__).resolve().parent
            config = dict(task='statistics', level='mix', precision='high', rank=[], step=[0],
                dump_path=str(self.folder/'statistics'), async_dump=False,
                statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
            (self.folder/'config.json').write_text(json.dumps(config, indent=2)+'\n')
            files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml')}
            identity = dict(package=str(package), files=files, edited_by_experiment=False,
                            step_meaning={'0': 'one complete natural prefill+first-decode request'},
                            diagnostic_tensor_readback_outside_capture=True)
            (self.folder/'collector_identity.json').write_text(json.dumps(identity, indent=2)+'\n')
            self.collector = PrecisionDebugger(config_path=str(self.folder/'config.json'))
            self.packet_attributes = CollectorPacketAttributes()
        adapter_type = TNDNativeAdapter
        s2_normal = bool(boundary_args is not None and boundary_args.get('arch22_s2_normal') is True)
        require(not s2_normal or (not statistics and (native_args is None or
            native_args.get('arch22_s2_normal') is True)), 'S2_NORMAL_SAME_CAPTURE_SCOPE_REQUIRED')
        if native_args is not None and native_args.get('arch22_s2_normal') is True:
            require(s2_normal and not native_args.get('normal_bridge'), 'S2_NORMAL_DISTINCT_VARIANT_REQUIRED')
            from .paged33_arch22_s2_normal import S2NormalProbeAdapter
            adapter_type = S2NormalProbeAdapter
            native_args = {name:value for name,value in native_args.items() if name!='arch22_s2_normal'}
        if native_args is not None and native_args.get('normal_bridge') is True:
            require(not statistics and boundary_args is not None, 'ARCH22_NORMAL_PROBE_NEEDS_INDEPENDENT_BOUNDARY')
            from .paged33_arch22_bridge import NormalBridgeProbeAdapter
            adapter_type = NormalBridgeProbeAdapter
            native_args = {name: value for name, value in native_args.items() if name != 'normal_bridge'}
        adapter = None if native_args is None else adapter_type(**native_args, evidence_dir=self.folder/'native')
        super().__init__(worker, adapter=adapter)
        self.s2_normal = s2_normal
        self.logical_binding = None
        self.logical_owner = object() if adapter is None else getattr(adapter,'owner',None)
        self.binding = class_forward_binding(self.model)
        boundary = discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor,
            actual_vocab_size=worker.model_runner.vllm_config.model_config.hf_config.vocab_size)
        self.logits_processor, self.head = boundary['processor'], boundary['lm_head']
        self.logits_binding = class_forward_binding(self.logits_processor)
        self.logits_contract = boundary['report']
        self.synced = False
        self.boundary_capture = self.boundary_stock_scope = None
        if boundary_args is not None:
            from . import paged33_boundary_probe
            from .paged33_boundary_probe import FirstDecodeBoundaryCapture, stock_scope_from_receipt
            require(hashlib.sha256(Path(paged33_boundary_probe.__file__).read_bytes()).hexdigest()
                    == boundary_args['capture_source_sha256'], 'TND33_BOUNDARY_INSTALLED_SOURCE_CHANGED')
            self.boundary_capture = FirstDecodeBoundaryCapture(worker, self.folder/'first_boundary', torch.npu.synchronize)
            if self.adapter is None:
                self.boundary_stock_scope = stock_scope_from_receipt(boundary_args['path'], boundary_args['expected_sha256'])

    def install(self):
        # start() installs msprobe API wrappers; install ours *after* that
        # ownership change. Keep the collector callable in the chain once.
        if self.collector is not None:
            self.collector.start(model=self.model, rank_id=self.worker.rank)
            self.capture_active = True
            self.packet_attributes.install()
        super().install()
        self.handles.append(self.logits_processor.register_forward_pre_hook(self._logits_pre, with_kwargs=True))
        self.handles.append(self.logits_processor.register_forward_hook(self._logits_post, with_kwargs=True, always_call=True))

    def expect(self, expected):
        super().expect(expected)

    def _root_pre(self, module, args, kwargs):
        super()._root_pre(module, args, kwargs)
        if self._inside():
            require(not self.pending_decode and self.root_hidden is None, 'TND33_SINGLE_FIRST_DECODE_SCOPE')
            self.pending_decode = True

    def _root_post(self, module, args, kwargs, output):
        if self._inside() and output is not None:
            require(output.ndim == 2 and output.shape[0] == 1, 'TND33_DECODE_HIDDEN_SHAPE')
            self.root_hidden = output  # retain, do not clone/transfer during capture
        super()._root_post(module, args, kwargs, output)

    def _logits_pre(self, module, args, kwargs):
        if not self.pending_decode:
            return
        actual = self.logits_binding.bind(module, args, kwargs)
        require(actual['lm_head'] is self.head, 'TND33_ACTUAL_LOGITS_HEAD_DIFFERS')
        self.logits_hidden = actual['hidden_states']

    def _logits_post(self, module, args, kwargs, output):
        if not self.pending_decode:
            return
        if output is not None:
            require(output.ndim == 2 and output.shape[0] == 1
                    and output.shape[1] == self.worker.model_runner.vllm_config.model_config.hf_config.vocab_size,
                    'TND33_COMPLETE_VOCAB_OUTPUT_REQUIRED')
            self.logits = output
        self.pending_decode = False

    def _invoke_api(self, name, original, args, kwargs, record):
        capture = self.boundary_capture
        first = capture is not None and record is not None and capture.data is None
        invoke = original
        if first:
            require(bool(self.stack), 'TND33_FIRST_BOUNDARY_OUTSIDE_MODULE')
            capture.start(name, original, args, kwargs, record, self.stack[-1][1], len(self.rows)-1)
            def invoke(*actual_args, **actual_kwargs):
                return capture.invoke_original(original, actual_args, actual_kwargs)
        if first and getattr(self,'s2_normal',False):
            from .paged33_logical_binding import LogicalPagedInput
            from vllm.forward_context import get_forward_context
            context=get_forward_context();path,module=self.stack[-1]
            actual=dict(kwargs)
            require(len(args)<=3,'S2_NORMAL_POSITIONAL_OPTIONALS_UNADAPTED')
            for argument_name,value in zip(('query','key','value'),args):
                require(argument_name not in actual,'S2_NORMAL_DUPLICATE_ARGUMENT');actual[argument_name]=value
            self.logical_binding=LogicalPagedInput(worker=self.worker,module=module,module_path=path,
                forward_context=context,metadata=context.attn_metadata[module.layer_name],actual=actual,
                scheduler=self.current,writer=self.writers[id(module)],owner=self.logical_owner,generation=1)
            if self.adapter is not None:self.adapter.bind_input(self.logical_binding)
            else:
                captured_invoke=invoke
                def invoke(*actual_args,**actual_kwargs):
                    return self.logical_binding.execute(lambda:captured_invoke(*actual_args,**actual_kwargs),
                                                        owner=self.logical_owner,generation=1)
        if first and self.boundary_stock_scope is not None:
            result = self.boundary_stock_scope.invoke(lambda: Paged33Audit._invoke_api(self, name, invoke, args, kwargs, record))
        else:
            result = (Paged33Audit._invoke_api(self, name, invoke, args, kwargs, record) if self.adapter is None
                      else super()._invoke_api(name, invoke, args, kwargs, record))
        if first:
            if getattr(self,'s2_normal',False):capture.data['logical_input_binding']=self.logical_binding.report()
            from .paged33_boundary_probe import mapped_api_libraries, read_scoped_launches, MappedLibraryObservationError
            token = self.boundary_stock_scope.token if self.boundary_stock_scope else self.adapter.calls[-1]['call_id']
            runtime = self.boundary_stock_scope.runtime_dir if self.boundary_stock_scope else self.adapter.runtime_dir
            capture.data['native_call_id'] = token
            for key, observe in (('runtime_observation', lambda: read_scoped_launches(runtime, token,
                    expected_pid=capture.pid,expected_native_tid=capture.native_tid)),
                                 ('mapped_api_libraries', mapped_api_libraries)):
                try:
                    capture.data[key] = observe()
                except Exception as error:
                    # Missing coverage is evidence, never a guessed stock ABI.
                    capture.data[key] = dict(status='UNOBSERVED_OR_INCOMPLETE', error=repr(error))
                    if isinstance(error,MappedLibraryObservationError):
                        capture.data[key]['error']=error.evidence['error']
                        capture.data[key]['diagnostic']=error.evidence
            capture.write()
        if record is not None:
            require(record.get('metadata_binding') == 'VERIFIED' and self.stack,
                    'TND33_ACTUAL_BOUNDARY_REQUIRED_FOR_READBACK')
            path, module = self.stack[-1]
            require(path not in self.tensor_refs, 'TND33_DUPLICATE_LAYER_CALL')
            value = result[0] if isinstance(result, (tuple, list)) else result
            self.tensor_refs[path] = dict(output=value, key=module.impl.key_cache, value=module.impl.value_cache,
                                          row=record)
        return result

    def stop_capture(self):
        import torch
        try:
            torch.npu.synchronize()
            self.synced = True
        finally:
            if self.packet_attributes is not None:
                self.packet_attributes.restore()
            if self.collector is not None and self.capture_active:
                self.collector.stop()
                self.collector.step()
                self.capture_active = False

    def collect(self):
        self.stop_capture()
        boundary = self.boundary_capture.finalize() if self.boundary_capture is not None else None
        require(self.synced and self.current is None and not self.pending_decode and not self.errors,
                'TND33_REQUEST_NOT_COMPLETED_CLEANLY')
        expected = {name for name, _ in self.candidates}
        require(self.decode_forwards == 1 and set(self.tensor_refs) == expected
                and all(row.get('consumer_returned') for row in self.rows), 'TND33_INCOMPLETE_NATURAL_LAYER_CENSUS')
        private = self.folder/'private';private.mkdir()
        report = dict(schema='demoncat.paged33.natural_probe.v1',
            hidden=tensor_file(self.root_hidden, private/'hidden.bin'),
            logits_hidden=tensor_file(self.logits_hidden, private/'logits-hidden.bin'),
            logits=tensor_file(self.logits, private/'logits.bin'), attention={},
            audit=self.report(), logits_contract=self.logits_contract,
            native_calls=self.adapter.calls if self.adapter is not None else [],
            statistical_first_difference='NOT_SELECTED_BY_RUNTIME_PROBE', c2=False, c3=False,
            public_worker_qualified=False, complete_tensor_readback_during_statistics=False,
            integer_page_slot_metadata_readback_during_audit=True)
        if boundary is not None:
            report.update(first_decode_boundary=boundary, complete_tensor_readback_at_first_api_boundary=True)
        require(all(report['hidden'][key] == report['logits_hidden'][key]
                    for key in ('shape', 'dtype', 'bytes', 'sha256')), 'TND33_ROOT_TO_LOGITS_HIDDEN_BYTES_DIFFER')
        for index, name in enumerate(sorted(expected)):
            value = self.tensor_refs[name]
            row = value['row']
            report['attention'][name] = dict(
                output=tensor_file(value['output'], private/f'layer-{index}-output.bin'),
                key=logical_cache_file(value['key'], row['used_pages'], row['sequence_lengths'][0], private/f'layer-{index}-key.bin'),
                value=logical_cache_file(value['value'], row['used_pages'], row['sequence_lengths'][0], private/f'layer-{index}-value.bin'))
        report['all_represented_tensors_finite'] = all(item['finite'] for item in
            [report['hidden'], report['logits_hidden'], report['logits'],
             *(item for layer in report['attention'].values() for item in layer.values())])
        (self.folder/'runtime_result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        require(report['all_represented_tensors_finite'], 'TND33_NONFINITE_COMPLETE_WORKER_TENSOR')
        self.tensor_refs.clear();self.root_hidden = self.logits_hidden = self.logits = None
        return report

    def close(self):
        self.stop_capture()
        stock_cleanup = self.boundary_stock_scope.close() if self.boundary_stock_scope is not None else None
        state = (self.adapter.disable() if self.adapter is not None else
                 dict(scope_cleared=True, local_streams_drained=self.synced, native_calls=0))
        Paged33Audit.close(self)
        require(not self.patches and not self.handles, 'TND33_OWNED_HOOK_OR_WRAPPER_LEAK')
        state.update(owned_hooks=0, owned_api_wrappers=0, core_methods_replaced=False)
        if self.boundary_capture is not None:
            self.boundary_capture.release()
            state.update(boundary_refs_released=True, stock_launch_scope=stock_cleanup)
        if getattr(self,'logical_binding',None) is not None:
            self.logical_binding.release(owner=self.logical_owner,generation=1)
            state['logical_binding_released']=self.logical_binding.closed
        return state
