"""Source-bound natural Fault41 writer/live-cache/consumer adapter.

Only observed first pure decode in a normal TP1 eager Worker is admitted. This
provider owns no request scheduler and never creates or repacks a cache. The
Fault33 audit is used only for its read-only live page/slot consistency checks.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import secrets
import threading

from fault_injection.deployments.postwrite import PostwriteRefused, NativePostwriteLibrary, _cpu, _equal, _storage
from fault_injection.deployments.postwrite_kv import live_value_tail_plan, LiveKVSlotLease
from .paged33_audit import ATTENTION_SOURCE_SHA256, LAYER_SOURCE_SHA256, APIS, _meta, _indices, _flat_alias
from .serving_context import require, require_serial_eager


_ACTIVE = None


def _output_hash(output):
    from torch.utils._pytree import tree_flatten
    import torch
    leaves, spec = tree_flatten(output)
    digest = hashlib.sha256(str(spec).encode())
    for value in leaves:
        require(isinstance(value, torch.Tensor), 'ACTUAL_MODEL_TENSOR_OUTPUT_REQUIRED')
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _target_key(path, module, context):
    value = module.impl.value_cache
    return (path, id(module), id(module.impl), context.virtual_engine, _storage(value),
            value.storage_offset(), tuple(value.shape), tuple(value.stride()))


class Postwrite41WorkerProvider:
    group_context = None
    def __init__(self, worker, transport, *, tail_elements=2, group_context=None):
        import torch_npu
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm_ascend.attention import attention_v1
        from .call_binding import ForwardBinding
        require(isinstance(transport, NativePostwriteLibrary), 'TAIL_COLUMN_VERIFIED_NATIVE_TRANSPORT_REQUIRED')
        self.worker, self.runner = worker, worker.model_runner
        self.model, self.transport = self.runner.model, transport
        self.group_context=group_context
        self._require_runner()
        self.source = {}
        for label, cls, expected in (('backend', attention_v1.AscendAttentionBackendImpl, ATTENTION_SOURCE_SHA256),
                                     ('layer', Attention, LAYER_SOURCE_SHA256)):
            source = Path(inspect.getsourcefile(cls)).resolve(strict=True)
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            require(actual == expected, 'TAIL_COLUMN_SOURCE_ADAPTER_REQUIRED:' + label)
            self.source[label] = dict(path=str(source), sha256=actual)
        self.structural_candidates = [(path, module) for path, module in self.model.named_modules()
                                      if isinstance(module, Attention)]
        self.candidates = [(path, module) for path, module in self.structural_candidates
                          if type(module) is Attention
                          and type(getattr(module, 'impl', None)) is attention_v1.AscendAttentionBackendImpl]
        require(bool(self.candidates), 'NO_REVIEWED_ATTENTION_IMPL_IN_LOADED_MODEL')
        self.binding = ForwardBinding(self.model)
        self.api_module, self.device_operator = torch_npu, attention_v1.DeviceOperator
        self.owner, self.generation = secrets.randbits(63) + 1, 0
        self.thread, self.tail_elements = threading.get_ident(), tail_elements
        self.handles, self.patches, self.stack = [], [], []
        self.hook_owners = {}
        self.install_epoch = 0
        self.current = self.expected = self.lease = self.collector = None
        self.phase, self.target, self.qualified, self.lossless = None, None, False, False
        self.fault_attempted, self.installed = False, False
        self.rows, self.errors, self.history, self.writers = [], [], [], {}
        self.outputs, self.forward_serial, self.pending_run = [], 0, None
        self.independent_calls = []
        self.collectors, self.cleanup_receipts = [], []
        from fault_injection.deployments.postwrite_kv_inventory import KVWorkloadInventory
        self.workload = KVWorkloadInventory()
        self.coverage = None
        self.process_start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        self.model_identity = self._identity()

    def _require_runner(self):
        if self.group_context is None:return require_serial_eager(self.runner,expected_max_seqs=1)
        from .multi_rank_prerequisites import tp2_eager_assessment
        c=self.group_context
        report=tp2_eager_assessment(self.runner,c.tp_group,c.identity['rank'])
        require(not report['unmet'],'TP2_KV41_PREREQUISITES:'+str(report['unmet']))

    def snapshot_decode(self,actual,context,expected):
        if self.group_context is None:
            from .batch_context import snapshot_first_decode_rows
            return snapshot_first_decode_rows(self.runner,self.model,actual,context,expected,expected_max_seqs=1)
        self._require_runner()
        from .batch_context import _snapshot_first_decode_rows
        return _snapshot_first_decode_rows(self.runner,self.model,actual,context,expected,expected_max_seqs=1)

    def fault_injector(self):
        if self.group_context is None:return True
        require(self.group_context.decision is not None,'TP2_KV41_ROLE_REQUIRED')
        return self.group_context.decision['selected_rank']==self.group_context.identity['rank']

    def _identity(self):
        return (id(self.model), tuple((id(module), type(module).__module__, type(module).__qualname__)
                                    for module in self.model.modules()),
                tuple((id(parameter), _storage(parameter), parameter._version)
                      for parameter in self.model.parameters()))

    def _inside(self):
        return threading.get_ident() == self.thread and self.current is not None

    def wrap_callback(self, kind, callback):
        wrapper = getattr(self, '_core_callback_wrapper', None)
        return wrapper(kind, callback) if wrapper is not None else callback

    def _hook(self, module, kind, callback, *, before=False):
        callback = self.wrap_callback(kind, callback)
        register = module.register_forward_pre_hook if before else module.register_forward_hook
        handle = register(callback, with_kwargs=True, **({} if before else {'always_call': True}))
        self.handles.append(handle)
        self.hook_owners[handle.id] = callback
        return handle

    def install(self):
        global _ACTIVE
        from .postwrite_collector import owned_api_patch
        require(_ACTIVE is None and not self.installed, 'TAIL_COLUMN_OWNER_ALREADY_PRESENT')
        _ACTIVE = self
        self.install_epoch = getattr(self, 'install_epoch', 0) + 1
        self.hook_owners = getattr(self, 'hook_owners', {})
        try:
            self._hook(self.model, 'root_pre', self._root_pre, before=True)
            self._hook(self.model, 'root_post', self._root_post)
            for path, module in self.candidates:
                def pre(m, args, kwargs, path=path):
                    if self._inside():
                        self.stack.append((path, m))
                        self.workload.module_entered(m)
                def post(m, args, kwargs, output):
                    if self._inside():
                        require(self.stack and self.stack[-1][1] is m, 'TAIL_COLUMN_ATTENTION_STACK_MISMATCH')
                        self.stack.pop()
                self._hook(module, 'attention_pre', pre, before=True)
                self._hook(module, 'attention_post', post)
            for name in APIS:
                if not hasattr(self.api_module, name):continue
                original = getattr(self.api_module, name)
                def wrapper(*args, _name=name, _original=original, **kwargs):
                    if self._inside():return self._consume(_name, _original, args, kwargs)
                    return _original(*args, **kwargs)
                wrapper = self.wrap_callback('consumer_api', wrapper)
                patch=owned_api_patch(self.api_module,name,original,wrapper)
                setattr(self.api_module, name, wrapper)
                self.patches.append(patch)
            descriptor = inspect.getattr_static(self.device_operator, 'reshape_and_cache')
            original = getattr(self.device_operator, 'reshape_and_cache')
            def writer(*args, **kwargs):
                result = original(*args, **kwargs)
                if self._inside():self._writer(args, kwargs)
                return result
            writer = self.wrap_callback('writer_api', writer)
            installed = staticmethod(writer)
            patch=owned_api_patch(self.device_operator,'reshape_and_cache',descriptor,installed)
            setattr(self.device_operator, 'reshape_and_cache', installed)
            self.patches.append(patch)
            self.installed = True
        except BaseException:
            self.close()
            raise

    def expect(self, phase, request_id, prompt, *, collector=None):
        require(self.installed and self.expected is None and self.current is None and self.lease is None,
                'TAIL_COLUMN_REQUEST_OR_RECOVERY_ACTIVE')
        require(self._identity() == self.model_identity, 'TAIL_COLUMN_MODEL_BINDING_CHANGED')
        ordinary = phase in ('observe', 'lossless', 'baseline', 'fault', 'off')
        grouped = (self.coverage is not None and phase in ('group_observe', 'group_lossless', 'group_fault', 'group_off'))
        require(ordinary or grouped, 'TAIL_COLUMN_INVALID_PHASE')
        if grouped:self.coverage.authorize_phase(phase)
        if phase != 'observe':require(self.qualified, 'TAIL_COLUMN_HEALTHY_QUALIFICATION_REQUIRED')
        if phase == 'fault':
            require(self.lossless and not self.fault_attempted, 'TAIL_COLUMN_LOSSLESS_ONCE_GATE')
            self.fault_attempted = True
        self.phase = phase
        self.expected = {request_id: dict(logical_id=0, prompt=list(prompt))}
        self.collector = collector
        if collector is not None and all(item is not collector for item in self.collectors):
            self.collectors.append(collector)
        self.rows, self.errors, self.pending_run = [], [], None
        self.generation += 1

    def _root_pre(self, module, args, kwargs):
        if threading.get_ident() != self.thread:return
        require(self.expected is not None and self.current is None, 'TAIL_COLUMN_EXPECTED_REQUEST_REQUIRED')
        if self.runner.with_prefill:return
        from vllm.forward_context import get_forward_context
        from .batch_context import snapshot_first_decode_rows
        context = get_forward_context()
        self.current = self.snapshot_decode(self.binding.bind(module, args, kwargs),context,self.expected)
        require(self.pending_run is None, 'TAIL_COLUMN_ONLY_FIRST_DECODE_PER_REQUEST')
        self.forward_serial += 1
        self.writers = {}
        self.workload.begin(self.forward_serial, self.current)
        self.pending_run = dict(phase=self.phase, forward_id=str(self.forward_serial),
            request_ids=list(self.current['request_ids']), actual_scheduler=self.current,
            model_instance=id(self.model), pid=os.getpid(), process_start=self.process_start,
            rows=self.rows, errors=self.errors, new_fault_hits=0)
        if self.collector is not None:
            try:self.collector.start(model=self.model, rank_id=0)
            except BaseException:
                self.collector.stop()
                raise

    def _root_post(self, module, args, kwargs, output):
        if not self._inside():return
        try:
            if self.collector is not None:
                import torch
                try:torch.npu.synchronize()
                finally:self.collector.stop()
            require(not self.stack, 'TAIL_COLUMN_ATTENTION_STACK_LEAK')
            self.recover()
            require(output is not None, 'TAIL_COLUMN_FORWARD_DID_NOT_RETURN')
            copied = _cpu(output)
            self.outputs.append(copied)
            self.pending_run['model_output_sha256'] = _output_hash(copied)
            self.pending_run['model_output_snapshot'] = 'FULL_ROOT_OUTPUT_AFTER_CAPTURE_STOP'
            self.pending_run['native_live_leases'] = int(self.lease is not None)
            self.pending_run['independent_inventory'] = self.workload.finish(self.pending_run['model_output_sha256'])
        finally:
            self.workload.abort('ACTUAL_MODEL_FORWARD_DID_NOT_COMPLETE')
            self.current = None

    def _writer(self, args, kwargs):
        require(not args and bool(self.stack), 'TAIL_COLUMN_WRITER_OUTSIDE_BOUND_ATTENTION')
        path, module = self.stack[-1]
        self.writers[id(module)] = dict(returned=True, module_path=path,
            key_cache=_meta(kwargs['key_cache']), value_cache=_meta(kwargs['value_cache']),
            slot_mapping=_meta(kwargs['slot_mapping']), slots=_indices(kwargs['slot_mapping'], 'writer_slots'))
        self.workload.writer_returned(module, self.writers[id(module)])

    def _consume(self, name, original, args, kwargs):
        require(bool(self.stack), 'TAIL_COLUMN_CONSUMER_OUTSIDE_BOUND_ATTENTION')
        path, module = self.stack[-1]
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        actual = dict(kwargs)
        require(len(args) <= 3, 'TAIL_COLUMN_POSITIONAL_OPTIONALS_UNADAPTED')
        for key, value in zip(('query', 'key', 'value'), args):
            require(key not in actual, 'TAIL_COLUMN_DUPLICATE_API_ARGUMENT')
            actual[key] = value
        require(isinstance(context.attn_metadata, dict)
                and context.no_compile_layers.get(module.layer_name) is module,
                'TAIL_COLUMN_ACTUAL_CONTEXT_OWNER_MISMATCH')
        live_cache = module.kv_cache[context.virtual_engine]
        require(len(live_cache) == 2, 'TAIL_COLUMN_LIVE_CACHE_PAIR_REQUIRED')
        _flat_alias(module.impl.key_cache, live_cache[0])
        _flat_alias(module.impl.value_cache, live_cache[1])
        row = dict(module_path=path, module_object_id=id(module), operator=name,
                   consumer_returned=False, target_key=_target_key(path, module, context))
        self.rows.append(row)
        try:
            plan = live_value_tail_plan(operator=name, actual=actual, impl=module.impl,
                metadata=context.attn_metadata[module.layer_name], scheduler=self.current,
                writer=self.writers.get(id(module)), tail_elements=self.tail_elements)
            row.update(eligible=True, boundary=plan.proof)
        except BaseException as error:
            row.update(eligible=False, refusal=repr(error))
            self.errors.append(row)
            # Observation still executes the genuine call. An armed run fails
            # closed instead of shrinking the healthy eligible denominator.
            if self.phase != 'observe':raise
            trace = self.workload.consumer_entered(module=module, path=path, operator=name,
                target_key=row['target_key'], eligible=False, rejection=repr(error))
            result = self.workload.call_original(trace, original, args, kwargs)
            row['consumer_returned'] = True
            return result
        trace = self.workload.consumer_entered(module=module, path=path, operator=name,
            target_key=row['target_key'], eligible=True)
        original_api = original
        def original(*actual_args, **actual_kwargs):
            return self.workload.call_original(trace, original_api, actual_args, actual_kwargs)
        if self.coverage is not None and self.phase.startswith('group_'):
            result = self.coverage.consume(trace, row, plan, actual, original, args, kwargs)
            row['consumer_returned'] = True
            return result
        # Separate append-only workload inventory is made BEFORE admission,
        # including lossless, unselected and OFF calls; injector hits never set it.
        call = dict(call_id=f'{self.forward_serial}:{len(self.rows)}',
                    forward_id=str(self.forward_serial), request_ids=list(self.current['request_ids']),
                    target_key=row['target_key'], eligible=True, phase=self.phase)
        self.independent_calls.append(call)
        row['call_id'] = call['call_id']
        selected = row['target_key'] == self.target
        if selected and self.phase=='fault' and not self.fault_injector():
            from fault_injection.deployments.postwrite_kv import valid_kv_snapshot
            before=valid_kv_snapshot(self.transport,plan,actual)
            result=original(*args,**kwargs)
            after=valid_kv_snapshot(self.transport,plan,actual)
            require(before==after,'TP2_KV41_GUARD_CACHE_CHANGED')
            row['guard_boundary']=dict(before=before,after=after,unchanged=True,native_writes=0)
            row['consumer_returned']=True
            return result
        if selected and self.phase in ('lossless', 'baseline', 'fault'):
            require(self.lease is None, 'TAIL_COLUMN_PRIOR_LEASE_NOT_DRAINED')
            self.lease = LiveKVSlotLease(self.transport, plan, self.owner, self.generation)
            try:
                self.lease.arm(fault=self.phase == 'fault')
                result = self.lease.consume(original, args, kwargs)
                row['intervention'] = self.lease.report
                self.pending_run['new_fault_hits'] += int(self.phase == 'fault')
                row['consumer_returned'] = True
                return result
            finally:
                if self.lease is not None and not self.lease.live:self.lease = None
        result = original(*args, **kwargs)
        row['consumer_returned'] = True
        return result

    def finish_request(self):
        require(self.current is None and self.expected is not None and self.pending_run is not None,
                'TAIL_COLUMN_REQUEST_NOT_COMPLETED')
        run = self.pending_run
        require(not self.errors and self.lease is None and 'model_output_sha256' in run,
                'TAIL_COLUMN_UNPROVEN_CALL_OR_RECOVERY')
        keys = [row['target_key'] for row in self.rows if row.get('eligible')]
        require(bool(keys) and len(set(keys)) == len(keys), 'TAIL_COLUMN_NO_DISTINCT_ELIGIBLE_TARGETS')
        require({row['module_object_id'] for row in self.rows} == {id(module) for _, module in self.candidates},
                'TAIL_COLUMN_STRUCTURAL_DENOMINATOR_INCOMPLETE')
        require(all(row['consumer_returned'] for row in self.rows), 'TAIL_COLUMN_INCOMPLETE_ACTUAL_CONSUMER')
        if self.phase.startswith('group_'):
            self.coverage.request_finished(run, self.outputs[-1])
        elif self.phase == 'observe':
            healthy = [item for item in self.history if item['phase'] == 'observe']
            require(len(healthy) < 2, 'TAIL_COLUMN_OBSERVATION_ALREADY_QUALIFIED')
            if healthy:
                require(keys == [row['target_key'] for row in healthy[0]['rows']]
                        and _equal(self.outputs[0], self.outputs[-1]), 'TAIL_COLUMN_HEALTHY_REPLAY_NOT_EXACT')
                self.target, self.qualified = keys[0], True
        else:
            require(keys == [row['target_key'] for row in self.history[0]['rows']],
                    'TAIL_COLUMN_ACTUAL_TARGET_SET_CHANGED')
            if self.phase in ('lossless', 'baseline', 'off'):
                require(_equal(self.outputs[0], self.outputs[-1]), 'TAIL_COLUMN_LOSSLESS_OR_OFF_OUTPUT_MISMATCH')
                if self.phase in ('lossless', 'baseline'):self.lossless = True
            if self.phase == 'fault':require(run['new_fault_hits'] == int(self.fault_injector()), 'TAIL_COLUMN_C1_HIT_COUNT_MISMATCH')
        run.update(recovery_confirmed=True, exact_healthy_output=_equal(self.outputs[0], self.outputs[-1]),
                   public_worker_qualified=False, all_eligible_qualified=False, every_match_qualified=False)
        self.history.append(run)
        self.expected = self.phase = self.pending_run = self.collector = None
        return run

    def recover(self):
        if self.lease is not None:
            self.lease.recover()
            if not self.lease.live:self.lease = None

    def close(self):
        global _ACTIVE
        require(threading.get_ident() == self.thread and self.current is None, 'TAIL_COLUMN_ACTIVE_OR_FOREIGN_CLOSE')
        self.recover()
        from .postwrite_collector import restore_owned_api_patches
        receipt = restore_owned_api_patches(self.patches, collectors=self.collectors)
        self.cleanup_receipts.append(receipt)
        self.patches.clear()
        for handle in reversed(self.handles[:]):
            hooks = handle.hooks_dict_ref()
            expected = getattr(self, 'hook_owners', {}).get(handle.id)
            require(hooks is not None and (handle.id not in hooks or expected is None or hooks[handle.id] is expected),
                    'TAIL_COLUMN_HOOK_OWNER_CHANGED')
            try:
                handle.remove()
            finally:
                if handle.id not in hooks:
                    self.handles.remove(handle)
                    getattr(self, 'hook_owners', {}).pop(handle.id, None)
            require(handle.id not in hooks, 'TAIL_COLUMN_HOOK_REMOVAL_UNCONFIRMED')
        self.installed = False
        if _ACTIVE is self:_ACTIVE = None

    def report(self):
        return dict(schema='demoncat.postwrite41.natural_worker.v1', mechanism='41.tail_column',
            boundary='actual_writer_live_value_slot_tnd_fia', controlled_legal_address_proxy=True,
            native_out_of_bounds_claim=False, source=self.source, owner=self.owner, generation=self.generation,
            candidate_count=len(self.candidates), qualified=self.qualified, lossless=self.lossless,
            target=self.target, native_live_leases=int(self.lease is not None), installed=self.installed,
            owned_hooks=len(self.handles), owned_api_patches=len(self.patches), cleanup_receipts=self.cleanup_receipts,
            history=self.history, pending=self.pending_run, independent_calls=self.independent_calls,
            independent_kv_inventory=self.workload.frames,
            public_worker_qualified=False, all_eligible_qualified=False, every_match_qualified=False,
            scope='natural first pure decode, BF16, TP1/PP1/DP1, serial eager, one actual request')
