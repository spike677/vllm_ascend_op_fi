"""On-demand discovery of real Attention calls on one held Worker.

This owns model/module hooks and current public API wrappers only. It never
calls forward, replaces a runner method, repacks caches or substitutes an API.
Every observed call, including unsupported layouts, stays in the denominator.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import sys
import threading

from .attention28_prefill import bind_actual_api, inspect_prefill_input
from .call_binding import ForwardBinding, same_callable
from .paged33_audit import Paged33Audit, _flat_alias, _meta, audit_paged_call
from .paged33_tnd import validate_tnd_call
from .providers.tensor_modules import _capture, _identity
from .serving_context import require, snapshot_fresh_prefill, discover_logits_boundary

APIS = ('npu_fused_infer_attention_score', 'npu_fused_infer_attention_score_v2',
        'npu_fusion_attention', 'npu_fusion_attention_v3', 'npu_incre_flash_attention', '_npu_paged_attention')
_ACTIVE = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def complete(value):
    return _identity(_capture(value))


def finite(value):
    if type(value) is dict:
        return value.get('finite', True) is True and all(finite(x) for x in value.values())
    if type(value) in (list, tuple):
        return all(finite(x) for x in value)
    return True


def inspect_fa_call(operator, actual, metadata, scheduler, *, simulation=False):
    """FA's API is not FIA's API, even when their QKV dimensions agree."""
    require(operator == 'npu_fusion_attention_v3', 'FA_ACTUAL_V3_API_REQUIRED')
    allowed = {'query', 'key', 'value', 'head_num', 'input_layout', 'pse', 'padding_mask', 'atten_mask',
        'prefix', 'actual_seq_qlen', 'actual_seq_kvlen', 'sink', 'scale', 'keep_prob', 'pre_tockens',
        'next_tockens', 'inner_precise', 'sparse_mode', 'gen_mask_parallel', 'sync', 'softmax_layout'}
    require(not set(actual)-allowed, 'FA_UNKNOWN_ACTUAL_ARGUMENT')
    require(all(actual.get(name) is None for name in
        ('pse', 'padding_mask', 'prefix', 'actual_seq_qlen', 'actual_seq_kvlen', 'sink')),
        'FA_OPTIONAL_FEATURE_UNADAPTED')
    require(actual.get('head_num') == 16 and actual.get('keep_prob', 1.0) == 1.0
        and actual.get('gen_mask_parallel', True) is True and actual.get('sync', False) is False
        and actual.get('softmax_layout', '') == '', 'FA_ACTUAL_SEMANTICS_UNADAPTED')
    # This is validation metadata only; the original FA args are never changed.
    check = {key: actual[key] for key in ('query', 'key', 'value', 'atten_mask', 'input_layout', 'scale')}
    check.update(num_heads=actual['head_num'], num_key_value_heads=8, actual_seq_lengths=[2048],
        actual_seq_lengths_kv=[2048], pre_tokens=actual.get('pre_tockens', 2147483647),
        next_tokens=actual.get('next_tockens', 0), sparse_mode=actual.get('sparse_mode', 0),
        inner_precise=actual.get('inner_precise', 0))
    row = inspect_prefill_input('npu_fused_infer_attention_score', check,
        metadata=metadata, scheduler=scheduler, allow_cpu_test=simulation)
    row.update(actual_operator='torch_npu.'+operator, validation_only_projection=True)
    return row


class AttentionExecutionAdapter:
    def __init__(self, deployment, identity_guard):
        self.deployment = deployment
        self.worker, self.runner, self.model = deployment.worker, deployment.runner, deployment.model
        self.driver, self.guard = deployment.native, identity_guard
        self.thread = threading.get_ident()
        self.handles, self.patches, self.rows, self.failures, self.seen = [], [], [], [], set()
        self.failed_requests = []
        self.expected = self.current = None
        self.root_active = self.callbacks = 0
        self.stack, self.parents, self.writers = [], [], {}
        self.live_api_calls = []
        self.installed, self.target, self.needs_off = False, None, False
        self.last_error = None
        self.binding = ForwardBinding(self.model)
        require(same_callable(self.binding.original, type(self.model).forward.__get__(self.model)),
                'ATTENTION_ROOT_FORWARD_SOURCE_CHANGED')
        if deployment.simulation:
            setup = deployment.test_context
            self.api_module, self.candidates = setup['api_module'], setup['candidates']
            self.device_operator, self.processor = setup.get('device_operator'), setup['processor']
            self.source, self.boundary_guard = {'simulation': True}, None
        else:
            # Constructor verifies actual vLLM/Ascend source/classes and eager
            # scheduling. Reuse its actual-object discovery, never its hooks.
            inspected = Paged33Audit(self.worker, max_seqs=1)
            self.api_module, self.device_operator = inspected.api_module, inspected.device_operator
            self.candidates, self.source = inspected.candidates, inspected.source
            from vllm.model_executor.layers.logits_processor import LogitsProcessor
            boundary = discover_logits_boundary(self.model, actual_logits_processor_type=LogitsProcessor,
                actual_vocab_size=self.runner.vllm_config.model_config.hf_config.vocab_size)
            self.processor, self.boundary_guard = boundary['processor'], boundary.get('branch_guard')
        all_modules = list(self.model.named_modules(remove_duplicate=False))
        self.aliases = {id(module): sorted(path for path, value in all_modules if value is module)
                        for _, module in self.candidates}
        self.bound_modules = {}
        for _, module in self.candidates:
            original = ForwardBinding(module)
            require(same_callable(original.original, type(module).forward.__get__(module)),
                    'ATTENTION_MODULE_FORWARD_SOURCE_CHANGED')
            self.bound_modules[id(module)] = (module, type(module), getattr(module, 'impl', None), original)
        self.parent_modules = {}
        for paths in self.aliases.values():
            for path in paths:
                parent = path.rpartition('.')[0]
                if parent:
                    self.parent_modules[parent] = next(value for name, value in all_modules if name == parent)

    def _fail(self, error):
        if self.last_error is None:self.last_error = repr(error)
        self.failures.append(dict(error=repr(error), first_error=self.last_error,
                                  request_id=self.expected[0] if self.expected else None))

    def _owned(self):
        require(threading.get_ident() == self.thread, 'ATTENTION_OWNER_THREAD_CHANGED')
        self.guard()
        require(self.runner.model is self.model and self.worker.model_runner is self.runner,
                'ATTENTION_WORKER_MODEL_CHANGED')
        for handle, callback in self.handles:
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is callback, 'ATTENTION_HOOK_OWNER_CHANGED')
        for obj, name, _, installed in self.patches:
            require(inspect.getattr_static(obj, name) is installed, 'ATTENTION_API_OWNER_CHANGED')

    def _hook(self, module, callback, *, before=False):
        def guarded(*args):
            self.callbacks += 1
            try:
                self._owned()
                return callback(*args)
            except BaseException as error:
                self._fail(error)
                raise
            finally:
                self.callbacks -= 1
        register = module.register_forward_pre_hook if before else module.register_forward_hook
        handle = register(guarded, with_kwargs=True, **({} if before else {'always_call': True}))
        self.handles.append((handle, guarded))
        return handle, guarded

    def install(self):
        global _ACTIVE
        self._owned()
        require(_ACTIVE is None and not self.installed, 'ATTENTION_OWNER_ALREADY_PRESENT')
        _ACTIVE = self
        self.installed = True  # A partial installation is still an owned resource.
        self.root_pre_handle, _ = self._hook(self.model, self._pre, before=True)
        self.root_handle, _ = self._hook(self.model, self._post)
        self.logits_handle, _ = self._hook(self.processor, self._logits)
        seen = set()
        for path, module in self.parent_modules.items():
            # Multiple aliases for a shared parent cannot prove its logical
            # path; the child callback below keeps this ambiguity explicit.
            if id(module) in seen:continue
            seen.add(id(module))
            parent_paths = [name for name, item in self.parent_modules.items() if item is module]
            logical_path = path if len(parent_paths) == 1 else None
            self._hook(module, lambda m, a, k, path=logical_path: self.parents.append((path, m)) if self.current else None,
                       before=True)
            def parent_post(m, a, k, out):
                if self.current:
                    require(self.parents and self.parents[-1][1] is m, 'ATTENTION_PARENT_STACK_CHANGED')
                    self.parents.pop()
            self._hook(module, parent_post)
        seen.clear()
        for _, module in self.candidates:
            if id(module) in seen:continue
            seen.add(id(module))
            self._hook(module, self._module_pre, before=True)
            self._hook(module, self._module_post)
        for name in APIS:
            if not hasattr(self.api_module, name):continue
            original = getattr(self.api_module, name)
            def wrapper(*args, _name=name, _original=original, **kwargs):
                if threading.get_ident() != self.thread or self.current is None:
                    return _original(*args, **kwargs)
                return self._invoke(_name, _original, args, kwargs)
            self.patches.append((self.api_module, name, original, wrapper))
            setattr(self.api_module, name, wrapper)
        if self.device_operator is not None:
            original_descriptor = inspect.getattr_static(self.device_operator, 'reshape_and_cache')
            original = self.device_operator.reshape_and_cache
            def writer(*args, **kwargs):
                result = original(*args, **kwargs)
                if threading.get_ident() == self.thread and self.current is not None:
                    require(self.stack and not args, 'ATTENTION_CACHE_WRITER_BOUNDARY_UNADAPTED')
                    path, module = self.stack[-1]
                    from .paged33_audit import _indices
                    self.writers[id(module)] = dict(returned=True, module_path=path,
                        key_cache=_meta(kwargs['key_cache']), value_cache=_meta(kwargs['value_cache']),
                        slot_mapping=_meta(kwargs['slot_mapping']), slots=_indices(kwargs['slot_mapping'], 'writer_slots'))
                return result
            installed = staticmethod(writer)
            self.patches.append((self.device_operator, 'reshape_and_cache', original_descriptor, installed))
            setattr(self.device_operator, 'reshape_and_cache', installed)

    def expect(self, request_id, phase):
        self._owned()
        require(self.installed and self.expected is None and self.current is None and not self.root_active
            and request_id not in self.seen and type(request_id) is str and bool(request_id),
            'ATTENTION_FRESH_ENQUEUED_REQUEST_REQUIRED')
        self.expected = (request_id, phase)
        self.request_rows = []

    def _context(self):
        if self.deployment.simulation:return self.deployment.test_context['context']()
        from vllm.forward_context import get_forward_context
        return get_forward_context()

    def _pre(self, module, args, kwargs):
        require(self.expected is not None and self.current is None and not self.root_active,
                'ATTENTION_UNEXPECTED_OR_REENTRANT_FORWARD')
        require(next(reversed(module._forward_pre_hooks)) == self.root_pre_handle.id,
                'ATTENTION_ACTUAL_ROOT_INPUT_NOT_FINAL')
        request_id, phase = self.expected
        self.root_active = 1
        self.current = dict(request_id=request_id, phase=phase, calls=[], entry_error=sys.exc_info()[1],
                            input=complete((args, kwargs)), root_returned=False)
        actual, context = self.binding.bind(module, args, kwargs), self._context()
        if self.boundary_guard is not None:self.boundary_guard()
        if self.deployment.simulation:
            scheduler = self.deployment.test_context['snapshot'](request_id, actual, context)
        elif self.runner.with_prefill:
            scheduler = snapshot_fresh_prefill(self.runner, self.model, actual, context,
                expected_prompt_token_ids=self.deployment.prompt, expected_internal_request_id=request_id,
                seen_internal_request_ids=list(self.seen))
        else:
            from .batch_context import snapshot_first_decode_rows
            scheduler = snapshot_first_decode_rows(self.runner, self.model, actual, context,
                {request_id: dict(prompt=list(self.deployment.prompt), logical_id=request_id)}, expected_max_seqs=1)
        self.current.update(scheduler=scheduler, decode=not self.runner.with_prefill)
        self.writers = {}

    def _post(self, module, args, kwargs, output):
        try:
            require(self.current is not None and output is not None and not self.stack and not self.parents
                and sys.exc_info()[1] is self.current['entry_error']
                and next(reversed(module._forward_hooks)) == self.root_handle.id,
                'ATTENTION_ACTUAL_ROOT_RETURN_UNCONFIRMED')
            self.current.update(hidden=complete(output), root_returned=True)
        finally:
            self.root_active = 0

    def _logits(self, module, args, kwargs, output):
        require(self.current is not None and self.current['root_returned'] and output is not None
            and sys.exc_info()[1] is self.current['entry_error']
            and next(reversed(module._forward_hooks)) == self.logits_handle.id,
            'ATTENTION_ACTUAL_LOGITS_RETURN_UNCONFIRMED')
        self.current['logits'] = complete(output)
        self.current.pop('entry_error')
        require(finite(self.current['hidden']) and finite(self.current['logits'])
            and all(finite(row.get('output')) for row in self.current['calls']), 'ATTENTION_NONFINITE_REQUEST')
        self.request_rows.append(self.current)
        self.current = None

    def _module_pre(self, module, args, kwargs):
        if self.current is None:return
        bound, cls, impl, original = self.bound_modules[id(module)]
        require(module is bound and type(module) is cls and getattr(module, 'impl', None) is impl,
                'ATTENTION_IMPLEMENTATION_OBJECT_CHANGED')
        original.bind(module, args, kwargs)
        aliases = self.aliases[id(module)]
        require(all(self.model.get_submodule(path) is module for path in aliases), 'ATTENTION_MODULE_ALIAS_GRAPH_CHANGED')
        choices = [path for path in aliases if not self.parents or path.rpartition('.')[0] == self.parents[-1][0]]
        require(len(choices) == 1, 'ATTENTION_SHARED_ALIAS_CALLSITE_UNPROVEN')
        self.stack.append((choices[0], module))

    def _module_post(self, module, args, kwargs, output):
        if self.current is None:return
        require(self.stack and self.stack[-1][1] is module, 'ATTENTION_MODULE_STACK_CHANGED')
        self.stack.pop()

    def _boundary(self, name, actual, path, module):
        context = self._context()
        require(context.model_instance is self.model and context.no_compile_layers.get(module.layer_name) is module,
                'ATTENTION_CONTEXT_OBJECT_CHANGED')
        metadata = context.attn_metadata[module.layer_name]
        scheduler = self.current['scheduler']
        if self.deployment.mechanism == '33.paged_decode':
            require(self.current['decode'], 'TND33_FIRST_DECODE_REQUIRED')
            cache = module.kv_cache[context.virtual_engine]
            require(len(cache) == 2, 'TND33_LIVE_CACHE_PAIR_REQUIRED')
            _flat_alias(module.impl.key_cache, cache[0]);_flat_alias(module.impl.value_cache, cache[1])
            row = audit_paged_call(name, actual, impl=module.impl, metadata=metadata,
                scheduler=scheduler, writer=self.writers.get(id(module)), allow_cpu_test=self.deployment.simulation)
            row.update(module_path=path, module_object_id=id(module), virtual_engine=context.virtual_engine)
            validate_tnd_call(actual, row, allow_cpu_test=self.deployment.simulation)
            row['input_contract_matches'] = True
        elif self.deployment.mechanism == '28.fa.legal_tiling':
            row = inspect_fa_call(name, actual, metadata, scheduler, simulation=self.deployment.simulation)
        else:
            row = inspect_prefill_input(name, actual, metadata=metadata, scheduler=scheduler,
                                        allow_cpu_test=self.deployment.simulation)
        if self.deployment.mechanism != '33.paged_decode':
            import torch
            mask = actual['atten_mask'].detach().cpu()
            require(torch.equal(mask, torch.ones_like(mask).triu(1)), 'ATTENTION_ACTUAL_CAUSAL_MASK_VALUES_DIFFER')
            row['mask_values_verified'] = True
        return row

    def _selected(self,row):
        return self.target is not None and row['target_key']==self.target

    def _selected_phase(self,phase):
        return phase not in ('observe','observe_repeat','health')

    def _execute_selected(self,actual,row,phase,once):
        return self.driver.execute(actual,row,phase,once)

    def _invoke(self, name, original, args, kwargs):
        self.callbacks += 1
        row = dict(actual_operator='torch_npu.'+name, original_calls=0, original_returned=False,
                   input_contract_matches=False, native=None, reasons=[], logical_role='attention_reader',
                   ordinal=len(self.current['calls']),
                   actual_callable=dict(object_id=id(original), type=type(original).__module__+'.'+type(original).__qualname__),
                   native_original_call_count='UNOBSERVED_UNLESS_INDEPENDENT_LAUNCH_PRESENT',native_census=None)
        self.current['calls'].append(row)
        live_call = (row, original, args, kwargs)
        self.live_api_calls.append(live_call)
        try:
            self._owned()
            require(len(self.current['calls']) <= self.deployment.max_calls and self.stack,
                    'ATTENTION_CALL_OVERFLOW_OR_MISSING_PARENT')
            path, module = self.stack[-1]
            occurrence = sum(x.get('module_path') == path and x['actual_operator'] == row['actual_operator']
                             for x in self.current['calls'][:-1])
            row.update(module_path=path, module_object_id=id(module), aliases=self.aliases[id(module)],
                parent_path=self.parents[-1][0] if self.parents else '', occurrence=occurrence,
                request_id=self.current['request_id'], scheduler=deepcopy(self.current['scheduler']))
            def once():
                require(row['original_calls'] == 0, 'ATTENTION_ORIGINAL_ALREADY_CALLED')
                row['original_calls'] += 1
                return original(*args, **kwargs)
            try:
                actual = bind_actual_api(args, kwargs)
            except Exception as error:
                # An unknown public signature remains a real observed call.
                # Do not guess its semantic arguments or acquire Native scope.
                row.update(input=complete((args, kwargs)), target_key=None)
                row['reasons'].append(str(error))
                if getattr(self.driver,'observer_only',False):
                    row['native_census']=dict(status='UNOBSERVED_OR_INCOMPLETE',reasons=[str(error)],
                        native_fault_qualified=False,fault_authorized=False)
                output = once()
                row.update(output=complete(output), inputs_unchanged=complete((args, kwargs)) == row['input'],
                           original_returned=True)
                require(row['inputs_unchanged'] and finite(row['output']), 'ATTENTION_UNADAPTED_CALL_CHANGED_INPUT')
                return output
            row['input'] = complete(actual)
            row['target_key'] = digest(dict(path=path, operator=row['actual_operator'], occurrence=occurrence,
                input=row['input'], decode=self.current['decode']))
            try:
                row.update(self._boundary(name, actual, path, module))
            except Exception as error:
                row['reasons'].append(str(error))
            selected = self._selected(row)
            phase = self.current['phase']
            if selected:
                require(row['input_contract_matches'], 'ATTENTION_SELECTED_ACTUAL_LAYOUT_CHANGED')
            if getattr(self.driver,'observer_only',False) and phase in ('observe','observe_repeat'):
                context=self._context()
                require(context.model_instance is self.model and context.no_compile_layers.get(module.layer_name) is module,
                        'ATTENTION_CENSUS_CONTEXT_OBJECT_CHANGED')
                require(len(self.deployment.leases) == 1, 'ATTENTION_CENSUS_ACTUAL_OWNER_TICKET_REQUIRED')
                owner,generation=next(iter(self.deployment.leases))
                row['native_census_scope']=dict(owner=owner,generation=generation,worker_object_id=id(self.worker),
                    runner_object_id=id(self.runner),model_object_id=id(self.model),rank=self.worker.rank,
                    device_type=getattr(self.worker.device,'type',None),logical_device=getattr(self.worker.device,'index',None),
                    phase=phase)
                output,receipt=self.driver.observe(original,args,actual,row,context.attn_metadata[module.layer_name],once)
                row['native_census']=receipt
            elif selected and self._selected_phase(phase):
                if phase == 'fault':self.needs_off = True
                output, receipt = self._execute_selected(actual,row,phase,once)
                row['native'] = receipt
            else:
                output = once()
            row['output'] = complete(output)
            row['inputs_unchanged'] = complete(actual) == row['input']
            require(row['original_calls'] == 1 and row['inputs_unchanged'] and finite(row['output']),
                    'ATTENTION_ORIGINAL_OR_INPUT_LIFETIME_CHANGED')
            row['original_returned'] = True
            return output
        except BaseException as error:
            row['reasons'].append(repr(error));self._fail(error)
            raise
        finally:
            self.callbacks -= 1
            if self.live_api_calls and self.live_api_calls[-1] is live_call:
                self.live_api_calls.pop()
            else:
                # An altered live argument owner cannot be silently forgotten
                # by cleanup. Retain the ledger; preserve an earlier failure.
                self._fail(RuntimeError('ATTENTION_LIVE_API_CALL_OWNER_CHANGED'))
                self.live_api_calls.append(live_call)

    def collect(self):
        self._owned()
        require(self.expected is not None and self.current is None and not self.root_active and not self.callbacks
            and len(self.request_rows) == (2 if self.deployment.mechanism == '33.paged_decode' else 1),
            'ATTENTION_ACTUAL_REQUEST_NOT_COMPLETE')
        require(all(row['root_returned'] and finite(row['hidden']) and finite(row['logits'])
            for row in self.request_rows), 'ATTENTION_INCOMPLETE_REQUEST')
        request_id, phase = self.expected
        row = dict(request_id=request_id, phase=phase, forwards=deepcopy(self.request_rows))
        self.rows.append(row);self.seen.add(request_id);self.expected = None
        return row

    def close(self):
        global _ACTIVE
        require(threading.get_ident() == self.thread and not self.root_active and not self.callbacks
                and not self.live_api_calls,
                'ATTENTION_INFLIGHT_OR_FOREIGN_CLOSE')
        self.guard()
        require(not self.needs_off, 'ATTENTION_ACTUAL_OFF_REQUEST_REQUIRED')
        self.driver.stop()
        for entry in self.patches[:][::-1]:
            obj, name, original, installed = entry
            require(inspect.getattr_static(obj, name) is installed, 'ATTENTION_API_OWNER_CHANGED')
            try:
                setattr(obj, name, original)
            finally:
                if inspect.getattr_static(obj, name) is original:self.patches.remove(entry)
        for entry in self.handles[:][::-1]:
            handle, callback = entry
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and (handle.id not in hooks or hooks[handle.id] is callback),
                    'ATTENTION_HOOK_OWNER_CHANGED')
            try:
                handle.remove()
            finally:
                if handle.id not in hooks:self.handles.remove(entry)
        if self.expected is not None or self.current is not None:
            partial = {key: value for key, value in (self.current or {}).items() if key != 'entry_error'}
            self.failed_requests.append(dict(expected=self.expected, partial=deepcopy(partial),
                returned_forwards=deepcopy(getattr(self, 'request_rows', [])), complete=False,
                first_error=self.last_error))
        self.current = self.expected = None
        self.stack.clear();self.parents.clear();self.writers.clear();self.installed = False
        if _ACTIVE is self:_ACTIVE = None

    def state(self):
        self._owned()
        return dict(owned_hooks=len(self.handles), owned_api_wrappers=len(self.patches),
            inflight=int(bool(self.root_active or self.callbacks or self.live_api_calls)), needs_off=self.needs_off,
            pending_request=self.expected, completed_requests=len(self.rows), failures=deepcopy(self.failures),
            failed_requests=deepcopy(self.failed_requests))
