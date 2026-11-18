"""Owner-scoped, real Python forward frames for the public output39 window.

This observes an already executing request. It neither calls a model nor
fabricates Module hooks. Unseen modules remain in the structural denominator.
Only the reviewed Ascend rotary branch has a bounded non-call proof here.
"""
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
import sys
import threading
import types

import torch

from fault_injection.deployments.postwrite_inventory import ProjectionWorkloadInventory
from fault_injection.deployments.postwrite_observed import PostwriteObservedForward, require
from .call_binding import original_before_collector, same_callable


_ROTARY_MODULE = 'vllm_ascend.ops.rotary_embedding'
_ROTARY_SOURCE_SHA256 = '74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7'
_ROTARY_PARENT = 'AscendRotaryEmbedding'
_ROTARY_CHILD = 'AscendApplyRotaryEmb'
_ROTARY_OPERATOR = 'vllm.npu_rotary_embedding.default'


def _codes(code):
    yield code
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            yield from _codes(value)


def _closure(function):
    return tuple(id(cell.cell_contents) for cell in (function.__closure__ or ()))


class _SourceCache:
    def __init__(self):
        self.files = {}

    def bind(self, function):
        code, namespace = function.__code__, function.__globals__
        module = sys.modules.get(namespace.get('__name__'))
        require(module is not None and vars(module) is namespace, 'OUTPUT39_FRAME_GLOBALS_UNPROVEN')
        path = Path(getattr(module, '__file__', '')).resolve(strict=True)
        require(path.is_file() and Path(code.co_filename).resolve() == path, 'OUTPUT39_FRAME_SOURCE_UNPROVEN')
        key = (path, code.co_filename)
        if key not in self.files:
            raw = path.read_bytes()
            compiled = compile(raw, code.co_filename, 'exec', dont_inherit=True,
                               optimize=sys.flags.optimize)
            self.files[key] = (hashlib.sha256(raw).hexdigest(), tuple(_codes(compiled)))
        sha, compiled_codes = self.files[key]
        require(any(code == item for item in compiled_codes), 'OUTPUT39_FRAME_CODE_SOURCE_CHANGED')
        return dict(module=module.__name__, path=str(path), sha256=sha,
            function=function.__qualname__, first_line=code.co_firstlineno,
            function_identity=id(function), code_identity=id(code), globals_identity=id(namespace))

    def verify(self):
        for (path, _), (sha, _) in self.files.items():
            require(hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                    'OUTPUT39_FRAME_SOURCE_CHANGED')


class _FrameBinding:
    def __init__(self, module, path, attribute, original, sources):
        actual = getattr(module, attribute)
        function = getattr(original, '__func__', original)
        require(isinstance(function, types.FunctionType) and getattr(original, '__self__', None) is module,
                'OUTPUT39_FRAME_BOUND_METHOD_REQUIRED')
        require(not function.__code__.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR),
                'OUTPUT39_SYNCHRONOUS_FRAME_REQUIRED')
        self.source = sources.bind(function)
        self.module, self.path, self.attribute = module, path, attribute
        self.original, self.function, self.code, self.globals = original, function, function.__code__, function.__globals__
        self.actual = actual
        outer = getattr(actual, '__func__', actual)
        self.outer, self.outer_code, self.outer_cells = outer, outer.__code__, _closure(outer)
        self.cells = _closure(function)

    def verify(self, frame=None):
        actual = getattr(self.module, self.attribute)
        require(same_callable(actual, self.actual) and self.outer.__code__ is self.outer_code
            and _closure(self.outer) == self.outer_cells and self.function.__code__ is self.code
            and self.function.__globals__ is self.globals and _closure(self.function) == self.cells,
            'OUTPUT39_FRAME_CURRENT_BINDING_CHANGED')
        require(vars(sys.modules[self.globals['__name__']]) is self.globals,
                'OUTPUT39_FRAME_GLOBALS_CHANGED')
        if frame is not None:
            require(frame.f_code is self.code and frame.f_globals is self.globals
                and frame.f_locals.get('self') is self.module, 'OUTPUT39_FRAME_ACTUAL_OWNER_CHANGED')

    def report(self):
        return dict(path=self.path, module_object_identity=id(self.module), attribute=self.attribute,
                    source=deepcopy(self.source))


class _DiscoveryInventory(ProjectionWorkloadInventory):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def original_returned(self, func, module_path, module, result):
        self.window.verify_live()
        self.window.operator_returned(func, result)
        return super().original_returned(func, module_path, module, result)

    def report(self):
        result = super().report()
        result['actual_forward_frames'] = deepcopy(self.window.events)
        result['frame_binding_gaps'] = deepcopy(self.window.binding_gaps)
        result['frame_failures'] = list(self.window.errors)
        result['bounded_noncall_proofs'] = deepcopy(self.window.noncall_proofs)
        result['frame_observer_scope'] = dict(self.window.scope)
        return result


def verify_noncall_binding(model, proof):
    """Read-only boundary recheck of a prior real, workload-bound observation."""
    aliases = tuple(model.named_modules(remove_duplicate=False))
    parents = {obj for _, obj in aliases if id(obj) == proof['parent_identity']}
    children = {obj for _, obj in aliases if id(obj) == proof['child_identity']}
    require(len(parents) == len(children) == 1, 'OUTPUT39_ROTARY_HELD_OBJECT_CHANGED')
    parent, child = next(iter(parents)), next(iter(children))
    source_module = sys.modules.get(_ROTARY_MODULE)
    require(source_module is not None and type(parent) is getattr(source_module, _ROTARY_PARENT, None)
        and type(child) is getattr(source_module, _ROTARY_CHILD, None)
        and getattr(parent, 'apply_rotary_emb', None) is child, 'OUTPUT39_ROTARY_BRANCH_BINDING_CHANGED')
    path = next(name for name, obj in aliases if obj is parent)
    source = _SourceCache()
    binding = _FrameBinding(parent, path, '_forward_method', parent._forward_method, source)
    function = getattr(getattr(torch.ops.vllm, 'npu_rotary_embedding', None), 'default', None)
    require(binding.report() == proof['delegate_binding'] and binding.source['sha256'] == _ROTARY_SOURCE_SHA256
        and binding.function is getattr(type(parent), 'forward_oot', None) and binding.globals.get('torch') is torch
        and isinstance(function, torch._ops.OpOverload) and id(function) == proof['operator_identity']
        and str(function) == proof['operator'] and str(function._schema) == proof['operator_schema']
        and [name for name, obj in aliases if obj is parent] == proof['parent_aliases']
        and [name for name, obj in aliases if obj is child] == proof['child_aliases'],
        'OUTPUT39_ROTARY_BRANCH_BINDING_CHANGED')


class Output39ObservedForward(PostwriteObservedForward):
    """One actual root/logits request; no global or future-thread profiler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer.inventory = _DiscoveryInventory(self)
        self.sources = _SourceCache()
        self.bindings, self.by_code, self.binding_gaps = [], {}, []
        self.by_module = {}
        self.entries, self.frames, self.events, self.errors = [], {}, [], []
        self.noncall_proofs, self.rotary = [], []
        self.profile_callback = self._profile
        self.profile_owned = False
        self.frame_serial = 0
        self.scope = dict(owner=self.provider.owner, native_generation=self.provider.generation,
            pid=os.getpid(), thread=self.provider._thread, model_identity=id(self.provider.model),
            phase=self.phase, profile='CURRENT_THREAD_ACTUAL_BOUND_FORWARD',
            serializable_authority=False, qualification_granted=False)

    def _prepare_bindings(self):
        aliases = tuple(self.provider.model.named_modules(remove_duplicate=False))
        for path, module in self.provider.model.named_modules():
            for attribute in ('forward', '_forward_method'):
                if not hasattr(module, attribute):
                    continue
                try:
                    original = original_before_collector(module) if attribute == 'forward' else getattr(module, attribute)
                    binding = _FrameBinding(module, path, attribute, original, self.sources)
                except Exception as error:
                    # A real hook still proves its call. An unverified direct
                    # callable grants no extra observation or non-call proof.
                    self.binding_gaps.append(dict(path=path, module_object_identity=id(module),
                        attribute=attribute, reason=str(error)))
                    continue
                self.bindings.append(binding)
                self.by_module.setdefault(id(module), []).append(binding)
                self.by_code.setdefault(id(binding.code), []).append(binding)
        self.aliases = aliases
        self._prepare_rotary()

    def _prepare_rotary(self):
        source_module = sys.modules.get(_ROTARY_MODULE)
        if source_module is None:
            return
        parent_type, child_type = (getattr(source_module, name, None) for name in (_ROTARY_PARENT, _ROTARY_CHILD))
        for binding in self.bindings:
            parent = binding.module
            child = getattr(parent, 'apply_rotary_emb', None)
            if (binding.attribute != '_forward_method' or type(parent) is not parent_type or type(child) is not child_type
                    or binding.source['sha256'] != _ROTARY_SOURCE_SHA256
                    or binding.function is not getattr(parent_type, 'forward_oot', None)
                    or binding.globals.get('torch') is not torch):
                continue
            function = getattr(getattr(torch.ops.vllm, 'npu_rotary_embedding', None), 'default', None)
            if not isinstance(function, torch._ops.OpOverload) or str(function) != _ROTARY_OPERATOR:
                continue
            self.rotary.append(dict(binding=binding, child=child, operator=function,
                parent_aliases=tuple(name for name, obj in self.aliases if obj is parent),
                child_aliases=tuple(name for name, obj in self.aliases if obj is child),
                returned=[]))

    def _verify_rotary(self, row):
        binding, child = row['binding'], row['child']
        binding.verify()
        require(getattr(binding.module, 'apply_rotary_emb', None) is child
            and binding.globals.get('torch') is torch
            and getattr(getattr(torch.ops.vllm, 'npu_rotary_embedding', None), 'default', None) is row['operator'],
            'OUTPUT39_ROTARY_BRANCH_BINDING_CHANGED')
        aliases = tuple(self.provider.model.named_modules(remove_duplicate=False))
        require(tuple(name for name, obj in aliases if obj is binding.module) == row['parent_aliases']
            and tuple(name for name, obj in aliases if obj is child) == row['child_aliases'],
            'OUTPUT39_ROTARY_ALIAS_BINDING_CHANGED')

    def start(self):
        self._context()
        require(sys.getprofile() is None, 'OUTPUT39_FOREIGN_PROFILE_PRESENT')
        self._prepare_bindings()
        super().start()
        try:
            sys.setprofile(self.profile_callback)
        finally:
            self.profile_owned = sys.getprofile() is self.profile_callback
        require(self.profile_owned, 'OUTPUT39_PROFILE_INSTALL_UNCONFIRMED')

    def verify_live(self):
        require(threading.get_ident() == self.provider._thread, 'OUTPUT39_FRAME_THREAD_CHANGED')
        require(self.profile_owned and sys.getprofile() is self.profile_callback, 'OUTPUT39_PROFILE_OWNER_CHANGED')
        require(self.active, 'OUTPUT39_LATE_FRAME_CALLBACK')
        require(not self.errors, 'OUTPUT39_FRAME_OBSERVATION_FAILED:'+str(self.errors[:1]))

    def _enter(self, name, module):
        require(self.active and not self.failure, 'POSTWRITE_LATE_OBSERVER_CALLBACK')
        self.verify_live()
        for binding in self.by_module.get(id(module), ()):
            binding.verify()
        super()._enter(name, module)
        self.entries.append(dict(module=module, path=name, kind='hook', claimed=False))

    def _leave(self, module):
        self.verify_live()
        require(self.entries and self.entries[-1]['module'] is module
            and self.entries[-1]['kind'] == 'hook', 'OUTPUT39_FRAME_HOOK_STACK_CHANGED')
        super()._leave(module)
        self.entries.pop()

    def _profile(self, frame, event, arg):
        # The hot path for unrelated Python/C events is only a code-id lookup.
        candidates = self.by_code.get(id(frame.f_code))
        if not candidates or event not in ('call', 'return') or self.errors:
            return
        try:
            self.verify_live()
            module = frame.f_locals.get('self')
            matches = [binding for binding in candidates if binding.module is module]
            if not matches and id(module) in self.by_module:
                # e.g. super().forward on a held subclass: this is not that
                # object's current bound forward. Its real outer entry alone
                # owns the call; do not invent another module invocation.
                return
            require(bool(matches), 'OUTPUT39_FRAME_FOREIGN_MODULE')
            # A forward and its delegate can be the same bound function.
            binding = next((item for item in matches if item.attribute == '_forward_method'), matches[0])
            binding.verify(frame)
            if event == 'call':
                self._frame_enter(frame, binding)
            else:
                self._frame_return(frame, binding, arg)
        except BaseException as error:
            # Raising from a profile callback silently removes CPython's
            # profiler. Keep its exact owner and surface the first error at a
            # real hook/API/complete-return boundary before Native admission.
            self.errors.append(str(error))

    def _frame_enter(self, frame, binding):
        require(id(frame) not in self.frames and len(self.events) < 32768,
                'OUTPUT39_FRAME_BOUND_OR_REENTRY_EXCEEDED')
        entry = self.entries[-1] if self.entries else None
        delegate = binding.attribute == '_forward_method' and not any(
            item['binding'] is binding and item['entry'] is entry for item in self.frames.values())
        reuse = (entry is not None and entry['module'] is binding.module and
            (not entry['claimed'] or delegate))
        direct = not reuse
        if direct:
            self._context()
            require(self.active, 'OUTPUT39_LATE_FRAME_CALLBACK')
            entry = dict(module=binding.module, path=binding.path, kind='frame', claimed=True)
            self.entries.append(entry)
            self.provider._stack.append((binding.path, binding.module))
            self.observer.inventory.module_entered(binding.path, binding.module)
        else:
            entry['claimed'] = True
        self.frame_serial += 1
        event = dict(sequence=self.frame_serial, **binding.report(), entry_kind='direct_frame' if direct else 'existing_module_entry',
            returned=False, nonnull_return=False, actual_rotary_returns=0)
        self.events.append(event)
        self.frames[id(frame)] = dict(frame=frame, binding=binding, entry=entry, direct=direct, event=event)

    def _frame_return(self, frame, binding, result):
        current = self.frames.get(id(frame))
        require(current is not None and current['frame'] is frame and current['binding'] is binding,
                'OUTPUT39_FRAME_RETURN_WITHOUT_ENTRY')
        event = current['event']
        event.update(returned=True, nonnull_return=result is not None)
        # Profile return(None) also represents exception unwinding. It cannot
        # prove successful execution, even when an always_call hook follows.
        require(result is not None, 'OUTPUT39_FRAME_NORMAL_RETURN_UNPROVEN')
        if current['direct']:
            require(self.entries and self.entries[-1] is current['entry']
                and self.provider._stack and self.provider._stack[-1][1] is binding.module,
                'OUTPUT39_FRAME_RETURN_STACK_CHANGED')
            self.entries.pop()
            self.provider._stack.pop()
        for row in self.rotary:
            if row['binding'] is binding:
                row['returned'].append(event)
        del self.frames[id(frame)]

    def operator_returned(self, function, result):
        if str(function) != _ROTARY_OPERATOR:
            return
        for row in self.rotary:
            if function is not row['operator']:
                continue
            matches = [item for item in self.frames.values() if item['binding'] is row['binding']]
            if not matches:
                continue  # Another parent's operator does not prove this parent ran.
            require(len(matches) == 1 and result is not None, 'OUTPUT39_ROTARY_ACTUAL_FRAME_UNPROVEN')
            row['binding'].verify(matches[0]['frame'])
            require(getattr(row['binding'].module, 'apply_rotary_emb', None) is row['child'],
                    'OUTPUT39_ROTARY_BRANCH_BINDING_CHANGED')
            matches[0]['event']['actual_rotary_returns'] += 1

    def _complete_discovery(self):
        self.verify_live()
        require(not self.frames and not self.entries, 'OUTPUT39_FRAME_RETURN_INCOMPLETE')
        self.sources.verify()
        for binding in self.bindings:
            binding.verify()
        for row in self.rotary:
            self._verify_rotary(row)
            binding, returned = row['binding'], row['returned']
            count = sum(value for (_, identity), value in self.observer.inventory.module_calls.items()
                        if identity == id(binding.module))
            child_seen = any(identity == id(row['child']) for _, identity in self.observer.inventory.module_calls)
            if (not child_seen and count > 0 and len(returned) == count
                    and all(event['nonnull_return'] and event['actual_rotary_returns'] == 1 for event in returned)):
                self.noncall_proofs.append(dict(schema='demoncat.output39.bounded_rotary_noncall.v1',
                    reason='ACTUAL_ASCEND_FORWARD_OOT_USES_PUBLIC_ROTARY_NOT_APPLY_MODULE',
                    parent_identity=id(binding.module), child_identity=id(row['child']),
                    parent_aliases=list(row['parent_aliases']), child_aliases=list(row['child_aliases']),
                    actual_parent_calls=count, actual_delegate_returns=len(returned),
                    actual_public_rotary_returns=count, delegate_binding=binding.report(),
                    operator=_ROTARY_OPERATOR, operator_identity=id(row['operator']),
                    operator_schema=str(row['operator']._schema), helper_forward_calls=0,
                    workload_only=True, qualification_granted=False))

    def _stop_profile(self):
        self._context()
        if not self.profile_owned:
            require(sys.getprofile() is not self.profile_callback, 'OUTPUT39_PROFILE_READBACK_CHANGED')
            return
        require(sys.getprofile() is self.profile_callback, 'OUTPUT39_PROFILE_OWNER_CHANGED')
        try:
            sys.setprofile(None)
        finally:
            if sys.getprofile() is None:
                self.profile_owned = False
        require(not self.profile_owned, 'OUTPUT39_PROFILE_REMOVAL_UNCONFIRMED')

    def finish(self, output):
        self._complete_discovery()
        self._stop_profile()
        return super().finish(output)

    def close(self, *, unwound):
        self._context()
        require(unwound, 'POSTWRITE_ACTUAL_FORWARD_STILL_ACTIVE')
        self._stop_profile()
        super().close(unwound=True)
        self.frames.clear()
        self.entries.clear()

    def readback(self):
        result = super().readback()
        require(self.profile_owned == (sys.getprofile() is self.profile_callback),
                'OUTPUT39_PROFILE_OWNER_CHANGED')
        result.update(profile_owned=self.profile_owned, actual_frames_pending=len(self.frames),
            actual_frame_events=len(self.events), frame_failures=list(self.errors))
        return result
