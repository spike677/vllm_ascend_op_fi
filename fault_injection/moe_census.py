"""Read-only exact-source MoE census across an opaque custom-op boundary.

A Python profile callback observes the real producer/MLP/combine frames. It
enters a read-only dispatch mode only inside their actual token_combine call,
so the nested native API is visible without replacing a model method, changing
routing, re-executing an operator, or reconstructing an output. This is census
evidence, never fault qualification or a native permutation-value proof.
"""
from collections import Counter
import hashlib
import marshal
from pathlib import Path
import sys
import threading

from .tensor_observer import metadata, tensors


REVIEWED = {
    'vllm_ascend.ops.fused_moe.experts_selector': (
        '92bebd721ecc86a7033065e37e73fe57facabc01cbd754939c014be8ebb810a8',
        {'select_experts': 'topk_producer'}),
    'vllm_ascend.ops.fused_moe.moe_comm_method': (
        'a23d681549fa24d69c597f6c1c7d22df84da62efea808f2415a260c4267a8c4f',
        {'MoECommMethod.fused_experts': 'expert_pipeline', 'MoECommMethod._apply_mlp': 'expert_mlp'}),
    'vllm_ascend.ops.fused_moe.token_dispatcher': (
        '21fd94e3d5eacc578cc3feb794bc5c135c148c2d495c611f224f22c8b96ac0cc',
        {'TokenDispatcherWithAllGather.token_dispatch': 'routing_producer',
         'TokenDispatcherWithAllGather.token_combine': 'native_combine',
         'TokenDispatcherWithAll2AllV.token_dispatch': 'routing_producer',
         'TokenDispatcherWithAll2AllV.token_combine': 'native_combine',
         'TokenDispatcherWithMC2.token_dispatch': 'routing_producer',
         'TokenDispatcherWithMC2.token_combine': 'native_combine'})}

_FIELDS = ('hidden_states', 'router_logits', 'topk_weights', 'topk_ids', 'expanded_row_idx',
    'group_list', 'dynamic_scale', 'combine_metadata', 'quant', 'routing', 'routed_out',
    'per_token_scale', 'pertoken_scale', 'active_num', 'expert_num', 'top_k', 'topk_group',
    'num_expert_group', 'renormalize', 'apply_router_weight_on_input', 'restore_shape',
    'reversed_local_input_permutation_mapping', 'reversed_global_input_permutation_mapping',
    'hidden_shape_before_permute', 'hidden_shape', 'dispatch_with_quant')


def describe(value, depth=0):
    import torch
    if isinstance(value, torch.Tensor):
        return metadata(value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (tuple, list)) and len(value) <= 16:
        return [describe(item, depth+1) for item in value]
    if depth < 3:
        fields = getattr(type(value), '__dataclass_fields__', {})
        if fields:
            return {key: describe(getattr(value, key), depth+1) for key in _FIELDS if key in fields}
    return dict(type=type(value).__module__+'.'+type(value).__qualname__)


class NaturalMoETrace:
    def __init__(self, model, bindings, sources, *, max_events=10000, simulation=False):
        self.model_paths = {id(module): path or '<root>' for path, module in model.named_modules()}
        self.bindings = {function.__code__: (function, role) for function, role in bindings}
        self.sources = sources
        self.max_events, self.simulation = max_events, simulation
        self.thread = None; self.callback = self._profile
        self.request_id = None
        self.rows, self.native_rows, self.errors, self.pending = [], [], [], {}
        # A Python frame returning does not prove that its dispatch mode left
        # the real stack. Keep ownership separate from frame bookkeeping.
        self.owned_modes, self.returned_frames = [], set()
        self.active = False

    @classmethod
    def from_runtime(cls, model):
        bindings, sources = [], {}
        for name, (expected, roles) in REVIEWED.items():
            module = sys.modules.get(name)
            if module is None:
                raise RuntimeError('MOE_RUNTIME_SOURCE_NOT_LOADED:' + name)
            path = Path(module.__file__).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError('UNREVIEWED_MOE_CENSUS_SOURCE:' + name)
            sources[name] = dict(path=str(path), sha256=digest, roles=roles, methods={})
            for qualname, role in roles.items():
                function = module
                for key in qualname.split('.'):
                    function = getattr(function, key, None)
                # Some communication subclasses inherit token_combine. The
                # actual loaded function is retained; no method is rebound.
                if function is None or not hasattr(function, '__code__'):
                    raise RuntimeError('MOE_ACTUAL_PYTHON_BOUNDARY_MISSING:' + qualname)
                if function.__globals__ is not vars(module) or Path(function.__code__.co_filename).resolve() != path:
                    raise RuntimeError('MOE_CENSUS_FOREIGN_CALLABLE:' + qualname)
                sources[name]['methods'][qualname] = dict(actual_qualname=function.__qualname__,
                    first_line=function.__code__.co_firstlineno, function_object_id=id(function),
                    code_sha256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest())
                bindings.append((function, role))
        return cls(model, bindings, sources)

    def _model_path(self, frame):
        while frame is not None:
            owner = frame.f_locals.get('self')
            if id(owner) in self.model_paths:
                return self.model_paths[id(owner)]
            frame = frame.f_back
        return '<no_model_owner>'

    def _open_native(self, row):
        from torch.utils._python_dispatch import TorchDispatchMode
        parent = self
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if len(parent.native_rows) >= parent.max_events:
                    raise RuntimeError('MOE_NATIVE_CENSUS_BUDGET_EXHAUSTED')
                schema = getattr(func, '_schema', None)
                arguments = ({arg.name: arg.default_value for arg in schema.arguments}
                             if schema is not None else {})
                arguments.update(zip([arg.name for arg in schema.arguments] if schema is not None else (), args))
                arguments.update(kwargs or {})
                native = dict(sequence=len(parent.native_rows), request_id=parent.request_id,
                    combine_sequence=row['sequence'], model_path=row['model_path'],
                    operator=str(func), schema=str(schema),
                    arguments={name: describe(value) for name, value in arguments.items()},
                    input=[metadata(value) for value in tensors((args, kwargs or {}))],
                    original_calls=0)
                parent.native_rows.append(native)
                result = func(*args, **(kwargs or {}))
                native.update(output=[metadata(value) for value in tensors(result)],
                              original_calls=1, original_returned=True)
                return result
        return self._enter_native_mode(Observe())

    def _enter_native_mode(self, mode):
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
        self.owned_modes.append(mode)
        try:
            mode.__enter__()
        except BaseException:
            if not any(item is mode for item in _get_current_dispatch_mode_stack()):
                self.owned_modes.remove(mode)
            raise
        return mode

    def _retire_returned_frames(self):
        for frame_id in tuple(self.returned_frames):
            item = self.pending.get(frame_id)
            if item is None or item[1] is None or not any(mode is item[1] for mode in self.owned_modes):
                self.pending.pop(frame_id, None)
                self.returned_frames.remove(frame_id)

    def _close_native_mode(self, mode):
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
        if threading.get_ident() != self.thread:
            raise RuntimeError('MOE_NATIVE_MODE_OWNER_THREAD_CHANGED')
        stack = _get_current_dispatch_mode_stack()
        if not self.owned_modes or not any(item is mode for item in self.owned_modes) or not stack or stack[-1] is not mode:
            raise RuntimeError('MOE_CENSUS_NATIVE_MODE_OWNERSHIP_CHANGED')
        try:
            mode.__exit__(None, None, None)
        finally:
            # Exit can throw before or after the actual pop. Only the latter
            # retires this handle; the original exception still propagates.
            if not any(item is mode for item in _get_current_dispatch_mode_stack()):
                self.owned_modes.remove(mode)
                self._retire_returned_frames()
        if any(item is mode for item in self.owned_modes):
            raise RuntimeError('MOE_CENSUS_NATIVE_MODE_REMOVAL_UNCONFIRMED')

    def _profile(self, frame, event, result):
        if event not in ('call', 'return') or frame.f_code not in self.bindings:
            return
        function, role = self.bindings[frame.f_code]
        try:
            if threading.get_ident() != self.thread or not self.request_id:
                raise RuntimeError('MOE_TRACE_REQUIRES_OWNED_ENQUEUED_REQUEST_THREAD')
            if event == 'call':
                if len(self.rows) >= self.max_events:
                    raise RuntimeError('MOE_PYTHON_CENSUS_BUDGET_EXHAUSTED')
                row = dict(sequence=len(self.rows), request_id=self.request_id, role=role,
                    function=function.__module__+'.'+function.__qualname__,
                    function_object_id=id(function), model_path=self._model_path(frame),
                    input={key: describe(value) for key, value in frame.f_locals.items()
                        if key in _FIELDS or key in ('token_dispatch_input', 'fused_experts_input',
                                                   'mlp_compute_input')}, returned=False)
                self.rows.append(row)
                self.pending[id(frame)] = (row, None)
                if role == 'native_combine':
                    self.pending[id(frame)] = (row, self._open_native(row))
            elif id(frame) in self.pending:
                frame_id = id(frame)
                row, mode = self.pending[frame_id]
                self.returned_frames.add(frame_id)
                if mode is not None:
                    self._close_native_mode(mode)
                self._retire_returned_frames()
                row.update(output=describe(result), returned=result is not None)
        except BaseException as exc:
            # A profile callback must not alter model control flow. An error is
            # retained and makes the census incomplete; no acceptance follows.
            self.errors.append(repr(exc))

    def __enter__(self):
        if self.active or self.owned_modes or self.pending or sys.getprofile() is not None:
            raise RuntimeError('FOREIGN_PYTHON_PROFILE_OWNER_OR_ACTIVE_MOE_TRACE')
        self.thread = threading.get_ident(); self.active = True
        sys.setprofile(self.callback)
        return self

    def close(self):
        if not self.active and not self.owned_modes and not self.pending:
            return
        if threading.get_ident() != self.thread or sys.getprofile() is not (self.callback if self.active else None):
            raise RuntimeError('MOE_PROFILE_OWNERSHIP_CHANGED')
        if set(self.pending) - self.returned_frames:
            raise RuntimeError('MOE_PYTHON_BOUNDARY_STILL_INFLIGHT')
        for mode in reversed(self.owned_modes[:]):
            self._close_native_mode(mode)
        self._retire_returned_frames()
        if self.pending or self.owned_modes:
            raise RuntimeError('MOE_NATIVE_MODE_CLEANUP_PENDING')
        try:
            sys.setprofile(None)
        finally:
            if sys.getprofile() is None:
                self.active = False

    def report(self):
        unpermute = [row for row in self.native_rows
            if row['schema'].startswith('npu::npu_moe_token_unpermute(')]
        by_request = Counter(row['request_id'] for row in unpermute)
        complete = (not self.active and not self.pending and not self.owned_modes and not self.errors
            and all(row['returned'] for row in self.rows)
            and all(row.get('original_returned') and row['original_calls'] == 1 for row in self.native_rows))
        return dict(simulation=self.simulation, complete=complete,
            sources=self.sources, python_boundaries=self.rows, native_operators=self.native_rows,
            owned_native_modes=len(self.owned_modes), pending_python_boundaries=len(self.pending),
            profile_active=self.active,
            observed_unpermute_calls=len(unpermute), observed_unpermute_per_request=dict(by_request),
            role_counts=dict(Counter(row['role'] for row in self.rows)), errors=self.errors,
            mutations=0, model_methods_replaced=False, operator_reexecution=False,
            tensor_values_read=False, complete_permutation_value_proven=False,
            qualification_granted=False, notes='Metadata/source/actual-call census only; eligibility still needs values')

    def __exit__(self, *exc):
        self.close()
        return False
