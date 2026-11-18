"""Bounded integer-only MoE observations, never a request-row admission API.

The checked map is source-expanded-slot -> sorted destination (row_idx_type=0).
It does not establish grouped-MLP or unpermute numerical/row semantics, nor
authenticate their Native registration. No result here binds a graph tensor.
"""
from dataclasses import dataclass
import hashlib
import os
import struct
import sys
import threading
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack

from .quant_row_provenance import digest, metadata, snapshot, tensor_values
from .serving_context import require


REVIEWED = {
    'vllm_ascend.ops.fused_moe.experts_selector': (
        '92bebd721ecc86a7033065e37e73fe57facabc01cbd754939c014be8ebb810a8',
        {'select_experts': 'route_selector'}),
    'vllm_ascend.ops.fused_moe.moe_comm_method': (
        'a23d681549fa24d69c597f6c1c7d22df84da62efea808f2415a260c4267a8c4f',
        {'MoECommMethod.fused_experts': 'route_pipeline', 'MoECommMethod._apply_mlp': 'route_mlp'}),
    'vllm_ascend.ops.fused_moe.token_dispatcher': (
        '21fd94e3d5eacc578cc3feb794bc5c135c148c2d495c611f224f22c8b96ac0cc',
        {'TokenDispatcherWithAllGather.token_dispatch': 'route_dispatch',
         'TokenDispatcherWithAllGather.token_combine': 'route_combine'}),
}
SOURCE_BUNDLE_SHA256 = '90f151c5fbc830748b0d8cdc04aab32339810b8f48c61d2a08bb08cf3672e264'
SCHEMAS = {
    'topk': '_C_ascend::moe_gating_top_k(Tensor x, int k, int k_group, int group_count, int group_select_mode, int renorm, int norm_type, bool out_flag, float routed_scaling_factor, float eps, Tensor? bias_opt=None) -> (Tensor y, Tensor expert_idx, Tensor out)',
    'routing': '_C_ascend::npu_moe_init_routing_custom(Tensor x, Tensor expert_idx, *, Tensor? scale=None, Tensor? offset=None, int active_num=-1, int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, int quant_mode=0, int[2] active_expert_range=[], int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)',
    'gmm1': 'npu::npu_grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor group_list, Tensor weight_scale, Tensor x_scale, *, Tensor? bias=None, Tensor? offset=None) -> (Tensor, Tensor, Tensor)',
    'gmm2': 'npu::npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, Tensor[]? bias=None, Tensor[]? scale=None, Tensor[]? offset=None, Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None, Tensor[]? per_token_scale=None, Tensor? group_list=None, Tensor[]? activation_input=None, Tensor[]? activation_quant_scale=None, Tensor[]? activation_quant_offset=None, int? split_item=0, int? group_type=None, int? group_list_type=0, int? act_type=0, int[]? tuning_config=None, ScalarType? output_dtype=None) -> Tensor[]',
    'unpermute': 'npu::npu_moe_token_unpermute(Tensor permuted_tokens, Tensor sorted_indices, Tensor? probs=None, bool padded_mode=False, int[]? restore_shape=None) -> Tensor',
    'cumsum': 'aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor',
    'abs': 'aten::abs(Tensor self) -> Tensor',
}
OPERATORS = {schema.split('(')[0].replace('::', '.') + '.default': role for role, schema in SCHEMAS.items()}
SOURCE_ROLES = dict(topk='route_selector', routing='route_dispatch', cumsum='route_mlp',
                    gmm1='route_mlp', gmm2='route_mlp', abs='route_combine', unpermute='route_combine')
INTEGER_LAYOUTS = dict(topk_ids=('torch.int32', (8, 6)), raw_indices=('torch.int32', (48,)),
                      expert_counts=('torch.int64', (64,)), cumulative_counts=('torch.int64', (64,)),
                      absolute_indices=('torch.int32', (48,)))


def _integer_bytes(values, dtype):
    width = {'torch.int32': 'i', 'torch.int64': 'q'}[dtype]
    require(all(type(value) is int for value in values), 'ROUTE_INTEGER_VALUES_REQUIRED')
    return struct.pack('<' + width * len(values), *values)


def _formals(func, args, kwargs):
    """Bind this invocation, retaining supplied/default origins separately."""
    parameters = list(func._schema.arguments)
    names = [item.name for item in parameters]
    require(len(args) <= len(names) and not set(kwargs).difference(names)
        and not set(names[:len(args)]).intersection(kwargs)
        and all(not item.kwarg_only for item in parameters[:len(args)]), 'ROUTE_ACTUAL_ARGUMENT_BINDING_CHANGED')
    bound, origins = {}, {}
    for index, item in enumerate(parameters):
        if index < len(args):
            bound[item.name], origins[item.name] = args[index], 'POSITIONAL'
        elif item.name in kwargs:
            bound[item.name], origins[item.name] = kwargs[item.name], 'KEYWORD'
        else:
            require(item.has_default_value(), 'ROUTE_REQUIRED_ACTUAL_ARGUMENT_MISSING:' + item.name)
            bound[item.name], origins[item.name] = item.default_value, 'CURRENT_SCHEMA_DEFAULT'
    return bound, origins


def _describe(value):
    if isinstance(value, torch.Tensor):
        return dict(tensor=metadata(value))
    if isinstance(value, (list, tuple)):
        return [_describe(item) for item in value]
    if value is None or type(value) in (int, float, str, bool):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    raise RuntimeError('ROUTE_UNREVIEWED_ARGUMENT_TYPE:' + type(value).__name__)


@dataclass(frozen=True)
class TokenRouteProof:
    """Immutable arithmetic result; not a CoverageSession or graph authority."""
    parent_sequence: int
    forward_id: str
    source_to_sorted: tuple
    sorted_slots: tuple
    expert_counts: tuple
    cumulative_counts: tuple
    evidence_sha256: str

    def report(self):
        return dict(schema='demoncat.quant.token_route_proof.v1',
            scope='INTEGER_FULL_RANGE_GATHER_RELATIONS_ONLY', integer_map_valid=True,
            parent_sequence=self.parent_sequence, forward_id=self.forward_id,
            source_to_sorted=list(self.source_to_sorted),
            sorted_slots=[dict(destination=i, source_token=row, route_slot=slot, expert=expert)
                          for i, row, slot, expert in self.sorted_slots],
            expert_counts=list(self.expert_counts), cumulative_counts=list(self.cumulative_counts),
            evidence_sha256=self.evidence_sha256, routed_output_row_proof=False,
            native_registration_proven=False, native_qualified=False, c2=False, c3=False)

    @classmethod
    def validate(cls, evidence):
        require(evidence.get('first_failure') is None and evidence.get('complete') is True,
                'ROUTE_COMPLETE_OBSERVATION_REQUIRED')
        require(type(evidence.get('parent_sequence')) is int and evidence['parent_sequence'] > 0
            and type(evidence.get('forward_id')) is str and bool(evidence['forward_id']),
            'ROUTE_PARENT_AND_FORWARD_ID_REQUIRED')
        stages = evidence['stages']
        order = ('topk', 'routing', 'cumsum', 'gmm1', 'gmm2', 'abs', 'unpermute')
        require(set(stages) == set(order)
            and [stages[k]['sequence'] for k in order] == sorted(set(stages[k]['sequence'] for k in order))
            and all(type(stages[k]['original_calls']) is int and stages[k]['original_calls'] == 1
                    and stages[k]['completed'] is True for k in order),
            'ROUTE_PRODUCER_CONSUMER_CALLS_INCOMPLETE')
        require(evidence.get('joins_complete') is True, 'ROUTE_CURRENT_OBJECT_JOINS_REQUIRED')
        options = stages['routing']['formals']
        expected = dict(active_num=48, expert_capacity=-1, expert_num=64, drop_pad_mode=0,
            expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=1,
            active_expert_range=[0, 64], row_idx_type=0, scale=None, offset=None)
        require(all(type(options.get(k)) is type(v) and options[k] == v for k, v in expected.items()),
                'ROUTE_ONLY_FULL_RANGE_B8_K6_E64_GATHER_COUNT_SUPPORTED')
        values = {}
        require(set(evidence['snapshots']) == set(INTEGER_LAYOUTS), 'ROUTE_FIVE_CURRENT_INTEGER_SNAPSHOTS_REQUIRED')
        for name, (dtype, shape) in INTEGER_LAYOUTS.items():
            row = evidence['snapshots'][name]
            count = 1
            for dimension in shape:
                count *= dimension
            require(row['dtype'] == dtype and row['shape'] == list(shape) and len(row['values']) == count
                and row['metadata_before'] == row['metadata_after'] and row['epoch_before'] == row['epoch_after']
                and row['forward_id'] == evidence['forward_id'] and row['parent_sequence'] == evidence['parent_sequence'],
                'ROUTE_SNAPSHOT_IDENTITY_OR_LAYOUT_CHANGED:' + name)
            raw = _integer_bytes(row['values'], dtype)
            require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                    'ROUTE_SNAPSHOT_BYTES_CHANGED:' + name)
            values[name] = tuple(row['values'])
        ids, index = values['topk_ids'], values['raw_indices']
        require(all(0 <= expert < 64 for expert in ids)
            and all(len(set(ids[start:start + 6])) == 6 for start in range(0, 48, 6)),
            'ROUTE_EXPERT_IDS_OUT_OF_RANGE_OR_DUPLICATED')
        # The reviewed no-drop full-range branch has no negative sentinel.
        # abs() must not silently convert an unsupported negative into a slot.
        require(sorted(index) == list(range(48)) and values['absolute_indices'] == index,
                'ROUTE_GATHER_INDEX_NOT_EXACT_NONNEGATIVE_PERMUTATION')
        counts = tuple(ids.count(expert) for expert in range(64))
        require(values['expert_counts'] == counts, 'ROUTE_EXPERT_COUNTS_DO_NOT_MATCH_ACTUAL_TOPK')
        cumulative, total = [], 0
        for count in counts:
            total += count
            cumulative.append(total)
        require(values['cumulative_counts'] == tuple(cumulative) and total == 48,
                'ROUTE_CUMSUM_DOES_NOT_MATCH_ACTUAL_COUNTS')
        inverse = [None] * 48
        for source, destination in enumerate(index):
            inverse[destination] = source
        sorted_experts = [ids[source] for source in inverse]
        require(sorted_experts == sorted(ids), 'ROUTE_GATHER_MAP_DOES_NOT_MATCH_SORTED_EXPERT_GROUPS')
        slots = tuple((destination, source // 6, source % 6, ids[source]) for destination, source in enumerate(inverse))
        return cls(evidence['parent_sequence'], evidence['forward_id'], index, slots, counts,
                   tuple(cumulative), digest({k: v for k, v in evidence.items() if k != 'token_route_proof'}))


class IntegerRouteObserver:
    """One-thread, 32 parents/forward, 1600 integer bytes/parent; no row bind."""
    def __init__(self, boundary, *, max_parents=32, max_bytes_per_parent=1600):
        require(type(max_parents) is int and 0 < max_parents <= 32
            and max_bytes_per_parent == 1600, 'BOUNDED_INTEGER_ROUTE_OBSERVER_REQUIRED')
        self.boundary, self.graph = boundary, boundary.graph
        self.max_parents, self.max_bytes = max_parents, max_bytes_per_parent
        self.reset()

    def reset(self):
        require(not getattr(self, 'live', {}) and not getattr(self, 'helper_modes', {})
            and not getattr(self, 'helper_active', False), 'ROUTE_OBSERVER_STILL_LIVE')
        self.records, self.live, self.errors = [], {}, []
        self.helper_modes = {}
        self.helper_active, self.helper_events = False, []
        self.total_parents = self.overflow = self.readback_bytes = self.readback_ns = self.observer_ns = 0
        self.peak_owned_cpu_bytes = 0
        self.profile_calls = self.profile_ns = 0

    def _fail(self, row, error):
        row['evidence']['first_failure'] = row['evidence']['first_failure'] or str(error)

    def _current(self, row):
        require(self.boundary.active and not self.boundary.ending and self.graph.active and not self.graph.ending
            and threading.get_ident() == self.boundary.thread and sys.getprofile() is self.boundary.callback
            and row['evidence']['forward_id'] == self.graph.forward_id and not self.graph.overflow
            and not self.graph.errors and not self.boundary.errors, 'ROUTE_CURRENT_OBSERVER_REQUIRED')
        require(torch.is_inference_mode_enabled() and not torch.is_grad_enabled() and not self.boundary.model.training,
                'ROUTE_ACTUAL_EVAL_INFERENCE_THREAD_REQUIRED')
        self.boundary._verify_bindings()

    def _hold(self, row, name, tensor):
        require(isinstance(tensor, torch.Tensor), 'ROUTE_ACTUAL_TENSOR_REQUIRED:' + name)
        before = metadata(tensor)
        row['held'][name] = (tensor, before, self.graph.epochs.get(self.graph.storage_key(before), 0))
        self.graph.watched_storages.append(snapshot(before))

    def _use(self, row, name, tensor):
        held, before, epoch = row['held'][name]
        require(tensor is held and metadata(tensor) == before
            and self.graph.epochs.get(self.graph.storage_key(before), 0) == epoch,
            'ROUTE_PRODUCER_OBJECT_OR_WRITE_EPOCH_CHANGED:' + name)
        return tensor

    def _read(self, row, name, tensor, sequence):
        dtype, shape = INTEGER_LAYOUTS[name]
        require(name not in row['evidence']['snapshots'], 'ROUTE_DUPLICATE_INTEGER_SNAPSHOT:' + name)
        self._current(row)
        before = metadata(tensor)
        epoch = self.graph.epochs.get(self.graph.storage_key(before), 0)
        count = tensor.numel()
        size = count * tensor.element_size()
        require(str(tensor.dtype) == dtype and tuple(tensor.shape) == shape and tensor.is_contiguous()
            and tensor.storage_offset() == 0 and before['storage_bytes'] == size
            and (self.boundary.simulation or tensor.device.type == 'npu')
            and row['bytes'] + size <= self.max_bytes, 'ROUTE_INTEGER_ONLY_EXACT_SPAN_BUDGET_REQUIRED')
        # Reserve before allocation/copy, including a failed attempt.
        row['bytes'] += size
        self.readback_bytes += size
        self.peak_owned_cpu_bytes = max(self.peak_owned_cpu_bytes, size)
        helper = dict(parent_sequence=row['evidence']['parent_sequence'], forward_id=self.graph.forward_id,
            model_sequence=sequence, snapshot=name, source=before, bytes=size, status='STARTED',
            operations=[], total_dispatches=0, overflow=0, cleanup='OWNED',
            classification='OBSERVER_INTEGER_READBACK_NOT_MODEL_INVOCATION')
        self.helper_events.append(helper)
        started = time.monotonic_ns()
        observer = self

        class ReadbackMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                helper['total_dispatches'] += 1
                if len(helper['operations']) >= 4:
                    helper['overflow'] += 1
                    raise RuntimeError('ROUTE_READBACK_HELPER_DISPATCH_BUDGET_EXHAUSTED')
                event = dict(operator=str(func), schema=str(func._schema), original_calls=0, status='STARTED',
                    inputs=[metadata(t) for t in tensor_values((args, kwargs or {}))])
                helper['operations'].append(event)
                operator = str(func)
                require(len(helper['operations']) <= 4 and not func._schema.is_mutable
                    and (operator == 'aten._to_copy.default' and len(helper['operations']) == 1
                         and len(args) == 1 and args[0] is tensor
                         or operator in ('aten.resolve_conj.default', 'aten.resolve_neg.default')
                         and cpu is not None and len(args) == 1 and args[0] is cpu),
                    'ROUTE_UNEXPECTED_READBACK_HELPER_OPERATOR')
                try:
                    event['original_calls'] = 1
                    output = func(*args, **(kwargs or {}))
                    event.update(status='RETURNED', outputs=[metadata(t) for t in tensor_values(output)])
                    return output
                except BaseException as error:
                    event.update(status='FAILED', error=repr(error))
                    raise

        mode = ReadbackMode()
        self.boundary.owned_modes.append(mode)
        self.helper_modes[id(mode)] = (mode, helper)
        self.helper_active = True
        cpu = None
        try:
            mode.__enter__()
            cpu = torch.ops.aten._to_copy.default(tensor, device=torch.device('cpu'), non_blocking=False)
            # No float tensor or activation is read. The one blocking copy
            # owns independent CPU storage, and is separately enumerated.
            require(cpu.device.type == 'cpu' and cpu.dtype == tensor.dtype and cpu.is_contiguous()
                and cpu.data_ptr() != tensor.data_ptr(), 'ROUTE_INDEPENDENT_CPU_INTEGER_COPY_REQUIRED')
            nested = cpu.tolist()
            values = [value for part in nested for value in part] if len(shape) == 2 else nested
            self._current(row)
            after = metadata(tensor)
            after_epoch = self.graph.epochs.get(self.graph.storage_key(after), 0)
            require(after == before and after_epoch == epoch, 'ROUTE_INTEGER_CHANGED_DURING_READBACK')
            raw = _integer_bytes(values, dtype)
            require(len(raw) == size, 'ROUTE_COMPLETE_INTEGER_READBACK_REQUIRED')
            row['evidence']['snapshots'][name] = dict(dtype=dtype, shape=list(shape), values=values,
                bytes=size, sha256=hashlib.sha256(raw).hexdigest(), metadata_before=before, metadata_after=after,
                epoch_before=epoch, epoch_after=after_epoch, sequence=sequence,
                parent_sequence=row['evidence']['parent_sequence'], forward_id=self.graph.forward_id)
            helper['status'] = 'COMPLETE'
        except BaseException as error:
            helper.update(status='FAILED', error=repr(error))
            raise
        finally:
            cpu = None
            self.helper_active = False
            elapsed = time.monotonic_ns() - started
            helper['elapsed_ns'] = elapsed
            self.readback_ns += elapsed
            # Retain a failed/foreign cleanup in the existing retryable mode
            # ledger; never pop somebody else's mode or claim zero resources.
            if any(item is mode for item in _get_current_dispatch_mode_stack()):
                observer.boundary._release_mode(mode)
            else:
                # __enter__ can fail before installing a mode. Do not leave a
                # phantom owner that would make legitimate cleanup impossible.
                require(observer.boundary.owned_modes and observer.boundary.owned_modes[-1] is mode,
                        'ROUTE_READBACK_MODE_LEDGER_CHANGED')
                observer.boundary.owned_modes.pop()
                self.mode_released(mode)

    def mode_released(self, mode):
        retained = self.helper_modes.pop(id(mode), None)
        if retained is not None:
            retained[1]['cleanup'] = 'REMOVED'

    def _registration(self, func, role):
        actual = str(func._schema)
        expected = SCHEMAS[role]
        if self.boundary.simulation:
            expected = expected.replace(expected.split('::')[0], actual.split('::')[0], 1)
        require(actual == expected and not func._schema.is_mutable
            and all(arg.alias_info is None for arg in list(func._schema.arguments) + list(func._schema.returns)),
            'ROUTE_UNREVIEWED_SCHEMA_OR_ALIAS:' + role)
        require(self.boundary.simulation or str(torch.__version__) == '2.9.0+cpu', 'ROUTE_UNREVIEWED_TORCH_BUILD')
        dump = torch._C._dispatch_dump(func._schema.name)
        key = 'CPU' if self.boundary.simulation else 'PrivateUse1'
        # Capture registration, but do not call this an authenticated Native
        # implementation. Any inactive override is an explicit refusal.
        # Torch itself overrides some Meta registrations. Those are not the
        # device dispatch used by these live tensors. Preserve the full dump;
        # refuse a shadow of the observed execution key, never label its
        # source as authenticated merely because it has no inactive entry.
        require(not any(line.startswith(key + ' (inactive)') for line in dump.splitlines()),
                'ROUTE_SHADOWED_DISPATCH_REGISTRATION')
        return dict(actual_schema=actual, actual_dispatch_dump=dump, torch_version=str(torch.__version__),
            observation_key=key, native_registration_proven=False, source_bundle_sha256=SOURCE_BUNDLE_SHA256)

    def before_operator(self, state):
        started = time.monotonic_ns()
        try:
            return self._before(state)
        finally:
            self.observer_ns += time.monotonic_ns() - started

    def _before(self, state):
        if self.helper_active:
            state['event']['observer_helper'] = 'QUANT_INTEGER_ROUTE_READBACK'
            return
        name = self.boundary.operator_aliases.get(state['func'], str(state['func']))
        if name == 'vllm.moe_forward_shared.default':
            self.total_parents += 1
            if self.total_parents > self.max_parents:
                self.overflow += 1
                state['event']['integer_route_evidence'] = dict(status='BUDGET_REFUSED', native_qualified=False)
                return
            record = dict(schema='demoncat.quant.integer_route_observation.v1',
                forward_id=self.graph.forward_id, parent_sequence=state['event']['sequence'],
                process_id=os.getpid(), thread=threading.get_ident(),
                request_scope_sha256=digest(self.boundary.scope),
                first_failure=None, stages={}, snapshots={}, complete=False, joins_complete=False,
                native_registration_proven=False, routed_output_row_proof=False)
            row = dict(evidence=record, held={}, bytes=0, parent=state)
            self.records.append(record)
            self.live[id(state)] = row
            self._hold(row, 'input', state['args'][0] if state['args'] else state['kwargs']['hidden_states'])
            record['parent_input_before_declaration'] = snapshot(row['held']['input'][1])
            # This precedes the parent's conservative schema write epoch. We
            # do not reuse it as a post-invalidation proof or suppress writes.
            return
        row = next((self.live[id(parent)] for parent in reversed(self.boundary.operator_stack)
                    if id(parent) in self.live), None)
        role = OPERATORS.get(name)
        if row is None or role is None or row['evidence']['first_failure'] is not None:
            return
        try:
            self._current(row)
            source = self.boundary._parent(SOURCE_ROLES[role])
            require(source is not None and source['status'] == 'INFLIGHT', 'ROUTE_ACTUAL_SOURCE_FRAME_REQUIRED:' + role)
            require(role not in row['evidence']['stages'], 'ROUTE_DUPLICATE_STAGE:' + role)
            formals, origins = _formals(state['func'], state['args'], state['kwargs'])
            stage = dict(sequence=state['event']['sequence'], original_calls=0, completed=False,
                formals={k: _describe(v) for k, v in formals.items()}, argument_origins=origins,
                source=dict(role=source['role'], source_sha256=source['source_sha256'],
                    function=source['function'], boundary_sequence=source['sequence'], model_path=source['model_path'],
                    owner_object_identity=id(source['input']['self']) if 'self' in source['input'] else None),
                registration=self._registration(state['func'], role))
            row['evidence']['stages'][role] = stage
            state['route_stage'] = (row, role, formals)
            if role == 'topk':
                require(formals['k'] == 6 and source['input']['hidden_states'] is row['held']['input'][0],
                        'ROUTE_ACTUAL_SELECTOR_INPUT_REQUIRED')
            elif role == 'routing':
                # Parent declaration writes were already observed; only the
                # current producer object/metadata are joined, no row grant.
                require(formals['x'] is row['held']['input'][0], 'ROUTE_ACTUAL_PARENT_INPUT_REQUIRED')
                self._use(row, 'topk_ids', formals['expert_idx'])
                dispatch = source['input']['token_dispatch_input']
                require(dispatch.hidden_states is formals['x'] and dispatch.topk_ids is formals['expert_idx']
                    and dispatch.topk_weights is row['held']['probs'][0]
                    and dispatch.routing.apply_router_weight_on_input is False
                    and dispatch.routing.expert_map is None and dispatch.routing.pertoken_scale is None
                    and dispatch.routing.global_redundant_expert_num == 0,
                    'ROUTE_UNTRANSFORMED_ALLGATHER_INPUT_REQUIRED')
            elif role == 'cumsum':
                self._use(row, 'expert_counts', formals['self'])
                require(formals['dim'] == 0 and formals['dtype'] in (None, torch.int64), 'ROUTE_COUNT_CUMSUM_AXIS_CHANGED')
            elif role == 'gmm1':
                for key, held in (('x', 'routed'), ('group_list', 'cumulative_counts'), ('x_scale', 'route_scale')):
                    self._use(row, held, formals[key])
            elif role == 'gmm2':
                require(len(formals['x']) == 1 and len(formals['per_token_scale'] or []) == 1
                    and formals['group_list_type'] == 1, 'ROUTE_GROUPED_MLP_COUNT_INPUT_REQUIRED')
                self._use(row, 'mlp1', formals['x'][0])
                self._use(row, 'mlp_scale', formals['per_token_scale'][0])
                self._use(row, 'expert_counts', formals['group_list'])
            elif role == 'abs':
                self._use(row, 'raw_indices', formals['self'])
            elif role == 'unpermute':
                for key, held in (('permuted_tokens', 'mlp2'), ('sorted_indices', 'absolute_indices'), ('probs', 'probs')):
                    self._use(row, held, formals[key])
                require(formals['padded_mode'] is False and formals['restore_shape'] is None,
                        'ROUTE_ONLY_UNPADDED_UNPERMUTE_OBSERVATION_SUPPORTED')
        except BaseException as error:
            self._fail(row, error)

    def after_operator(self, state, output=None, error=None):
        started = time.monotonic_ns()
        try:
            self._after(state, output, error)
        finally:
            self.observer_ns += time.monotonic_ns() - started

    def _after(self, state, output, error):
        if state is None:
            return
        child = state.pop('route_stage', None)
        if child is not None:
            row, role, formals = child
            stage = row['evidence']['stages'][role]
            stage['original_calls'] = state['event']['original_calls']
            if error is not None:
                self._fail(row, 'ROUTE_ORIGINAL_CALL_FAILED:' + repr(error))
            if row['evidence']['first_failure'] is None:
                try:
                    self._current(row)
                    require(self._registration(state['func'], role) == stage['registration'],
                            'ROUTE_DISPATCH_REGISTRATION_CHANGED_DURING_CALL')
                    sequence = state['event']['sequence']
                    names = {'topk': ('probs', 'topk_ids', None),
                             'routing': ('routed', 'raw_indices', 'expert_counts', 'route_scale'),
                             'gmm1': ('mlp1', 'mlp_scale', None), 'gmm2': ('mlp2',)}.get(role)
                    if names:
                        require(isinstance(output, (tuple, list)) and len(output) == len(names), 'ROUTE_ACTUAL_RETURN_ARITY_CHANGED')
                        for name, tensor in zip(names, output):
                            if name is not None:
                                self._hold(row, name, tensor)
                        for name in names:
                            if name in INTEGER_LAYOUTS:
                                self._read(row, name, row['held'][name][0], sequence)
                    elif role in ('cumsum', 'abs'):
                        name = 'cumulative_counts' if role == 'cumsum' else 'absolute_indices'
                        self._hold(row, name, output)
                        self._read(row, name, output, sequence)
                    else:
                        require(isinstance(output, torch.Tensor), 'ROUTE_ACTUAL_UNPERMUTE_RETURN_REQUIRED')
                    stage.update(completed=True, outputs=[metadata(t) for t in tensor_values(output)])
                    # The real pipeline subsequently resizes this retired
                    # temporary. Its already-completed consumption remains a
                    # snapshot, never a late storage-address lookup.
                    if role == 'gmm1':
                        row['held'].pop('routed', None)
                        row['held'].pop('route_scale', None)
                except BaseException as caught:
                    self._fail(row, caught)
        row = self.live.pop(id(state), None)
        if row is not None:
            try:
                if error is not None:
                    self._fail(row, 'ROUTE_PARENT_FAILED:' + repr(error))
                evidence = row['evidence']
                evidence['complete'] = set(evidence['stages']) == set(SCHEMAS) and all(
                    stage['completed'] for stage in evidence['stages'].values())
                evidence['joins_complete'] = evidence['complete'] and evidence['first_failure'] is None
                evidence['readback_bytes'] = row['bytes']
                try:
                    evidence['token_route_proof'] = TokenRouteProof.validate(evidence).report()
                except BaseException as caught:
                    self._fail(row, caught)
                    evidence['token_route_proof'] = None
                state['event']['integer_route_evidence'] = dict(record_index=self.records.index(evidence),
                    sha256=digest(evidence), integer_map_valid=evidence['token_route_proof'] is not None,
                    first_failure=evidence['first_failure'], routed_output_row_proof=False)
            finally:
                row['held'].clear()
                row.pop('parent', None)

    def report(self):
        return snapshot(dict(schema='demoncat.quant.integer_route_census.v1', records=self.records,
            helper_calls=self.helper_events, total_parents=self.total_parents, overflow=self.overflow,
            readback_bytes=self.readback_bytes, readback_ns=self.readback_ns, observer_ns=self.observer_ns,
            retained_integer_payload_bytes=sum(item['bytes'] for row in self.records for item in row['snapshots'].values()),
            helper_operator_calls=sum(len(item['operations']) for item in self.helper_events),
            helper_total_dispatches=sum(item['total_dispatches'] for item in self.helper_events),
            helper_dispatch_overflow=sum(item['overflow'] for item in self.helper_events),
            source_profile_callbacks=self.profile_calls, source_profile_ns=self.profile_ns,
            max_parents=self.max_parents, max_integer_bytes_per_parent=self.max_bytes,
            peak_owned_cpu_copy_bytes=self.peak_owned_cpu_bytes, live_parent_refs=len(self.live),
            live_tensor_refs=sum(len(row['held']) for row in self.live.values()) + len(self.helper_modes),
            retained_helper_modes=len(self.helper_modes),
            helper_active=self.helper_active, native_registration_proven=False, routed_output_row_proof=False,
            native_qualified=False, c2=False, c3=False))
