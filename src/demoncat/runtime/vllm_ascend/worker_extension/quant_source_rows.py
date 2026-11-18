"""Exact-source, actual-object row bridges inside opaque Ascend operations.

This module observes existing calls. It never calls a model/operator twice and
never turns a structural module list into a qualified Native target set.
"""
import hashlib
import marshal
import math
from pathlib import Path
import sys
import threading
import types
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode, _get_current_dispatch_mode_stack

from .quant_row_provenance import digest, exit_owned_mode, metadata, snapshot, tensor_values
from .serving_context import require


REVIEWED = {
    'vllm_ascend.ops.mla': (
        '7f2260f768f5549c2ef00d18fc0457b3c6a10e8184f1aba20ed1b172bcfc9d39',
        {'mla_forward': 'mla_opaque'}),
    'vllm_ascend.attention.mla_v1': (
        '17892014ecd431a1804fb81f5380509275e1440b41e1f9e0870ba8440c365516',
        {'AscendMLAImpl.forward': 'mla_forward',
         'AscendMLAImpl._mla_preprocess': 'mla_standard_preprocess',
         'AscendMLAImpl._q_proj_and_k_up_proj': 'mla_q',
         'AscendMLAImpl.rope_single': 'mla_rope',
         'AscendMLAImpl.mla_preprocess_decode': 'mla_preprocess',
         'AscendMLAImpl._forward_decode': 'mla_decode',
         'AscendMLAImpl._v_up_proj': 'mla_v'}),
    'vllm_ascend.ops.register_custom_ops': (
        'c5eaccde6568f6f480c00a07cc582c8650e6849f25933c8e82787f4b0477ca9b',
        {'_maybe_all_gather_and_maybe_unpad_impl': 'no_gather',
         '_maybe_chunk_residual_impl': 'no_chunk',
         '_maybe_pad_and_reduce_impl': 'embedding_pad'}),
    'vllm_ascend.ops.vocab_parallel_embedding': (
        '27d9c4f38c11d051b22dadc903ed0009269a303199639287ba937d882442f094',
        {'AscendVocabParallelEmbedding._forward_origin': 'embedding_origin'}),
    'vllm.model_executor.layers.vocab_parallel_embedding': (
        '9dff6cab7c17798991cc48e854f275a6b596cc637e5694f166c20cdc68d31879',
        {'UnquantizedEmbeddingMethod.embedding': 'embedding_lookup'}),
    'vllm.distributed.communication_op': (
        '3c3f8ac60db38ece891d39dc1e2f6f1947ad438685893d2e5777d072320bd1b8',
        {'tensor_model_parallel_all_reduce': 'embedding_tp_reduce'}),
    'vllm.distributed.parallel_state': (
        '1cb040ec56cca83fc82613413b8bda2aa697c71981ba895189713e902ff4e570',
        {'GroupCoordinator.all_reduce': 'embedding_comm_return', 'get_tp_group': 'embedding_get_tp'}),
}


def _code_nodes(code):
    yield code
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            yield from _code_nodes(item)


def _resolve(module, qualname):
    value = module
    for name in qualname.split('.'):
        value = getattr(value, name)
    return value


def source_bindings(reviewed=REVIEWED):
    """Bind loaded globals, function objects and compiled reviewed code together."""
    bindings, sources = [], {}
    for name, (expected, roles) in reviewed.items():
        module = sys.modules.get(name)
        require(module is not None, 'QUANT_ROW_SOURCE_NOT_LOADED:' + name)
        path = Path(module.__file__).resolve()
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, 'QUANT_ROW_SOURCE_CHANGED:' + name)
        functions = {}
        for qualname, role in roles.items():
            function = _resolve(module, qualname)
            require(isinstance(function, types.FunctionType) and function.__globals__ is vars(module)
                    and Path(function.__code__.co_filename).resolve() == path,
                    'QUANT_ROW_FOREIGN_FUNCTION:' + qualname)
            compiled = compile(raw, function.__code__.co_filename, 'exec', optimize=sys.flags.optimize)
            matches = [code for code in _code_nodes(compiled) if code.co_qualname == function.__code__.co_qualname]
            # marshal encodes reference/interning flags that can differ after a
            # legitimate pyc import. Code equality compares executable fields;
            # retain explicit filename/line checks as well as globals ownership.
            require(len(matches) == 1 and matches[0] == function.__code__
                    and matches[0].co_filename == function.__code__.co_filename
                    and matches[0].co_firstlineno == function.__code__.co_firstlineno,
                    'QUANT_ROW_LOADED_CODE_DIFFERS_FROM_REVIEWED_SOURCE:' + qualname)
            binding = dict(module=module, qualname=qualname, function=function, role=role, source_sha256=expected)
            bindings.append(binding)
            functions[qualname] = dict(role=role, function_object_identity=id(function),
                code_sha256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest())
        sources[name] = dict(path=str(path), sha256=expected, functions=functions)
    return bindings, sources


def _arguments(func, args, kwargs):
    schema = getattr(func, '_schema', None)
    require(schema is not None, 'ACTUAL_OPERATOR_SCHEMA_REQUIRED')
    names = [arg.name for arg in schema.arguments]
    require(len(args) <= len(names) and not set(kwargs).difference(names), 'OPERATOR_ARGUMENT_BINDING_CHANGED')
    result = {arg.name: arg.default_value for arg in schema.arguments if arg.has_default_value()}
    result.update(zip(names, args))
    require(not set(names[:len(args)]).intersection(kwargs), 'OPERATOR_ARGUMENT_DUPLICATED')
    result.update(kwargs)
    return result


def _same_extent(left, right):
    a, b = metadata(left), metadata(right)
    return all(a[name] == b[name] for name in ('storage_address', 'storage_bytes', 'storage_identity',
        'offset', 'shape', 'stride', 'dtype', 'device'))


def _disjoint(left, right):
    a, b = metadata(left), metadata(right)
    return a['device'] != b['device'] or (a['storage_address'] + a['storage_bytes'] <= b['storage_address']
        or b['storage_address'] + b['storage_bytes'] <= a['storage_address'])


def _singleton_view_axis(before, after, axis):
    """Prove an actual view only inserted/removed dimensions of size one.

    MLA splits and transposes leave noncontiguous query/rope tensors. Adding
    the S=1 dimension preserves every nontrivial coordinate and its address;
    contiguity is neither necessary nor evidence for that particular edge.
    No feature/request dimension is split, merged, reordered or broadcast.
    """
    shared = ('storage_address', 'storage_bytes', 'storage_identity', 'offset', 'dtype', 'device')
    if (any(before[key] != after[key] for key in shared)
            or not 0 <= axis < len(before['shape']) or before['shape'][axis] <= 1):
        return None
    source = [(index, size, stride) for index, (size, stride) in enumerate(zip(before['shape'], before['stride']))
              if size != 1]
    target = [(index, size, stride) for index, (size, stride) in enumerate(zip(after['shape'], after['stride']))
              if size != 1]
    if (any(size <= 0 or stride <= 0 for _, size, stride in source + target)
            or [(size, stride) for _, size, stride in source] != [(size, stride) for _, size, stride in target]):
        return None
    ordinal = next((index for index, (dimension, _, _) in enumerate(source) if dimension == axis), None)
    return target[ordinal][0] if ordinal is not None else None


class SourceBoundRows:
    """One thread, one source-checked profile owner, bounded nested dispatch."""
    def __init__(self, graph, model, bindings, sources, *, max_boundaries=4096, simulation=False, operator_aliases=None):
        self.graph, self.model = graph, model
        self.model_paths = {id(module): path or '<root>' for path, module in model.named_modules()}
        self.bindings = {item['function'].__code__: item for item in bindings}
        self.sources = snapshot(sources)
        self.max_boundaries, self.simulation = max_boundaries, simulation
        require(type(max_boundaries) is int and 0 < max_boundaries <= 8192, 'BOUNDED_SOURCE_ROW_PROFILE_REQUIRED')
        require(not operator_aliases or simulation, 'OPERATOR_ALIASES_ARE_CPU_FIXTURES_ONLY')
        self.operator_aliases = operator_aliases or {}
        self.callback = self._profile
        self.active = False
        self.pending, self.stack, self.rows, self.errors = {}, [], [], []
        self.operator_stack = []
        self.constants = {}
        self.impl_methods = []
        self.scope = None
        self.alias_parents = {}
        self.metadata_producers = None
        self.owned_modes = []
        self.profile_released = True
        self.ending = False
        self.input_ids = self.input_ids_metadata = None
        self.embedding_attempts = 0
        self.moe = None

    @classmethod
    def from_runtime(cls, graph, model):
        from .quant_moe_rows import SharedMoERows, REVIEWED as MOE_REVIEWED, has_runtime_owner
        moe_enabled = has_runtime_owner(model)
        bindings, sources = source_bindings(dict(REVIEWED, **MOE_REVIEWED) if moe_enabled else REVIEWED)
        value = cls(graph, model, bindings, sources)
        if moe_enabled:
            value.moe = SharedMoERows(value)
        # Freeze the actual installed implementation and its derived weights.
        # A later tensor replacement cannot become a new trusted constant.
        for _, module in model.named_modules():
            attention = getattr(module, 'mla_attn', None)
            impl = getattr(attention, 'impl', None)
            if impl is None:
                continue
            for binding in bindings:
                if binding['qualname'].startswith('AscendMLAImpl.'):
                    method = binding['qualname'].rsplit('.', 1)[1]
                    require(getattr(getattr(impl, method, None), '__func__', None) is binding['function'],
                            'UNREVIEWED_ACTUAL_MLA_IMPLEMENTATION_METHOD:' + method)
                    value.impl_methods.append((impl, method, binding['function']))
            for name in ('W_UK_T', 'W_UV'):
                tensor = getattr(impl, name, None)
                if isinstance(tensor, torch.Tensor):
                    value.constants[(id(impl), name)] = (impl, tensor, metadata(tensor))
        return value

    def begin(self, scope, *, input_ids=None):
        require(not self.active and not self.owned_modes and sys.getprofile() is None
                and self.graph.active and not self.graph.ending,
                'EXCLUSIVE_QUANT_ROW_PROFILE_OWNER_REQUIRED')
        self.scope = snapshot(scope)
        self.thread = threading.get_ident()
        self.pending, self.stack, self.rows, self.errors, self.operator_stack = {}, [], [], [], []
        self.alias_parents = {}
        input_ids_metadata = metadata(input_ids) if isinstance(input_ids, torch.Tensor) else None
        self.embedding_attempts = 0
        self.profile_released, self.ending = True, False
        self.graph.watched_storages.extend(snapshot(item[2]) for item in self.constants.values())
        if input_ids_metadata is not None:
            self.graph.watched_storages.append(snapshot(input_ids_metadata))
        if self.metadata_producers is not None:
            self.graph.watched_storages.extend(self.metadata_producers.watched_storages())
        for name, row in self.sources.items():
            require(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256'],
                    'QUANT_ROW_SOURCE_CHANGED_BEFORE_FORWARD:' + name)
        self._verify_bindings()
        self.input_ids, self.input_ids_metadata = input_ids, input_ids_metadata
        self.active = True
        try:
            sys.setprofile(self.callback)
        finally:
            self.profile_released = sys.getprofile() is not self.callback

    def _verify_bindings(self):
        require(all(item['function'].__code__ is code and item['function'].__globals__ is vars(item['module'])
            and _resolve(item['module'], item['qualname']) is item['function']
            for code, item in self.bindings.items()), 'QUANT_ROW_FROZEN_FUNCTION_CHANGED')
        require(all(getattr(getattr(owner, name, None), '__func__', None) is function
                    for owner, name, function in self.impl_methods), 'QUANT_ROW_INSTALLED_IMPL_METHOD_CHANGED')

    def end(self):
        require(self.active and threading.get_ident() == self.thread
                and sys.getprofile() is (None if self.profile_released else self.callback),
                'QUANT_ROW_PROFILE_OWNER_CHANGED')
        require(not self.pending and not self.stack and not self.operator_stack, 'QUANT_ROW_BOUNDARIES_STILL_INFLIGHT')
        self.ending = True
        while self.owned_modes:
            self._release_mode(self.owned_modes[-1])
        if not self.profile_released:
            sys.setprofile(None)
            self.profile_released = sys.getprofile() is None
            require(self.profile_released, 'QUANT_ROW_PROFILE_NOT_RELEASED')
        if self.metadata_producers is not None:
            self.metadata_producers.release()
        self.input_ids = self.input_ids_metadata = None
        self.active = False
        return self.report()

    def _release_mode(self, mode):
        require(self.owned_modes and self.owned_modes[-1] is mode, 'QUANT_ROW_NESTED_MODE_ORDER_CHANGED')
        removed, error = exit_owned_mode(mode, 'QUANT_ROW_NESTED_MODE_OWNER_CHANGED')
        if removed:
            self.owned_modes.pop()
        if error is not None:
            raise error

    def _parent(self, role):
        return next((row for row in reversed(self.stack) if row['role'] == role), None)

    def permits_public_rows(self, operators):
        if not self.active or self.ending or self.errors:
            return False
        for operator in operators:
            matches = [state for state in self.operator_stack if state['func'] is operator]
            if len(matches) != 1:
                return False
            state = matches[0]
            role = state['expected_role']
            if role == 'moe_opaque':
                if self.moe is None or not self.moe.permits(state):
                    return False
            elif role != 'mla_opaque':
                return False
            actual = [row for row in state['actual_boundaries'] if row['role'] == role]
            if len(actual) != 1 or actual[0]['status'] != 'INFLIGHT':
                return False
        return True

    def _live_constant(self, owner, name, tensor):
        item = self.constants.get((id(owner), name))
        return (item is not None and item[0] is owner and item[1] is tensor
                and getattr(owner, name, None) is tensor and metadata(tensor) == item[2]
                and self.graph.epochs.get(self.graph.storage_key(item[2]), 0) == 0)

    def _model_path(self, frame):
        while frame is not None:
            owner = frame.f_locals.get('self')
            if id(owner) in self.model_paths:
                return self.model_paths[id(owner)]
            frame = frame.f_back
        return None

    def _mode(self):
        current = _get_current_dispatch_mode()
        if current is self.graph or any(mode is current for mode in self.owned_modes):
            return None
        require(current is None, 'FOREIGN_MODE_INSIDE_OPAQUE_QUANT_BOUNDARY')
        graph = self.graph
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return graph.__torch_dispatch__(func, types, args, kwargs)
        mode = Observe()
        self.owned_modes.append(mode)
        try:
            mode.__enter__()
        except BaseException:
            if not any(value is mode for value in _get_current_dispatch_mode_stack()):
                self.owned_modes.pop()
            raise
        return mode

    def _profile(self, frame, event, result):
        if event not in ('call', 'return') or frame.f_code not in self.bindings:
            return
        row = None
        try:
            binding = self.bindings[frame.f_code]
            # These global helpers run elsewhere too. Only the actual checked
            # embedding frame may attach them to a scheduler row producer.
            if (binding['role'].startswith('embedding_') and binding['role'] != 'embedding_origin'
                    and self._parent('embedding_origin') is None):
                return
            require(self.active and not self.ending and threading.get_ident() == self.thread and frame.f_globals is vars(binding['module'])
                    and _resolve(binding['module'], binding['qualname']) is binding['function']
                    and binding['function'].__code__ is frame.f_code,
                    'QUANT_ROW_BOUNDARY_CALLABLE_IDENTITY_CHANGED')
            if event == 'call':
                require(len(self.rows) < self.max_boundaries, 'QUANT_ROW_BOUNDARY_BUDGET_EXHAUSTED')
                path = self._model_path(frame)
                require(path is not None, 'QUANT_ROW_BOUNDARY_HAS_NO_ACTUAL_MODEL_OWNER')
                row = dict(sequence=len(self.rows) + 1, role=binding['role'], model_path=path,
                    source_sha256=binding['source_sha256'], function=binding['qualname'],
                    input={name: value for name, value in frame.f_locals.items()},
                    snapshots={id(t): metadata(t) for t in tensor_values(tuple(frame.f_locals.values()))},
                    parent_operator=self.operator_stack[-1] if self.operator_stack else None,
                    mode=None, status='INFLIGHT', accepted=[], operator_sequences=[])
                row['parent_source_sequence'] = self.stack[-1]['sequence'] if self.stack else None
                self.rows.append(row)
                self.pending[id(frame)] = row
                self.stack.append(row)
                if row['parent_operator'] is not None:
                    row['parent_operator']['actual_boundaries'].append(row)
                if binding['role'] in ('mla_opaque', 'moe_opaque'):
                    row['mode'] = self._mode()
                if binding['role'] == 'embedding_origin':
                    self._embedding_start(row)
                if self.moe is not None and binding['role'].startswith('moe_'):
                    self.moe.called(row, frame)
            else:
                row = self.pending.get(id(frame))
                require(row is not None and self.stack[-1] is row, 'QUANT_ROW_PROFILE_RETURN_ORDER_CHANGED')
                self._returned(row, frame.f_locals, result)
                row['status'] = 'OBSERVED'
        except BaseException as error:
            self.errors.append(type(error).__name__ + ':' + str(error))
            if row is not None:
                row['status'] = 'REFUSED'
                row['reason'] = str(error)
        finally:
            if event == 'return' and id(frame) in self.pending:
                row = self.pending.pop(id(frame))
                if row.get('mode') is not None:
                    try:
                        self._release_mode(row['mode'])
                    except BaseException as error:
                        self.errors.append(type(error).__name__ + ':' + str(error))
                    # The independent ledger retains the owned handle even
                    # when this Python frame has completed and been removed.
                if self.stack and self.stack[-1] is row:
                    self.stack.pop()
                # Tensor references only live through their actual source call.
                row.pop('input', None)
                row.pop('snapshots', None)
                row.pop('mode', None)
                row.pop('parent_operator', None)
                row.pop('embedding_state', None)

    def _returned(self, row, values, result):
        original = row['input']
        if self.moe is not None and row['role'].startswith('moe_'):
            self.moe.returned(row, values, result)
        elif row['role'].startswith('embedding_'):
            self._embedding_return(row, result)
        elif row['role'] in ('no_gather', 'no_chunk'):
            name = 'x' if row['role'] == 'no_gather' else 'residual'
            source = original[name]
            proof = self.graph._lookup(source)
            allowed = (original.get('label') is False if row['role'] == 'no_gather'
                       else original['x'].shape[0] == source.shape[0])
            if allowed and result is source and proof is not None and metadata(source) == row['snapshots'][id(source)]:
                self._bind(result, proof, 'PINNED_NO_COMMUNICATION_SAME_OBJECT', row,
                           axis=proof['node']['row_axis'])
                if row['parent_operator'] is not None:
                    row['parent_operator']['certified'].add(id(result))
        elif row['role'] == 'mla_opaque':
            source = original['hidden_states']
            require(metadata(source) == row['snapshots'][id(source)], 'MLA_PUBLIC_INPUT_CHANGED')
            owner = values.get('self')
            context = values.get('forward_context')
            require(owner is not None and id(owner) in self.model_paths and context is not None
                    and context.no_compile_layers[original['layer_name']] is owner
                    and any(values.get('kv_cache') is cache for cache in owner.mla_attn.kv_cache),
                    'MLA_CURRENT_MODEL_OR_KV_OWNER_CHANGED')

    def _bind(self, output, proof, operator, row, *, axis=0, extra=None):
        node = self.graph._bind(output, rows=proof['rows'], parents=(proof['node']['node_id'],), row_axis=axis,
            operator=operator, evidence=dict(source_boundary=row['sequence'], source_sha256=row['source_sha256'],
                actual_model_path=row['model_path'], simulation=self.simulation, details=extra))
        row['accepted'].append(node['node_id'])
        return id(output)

    def _embedding_start(self, row):
        self.embedding_attempts += 1
        state = row['embedding_state'] = dict(errors=[], operators=[], returns=[], indices=None, output=None)
        try:
            owner, ids = row['input']['self'], row['input']['input_']
            require(self.embedding_attempts == 1 and ids is self.input_ids
                and self.input_ids_metadata is not None and metadata(ids) == self.input_ids_metadata
                and ids.ndim == 1 and ids.dtype in (torch.int32, torch.int64)
                and ids.shape[0] == len(self.scope['request_ids']) == 8
                and self.graph.epochs.get(self.graph.storage_key(self.input_ids_metadata), 0) == 0,
                'CURRENT_SCHEDULER_EMBEDDING_INPUT_OBJECT_REQUIRED')
            group = owner.comm_group
            require(owner.tp_size == group.world_size == 1 and owner.tp_rank == group.rank_in_group == 0
                and owner.forward_type is None, 'ACTUAL_TP1_ORIGIN_EMBEDDING_ONLY')
            weight = owner.weight
            record = self.graph.parameters.get(id(weight))
            context = self.graph._inference_parameter_context(record, weight) if record is not None else None
            require(record is not None and record['owner']() is owner and record['tensor']() is weight
                and record['name'] == 'weight' and owner._parameters.get('weight') is weight
                and (context is None or context['current'])
                and metadata(weight) == record['metadata'] and weight.ndim == 2
                and self.graph.epochs.get(self.graph.storage_key(record['metadata']), 0) == record['write_epoch'],
                'ACTUAL_CURRENT_EMBEDDING_PARAMETER_REQUIRED')
            state.update(owner=owner, group=group, weight=weight, weight_record=record,
                indices=ids if ids.dtype == torch.int64 else None,
                indices_metadata=metadata(ids) if ids.dtype == torch.int64 else None)
        except BaseException as error:
            state['errors'].append(str(error))

    def _embedding_operator(self, state, output):
        row = self._parent('embedding_origin')
        capture = row.get('embedding_state') if row is not None else None
        if capture is None or capture['errors']:
            return
        try:
            func, args, kwargs = state['func'], state['args'], state['kwargs']
            name = self.operator_aliases.get(func, str(func))
            bound = _arguments(func, args, kwargs)
            require(not func._schema.is_mutable, 'EMBEDDING_PRODUCER_MUTATING_OPERATOR')
            if func is torch.ops.aten.to.dtype:
                require(not capture['operators'] and bound['self'] is self.input_ids
                    and set(bound) == {'self', 'dtype', 'non_blocking', 'copy', 'memory_format'}
                    and bound['dtype'] == torch.int64 and bound['non_blocking'] is False
                    and bound['copy'] is False and bound['memory_format'] is None
                    and isinstance(output, torch.Tensor) and output.dtype == torch.int64
                    and output.shape == self.input_ids.shape and output.device == self.input_ids.device
                    and output.is_contiguous() and metadata(self.input_ids) == self.input_ids_metadata,
                    'ACTUAL_DEFAULT_INT64_INDEX_CONVERSION_REQUIRED')
                capture.update(indices=output, indices_metadata=metadata(output))
                scalar_args = dict(dtype=str(bound['dtype']), non_blocking=bound['non_blocking'],
                    copy=bound['copy'], memory_format=bound['memory_format'])
            elif func is torch.ops.aten.embedding.default:
                lookup = self._parent('embedding_lookup')
                require(capture['output'] is None and lookup is not None
                    and lookup['input']['self'] is capture['owner'].quant_method
                    and lookup['input']['layer'] is capture['owner']
                    and lookup['input']['input_'] is capture['indices']
                    and set(bound) == {'weight', 'indices', 'padding_idx', 'scale_grad_by_freq', 'sparse'}
                    and bound['weight'] is capture['weight'] and bound['indices'] is capture['indices']
                    and bound['padding_idx'] == -1 and bound['scale_grad_by_freq'] is False and bound['sparse'] is False
                    and metadata(capture['indices']) == capture['indices_metadata']
                    and self.graph.epochs.get(self.graph.storage_key(capture['indices_metadata']), 0) == 0
                    and isinstance(output, torch.Tensor) and tuple(output.shape) == (8, capture['weight'].shape[1])
                    and output.dtype == capture['weight'].dtype and output.device == capture['weight'].device
                    and _disjoint(output, capture['weight']) and _disjoint(output, capture['indices']),
                    'ACTUAL_OWNED_EMBEDDING_INDEX_LOOKUP_REQUIRED')
                capture.update(output=output, output_metadata=metadata(output))
                scalar_args = {key: bound[key] for key in ('padding_idx', 'scale_grad_by_freq', 'sparse')}
            elif name == 'vllm.maybe_pad_and_reduce.default':
                require(set(bound) == {'x', 'is_ep_comm'} and bound['x'] is capture['output']
                    and bound['is_ep_comm'] is False and output is capture['output']
                    and metadata(output) == capture['output_metadata'], 'ACTUAL_SAME_OBJECT_EMBEDDING_REDUCTION_REQUIRED')
                scalar_args = dict(is_ep_comm=bound['is_ep_comm'])
            else:
                raise RuntimeError('UNREVIEWED_EMBEDDING_PRODUCER_OPERATOR:' + str(func))
            capture['operators'].append(dict(sequence=state['event']['sequence'], operator=str(func),
                canonical_operator=name,
                schema=str(func._schema), actual_scalar_args=scalar_args,
                inputs=state['event']['inputs'], outputs=[metadata(t) for t in tensor_values(output)]))
        except BaseException as error:
            capture['errors'].append(str(error))

    def _embedding_return(self, row, result):
        origin = self._parent('embedding_origin')
        capture = origin.get('embedding_state') if origin is not None else None
        if capture is None:
            return
        try:
            require(not capture['errors'], 'EMBEDDING_PRODUCER_CHAIN_REFUSED')
            role, original = row['role'], row['input']
            owner, group = capture['owner'], capture['group']
            record = dict(sequence=row['sequence'], role=role, parent_source_sequence=row['parent_source_sequence'],
                source_sha256=row['source_sha256'], function=row['function'])
            if role == 'embedding_get_tp':
                require(result is group and group.world_size == 1 and group.rank_in_group == 0,
                    'ACTUAL_TP_GROUP_OBJECT_REQUIRED')
                record.update(group_object_id=id(group), world_size=group.world_size, rank=group.rank_in_group)
            elif role == 'embedding_comm_return':
                require(original['self'] is group and group.world_size == 1 and group.rank_in_group == 0
                    and original['input_'] is capture['output'] and result is original['input_']
                    and not row['operator_sequences'] and metadata(result) == row['snapshots'][id(result)],
                    'ACTUAL_TP1_ALL_REDUCE_EARLY_RETURN_REQUIRED')
                record.update(group_object_id=id(group), world_size=group.world_size, rank=group.rank_in_group,
                    input=metadata(result), returned_same_object=True, observed_child_operators=[])
            elif role in ('embedding_tp_reduce', 'embedding_pad', 'embedding_lookup'):
                name = 'x' if role == 'embedding_pad' else 'input_'
                expected = capture['indices'] if role == 'embedding_lookup' else capture['output']
                require(original[name] is expected and result is capture['output']
                    and metadata(expected) == row['snapshots'][id(expected)]
                    and metadata(result) == capture['output_metadata'], 'EMBEDDING_SOURCE_RETURN_OBJECT_CHANGED')
                record.update(input=metadata(expected), output=metadata(result), returned_same_object=result is expected)
            elif role == 'embedding_origin':
                events = capture['operators']
                context = self.graph._inference_parameter_context(capture['weight_record'], capture['weight'])
                names = [item['canonical_operator'] for item in events]
                expected = ['aten.embedding.default', 'vllm.maybe_pad_and_reduce.default']
                require(names in (expected, ['aten.to.dtype'] + expected)
                    and result is capture['output'] and metadata(result) == capture['output_metadata']
                    and metadata(self.input_ids) == self.input_ids_metadata
                    and self.graph.epochs.get(self.graph.storage_key(self.input_ids_metadata), 0) == 0
                    and owner.weight is capture['weight'] and metadata(owner.weight) == capture['weight_record']['metadata']
                    and (context is None or context['current'])
                    and self.graph.epochs.get(self.graph.storage_key(capture['weight_record']['metadata']), 0)
                        == capture['weight_record']['write_epoch']
                    and owner.comm_group is group and owner.tp_size == group.world_size == 1
                    and owner.tp_rank == group.rank_in_group == 0 and owner.forward_type is None,
                    'COMPLETE_CURRENT_EMBEDDING_PRODUCER_REQUIRED')
                returns = capture['returns']
                def one(wanted):
                    found = [item for item in returns if item['role'] == wanted]
                    require(len(found) == 1, 'EXACT_EMBEDDING_SOURCE_CHILD_REQUIRED:' + wanted)
                    return found[0]
                lookup, pad, tp, get, comm = [one(wanted) for wanted in ('embedding_lookup', 'embedding_pad',
                    'embedding_tp_reduce', 'embedding_get_tp', 'embedding_comm_return')]
                require(lookup['parent_source_sequence'] == pad['parent_source_sequence'] == row['sequence']
                    and tp['parent_source_sequence'] == pad['sequence']
                    and get['parent_source_sequence'] == comm['parent_source_sequence'] == tp['sequence'],
                    'EMBEDDING_SOURCE_CHILD_CALL_ORDER_CHANGED')
                evidence = dict(verified=True, source='ACTUAL_TP1_SCHEDULER_IDS_EMBEDDING_PRODUCER',
                    actual_rows=self.scope, forward_id=self.graph.forward_id, model_path=row['model_path'],
                    model_module_object_id=id(owner), input_ids=metadata(self.input_ids), weight=metadata(owner.weight),
                    source_boundary=row['sequence'], source_sha256=row['source_sha256'],
                    actual_operators=events, source_returns=returns, simulation=self.simulation)
                if context is not None:
                    evidence.update(weight_inference_registration=capture['weight_record']['inference_registration'],
                        weight_inference_context=context)
                self.graph.seed(result, evidence, operator='ACTUAL_SCHEDULER_IDS_TO_TP1_EMBEDDING')
                row['accepted'].append(self.graph._lookup(result)['node']['node_id'])
                row['embedding_producer'] = snapshot(evidence)
                return
            capture['returns'].append(record)
        except BaseException as error:
            capture['errors'].append(str(error))
            row['row_refusal'] = list(capture['errors'])

    def _row_diagnostic(self, tensor):
        """Describe the existing lookup predicate without deleting stale rows."""
        if not isinstance(tensor, torch.Tensor):
            return dict(current=False, tensor=False, reason='NOT_A_TENSOR')
        actual = metadata(tensor)
        record = self.graph.proofs.get(id(tensor))
        result = dict(tensor=True, actual_metadata=actual, record_present=record is not None,
            actual_write_epoch=self.graph.epochs.get(self.graph.storage_key(actual), 0))
        checks = dict(record_present=record is not None)
        if record is not None:
            checks.update(same_tensor_object=record['tensor']() is tensor,
                rows_digest_matches=digest(record['rows']) == record['node']['rows_sha256'],
                node_digest_matches=digest(record['node']) == record['node_sha256'],
                metadata_matches=actual == record['metadata'],
                write_epoch_matches=result['actual_write_epoch'] == record['write_epoch'])
            result.update(recorded_metadata=record['metadata'], recorded_write_epoch=record['write_epoch'],
                metadata_changed_fields=[key for key in actual if actual[key] != record['metadata'].get(key)],
                node_id=record['node']['node_id'], row_axis=record['node']['row_axis'],
                rows_sha256=record['node']['rows_sha256'])
        result.update(checks=checks, current=all(checks.values()),
            failures=[name for name, ok in checks.items() if not ok])
        return snapshot(result)

    def _parameter_diagnostic(self, tensor, output_shape):
        """Read the frozen-parameter predicate; never register a missed owner."""
        if not isinstance(tensor, torch.Tensor):
            return dict(broadcast_proven=False, tensor=False, reason='NOT_A_TENSOR')
        actual = metadata(tensor)
        record = self.graph.parameters.get(id(tensor))
        result = dict(tensor=True, actual_metadata=actual, requires_grad=bool(tensor.requires_grad),
            parameter_type=type(tensor).__module__ + '.' + type(tensor).__name__,
            registered=record is not None,
            actual_write_epoch=self.graph.epochs.get(self.graph.storage_key(actual), 0))
        result['observed_model_parameter_owners'] = [dict(module_object_id=id(owner), model_path=path or '<root>',
            parameter_name=name) for path, owner in self.model.named_modules()
            for name, value in owner._parameters.items() if value is tensor]
        checks = dict(registered=record is not None)
        if record is not None:
            owner = record['owner']()
            checks.update(same_tensor_object=record['tensor']() is tensor, owner_alive=owner is not None,
                owner_parameter_same_object=owner is not None and
                    getattr(owner, record['kind']).get(record['name']) is tensor,
                metadata_matches=actual == record['metadata'],
                write_epoch_matches=result['actual_write_epoch'] == record['write_epoch'],
                no_request_dimension=tensor.ndim < len(output_shape) or
                    tensor.ndim == len(output_shape) and tensor.shape[0] == 1)
            result.update(owner_object_id=id(owner) if owner is not None else None,
                owner_model_path=self.model_paths.get(id(owner)), parameter_name=record['name'], kind=record['kind'],
                recorded_metadata=record['metadata'], recorded_write_epoch=record['write_epoch'],
                metadata_changed_fields=[key for key in actual if actual[key] != record['metadata'].get(key)])
            context = self.graph._inference_parameter_context(record, tensor)
            if context is not None:
                checks['inference_context_current'] = context['current']
                result.update(inference_registration=record['inference_registration'], inference_context=context)
        result.update(checks=checks, broadcast_proven=all(checks.values()),
            failures=[name for name, ok in checks.items() if not ok])
        return snapshot(result)

    def _add_rms_diagnostic(self, state, output=None, *, phase):
        """Bounded event metadata only; the existing admission rule is unchanged."""
        try:
            func, args, kwargs = state['func'], state['args'], state['kwargs']
            bound = _arguments(func, args, kwargs)
            inputs, outputs = tensor_values((args, kwargs)), tensor_values(output)
            first = inputs[0] if inputs else None
            shape = tuple(first.shape) if first is not None else ()
            left, right = self._row_diagnostic(first), self._row_diagnostic(bound.get('x2'))
            gamma = self._parameter_diagnostic(bound.get('gamma'), shape)
            beta = self._parameter_diagnostic(bound.get('beta'), shape) if bound.get('beta') is not None else None
            # Compute the identical existing gates. This deliberately does not
            # call _lookup (which evicts stale records), bind or mutate epochs.
            lrecord = self.graph.proofs.get(id(first))
            rrecord = self.graph.proofs.get(id(bound.get('x2')))
            checks = dict(has_input=bool(inputs), readonly_schema=not func._schema.is_mutable,
                first_row_proof_current=left['current'], first_row_axis_zero=left.get('row_axis') == 0,
                argument_names_exact=set(bound) == {'x1', 'x2', 'gamma', 'beta', 'epsilon'},
                second_row_proof_current=right['current'], second_row_axis_zero=right.get('row_axis') == 0,
                actual_request_rows_equal=bool(lrecord is not None and rrecord is not None and
                    lrecord['rows'] == rrecord['rows']),
                input_shapes_equal=isinstance(bound.get('x2'), torch.Tensor) and tuple(bound['x2'].shape) == shape,
                gamma_broadcast_proven=gamma['broadcast_proven'],
                beta_absent_or_broadcast_proven=beta is None or beta['broadcast_proven'])
            if phase == 'after':
                checks.update(output_count_three=len(outputs) == 3,
                    output_shapes_match=len(outputs) == 3 and tuple(outputs[0].shape) == tuple(outputs[2].shape) == shape)
            state['event'].setdefault('add_rms_row_diagnostic', {})[phase] = snapshot(dict(
                schema='demoncat.quant.add_rms_row_diagnostic.v1', actual_operator=str(func),
                actual_schema=str(func._schema), argument_names=list(bound), observer_id=id(self.graph),
                forward_id=self.graph.forward_id, actual_dispatch_sequence=state['event']['sequence'],
                first_input=left, second_input=right, gamma=gamma, beta=beta,
                outputs=[metadata(tensor) for tensor in outputs], checks=checks,
                failures=[name for name, ok in checks.items() if not ok], complete=True))
        except BaseException as error:
            state['event'].setdefault('add_rms_row_diagnostic', {})[phase] = dict(
                complete=False, diagnostic_error=type(error).__name__ + ':' + str(error))

    def before_operator(self, func, args, kwargs, event):
        try:
            return self._before_operator(func, args, kwargs, event)
        except BaseException as error:
            self.errors.append('SOURCE_ROW_PRE_OBSERVER:' + repr(error))
            # The model still runs once; return the owned state so the normal
            # unwind can close it even when a source/schema check refused.
            if self.operator_stack and self.operator_stack[-1]['event'] is event:
                return self.operator_stack[-1]
            raise

    def _before_operator(self, func, args, kwargs, event):
        state = dict(func=func, args=args, kwargs=kwargs, event=event, written=None, certified=set(), mutation=None,
                     actual_boundaries=[], expected_role=None)
        self.operator_stack.append(state)
        for row in self.stack:
            row['operator_sequences'].append(event['sequence'])
        if not self.active or self.errors:
            return state
        schema = getattr(func, '_schema', None)
        if schema is None:
            return state
        name = self.operator_aliases.get(func, str(func))
        if name == '_C_ascend.npu_add_rms_norm_bias.default':
            self._add_rms_diagnostic(state, phase='before')
        expected = {'vllm.mla_forward.default': 'mla_opaque',
            'vllm.maybe_all_gather_and_maybe_unpad.default': 'no_gather',
            'vllm.maybe_chunk_residual.default': 'no_chunk'}.get(name)
        if name == 'vllm.moe_forward_shared.default' and self.moe is not None:
            expected = 'moe_opaque'
        if name == 'vllm.maybe_pad_and_reduce.default' and self._parent('embedding_origin') is not None:
            expected = 'embedding_pad'
        if expected is not None:
            self._verify_bindings()
            state['expected_role'] = expected
        if expected == 'moe_opaque':
            self.moe.before_operator(state)
        if name == 'vllm.mla_forward.default':
            bound = _arguments(func, args, kwargs)
            writes = [arg.name for arg in schema.arguments if arg.alias_info is not None and arg.alias_info.is_write]
            if list(bound) == ['hidden_states', 'need_gather_q_kv', 'output', 'layer_name'] and writes == ['output']:
                state['written'] = [bound['output']]
        elif name in ('aten.copy_.default', 'aten.add_.Tensor', 'aten.mul_.Tensor', 'aten.mul_.Scalar', 'aten.add_.Scalar'):
            inputs = tensor_values((args, kwargs))
            target = args[0]
            source = args[1] if len(args) > 1 else kwargs.get('other', kwargs.get('src'))
            source_proof = self.graph._lookup(source) if isinstance(source, torch.Tensor) else None
            prior = self.graph._lookup(target)
            if name == 'aten.copy_.default':
                good = (source_proof is not None and target.shape == source.shape and target.dtype == source.dtype
                    and target.device == source.device and _disjoint(target, source))
                proof = source_proof
            else:
                good = prior is not None and (not isinstance(source, torch.Tensor) or source.ndim == 0
                    or self.graph._broadcast_parameter(source, tuple(target.shape))
                    or source_proof is not None and source_proof['rows'] == prior['rows']
                       and source_proof['node']['row_axis'] == prior['node']['row_axis'] and source.shape == target.shape)
                proof = prior
            if good and all(t is target or _disjoint(t, target)
                            for t in inputs[1:]):
                state.update(written=[target], mutation=dict(target=target, source=source, proof=proof,
                    target_meta=metadata(target), source_meta=metadata(source) if isinstance(source, torch.Tensor) else None))
        return state

    def failed_operator(self, state, error):
        if state is not None and 'moe' in state and self.moe is not None:
            self.moe.after_operator(state)
        if state is not None and self.operator_stack and self.operator_stack[-1] is state:
            self.operator_stack.pop()
        self.errors.append('ACTUAL_OPERATOR_FAILED:' + repr(error))

    def after_operator(self, state, output):
        if state is None:
            return set()
        require(self.operator_stack and self.operator_stack[-1] is state, 'QUANT_ROW_OPERATOR_RETURN_ORDER_CHANGED')
        self.operator_stack.pop()
        if 'moe' in state and self.moe is not None:
            self.moe.after_operator(state)
        if not self.active or self.errors:
            return set()
        try:
            if state['expected_role'] is not None:
                actual = [row for row in state['actual_boundaries'] if row['role'] == state['expected_role']]
                if len(actual) != 1 or actual[0]['status'] != 'OBSERVED':
                    self.errors.append('OPAQUE_OPERATOR_HAS_NO_COMPLETE_REVIEWED_PYTHON_CALL')
                    return set()
            # Inference-mode views deliberately have no _base. Keep the actual
            # observed, exact-extent view edge instead of inventing one later.
            if (str(state['func']) in ('aten.slice.Tensor', 'aten.alias.default', 'aten.detach.default',
                                      'aten.view.default', 'aten._unsafe_view.default')
                    and isinstance(output, torch.Tensor) and state['args']
                    and isinstance(state['args'][0], torch.Tensor) and _same_extent(output, state['args'][0])):
                self.alias_parents[id(output)] = (weakref.ref(output), weakref.ref(state['args'][0]))
            accepted = set(state['certified'])
            self._embedding_operator(state, output)
            add_rms = self.operator_aliases.get(state['func'], str(state['func'])) == '_C_ascend.npu_add_rms_norm_bias.default'
            if add_rms:
                self._add_rms_diagnostic(state, output, phase='after')
            accepted.update(self._operator_rule(state, output))
            if add_rms:
                state['event']['add_rms_row_diagnostic']['actual_rule_accepted_outputs'] = len(accepted)
            return accepted
        except BaseException as error:
            state['event']['boundary_refusal'] = str(error)
            return set()

    def _operator_rule(self, state, output):
        func, args, kwargs = state['func'], state['args'], state['kwargs']
        name = self.operator_aliases.get(func, str(func))
        outputs = tensor_values(output)
        inputs = tensor_values((args, kwargs))
        row = self.stack[-1] if self.stack else dict(sequence=0, source_sha256=None,
            model_path='<actual_operator>', accepted=[])
        mutation = state['mutation']
        if mutation is not None:
            target, source, proof = (mutation[key] for key in ('target', 'source', 'proof'))
            before, after = mutation['target_meta'], metadata(target)
            require(output is target and all(before[k] == after[k] for k in before if k != 'version'),
                    'ROW_MUTATION_TARGET_GEOMETRY_CHANGED')
            require(not isinstance(source, torch.Tensor) or metadata(source) == mutation['source_meta'],
                    'ROW_MUTATION_SECOND_OPERAND_CHANGED')
            accepted = {self._bind(target, proof, name, row, axis=proof['node']['row_axis'],
                extra=dict(actual_dispatch_sequence=state['event']['sequence'], exact_written_storage=True))}
            base = getattr(target, '_base', None)
            edge = self.alias_parents.get(id(target))
            if base is None and edge is not None and edge[0]() is target:
                base = edge[1]()
            if isinstance(base, torch.Tensor) and _same_extent(base, target):
                accepted.add(self._bind(base, proof, name + ':actual_full_extent_base', row, axis=proof['node']['row_axis']))
            return accepted
        if not inputs or not outputs or getattr(func._schema, 'is_mutable', True):
            return set()
        first = inputs[0]
        proof = self.graph._lookup(first)
        if proof is None:
            return set()
        axis = proof['node']['row_axis']
        # Axis changes are only enabled within reviewed MLA implementation code.
        mla = any(item['role'].startswith('mla_') for item in self.stack)
        if mla and len(outputs) == 1:
            target = outputs[0]
            if name == 'aten.transpose.int':
                a, b = args[1] % first.ndim, args[2] % first.ndim
                new_axis = b if axis == a else a if axis == b else axis
                return {self._bind(target, proof, name, row, axis=new_axis)}
            if name == 'aten.permute.default':
                perm = [int(i) % first.ndim for i in args[1]]
                return {self._bind(target, proof, name, row, axis=perm.index(axis))}
            if name in ('aten.view.default', 'aten._unsafe_view.default', 'aten.reshape.default'):
                axes = [i for i, size in enumerate(target.shape) if size == first.shape[axis]
                    and math.prod(target.shape[:i]) == math.prod(first.shape[:axis])
                    and math.prod(target.shape[i+1:]) == math.prod(first.shape[axis+1:])]
                if len(axes) == 1 and first.is_contiguous() and target.is_contiguous():
                    return {self._bind(target, proof, name, row, axis=axes[0])}
                # Only the real readonly aten view overload, under an active
                # source-bound MLA frame, can use the noncontiguous bridge.
                # Object identity is supplied by this observed call; storage
                # and exact non-singleton strides prove the coordinate map.
                if func is torch.ops.aten.view.default and state['event']['inputs'][0] == metadata(first):
                    after = metadata(target)
                    new_axis = _singleton_view_axis(proof['metadata'], after, axis)
                    if new_axis is not None:
                        return {self._bind(target, proof, name, row, axis=new_axis,
                            extra=dict(actual_dispatch_sequence=state['event']['sequence'],
                                view_rule='ACTUAL_ATEN_VIEW_SINGLETON_COORDINATES_V1',
                                input_metadata=proof['metadata'], output_metadata=after,
                                input_write_epoch=proof['write_epoch'],
                                input_row_axis=axis, output_row_axis=new_axis))}
            if name in ('aten.clone.default', 'aten.detach.default', 'aten.alias.default', 'aten._to_copy.default'):
                if target.shape == first.shape:
                    return {self._bind(target, proof, name, row, axis=axis)}
            if func is torch.ops.aten.contiguous.default:
                bound = _arguments(func, args, kwargs)
                if (set(bound) == {'self', 'memory_format'} and bound['self'] is first
                    and bound['memory_format'] in (0, torch.contiguous_format)
                    and target.shape == first.shape and target.dtype == first.dtype and target.device == first.device
                    and target.is_contiguous() and state['event']['inputs'][0] == metadata(first)
                    and (target is first or _disjoint(target, first))):
                    return {self._bind(target, proof, name, row, axis=axis,
                        extra=dict(actual_dispatch_sequence=state['event']['sequence'],
                            view_rule='ACTUAL_ATEN_CONTIGUOUS_DEFAULT_COORDINATES_V1',
                            input_metadata=proof['metadata'], output_metadata=metadata(target),
                            input_write_epoch=proof['write_epoch'], input_row_axis=axis, output_row_axis=axis))}
            qrow = self._parent('mla_q')
            if name == 'aten.bmm.default' and qrow is not None:
                weight = args[1]
                owner = qrow['input']['self']
                if axis == 1 and self._live_constant(owner, 'W_UK_T', weight):
                    return {self._bind(target, proof, name, qrow, axis=1,
                        extra=dict(actual_weight=metadata(weight), actual_dispatch_sequence=state['event']['sequence']))}
            vrow = self._parent('mla_v')
            if name == 'npu.npu_transpose_batchmatmul.default' and vrow is not None:
                bound = _arguments(func, args, kwargs)
                require(list(bound) == ['bias', 'scale', 'perm_x1', 'perm_x2', 'perm_y', 'batch_split_factor', 'input', 'weight']
                    or set(bound) == {'input', 'weight', 'bias', 'scale', 'perm_x1', 'perm_x2', 'perm_y', 'batch_split_factor'},
                    'TRANSPOSE_BMM_SCHEMA_CHANGED')
                weight = bound['weight']
                if (axis == 1 and self._live_constant(vrow['input']['self'], 'W_UV', weight)
                    and bound['bias'] is None and bound['scale'] is None and bound['perm_x1'] in (None, [0, 1, 2])
                    and bound['perm_x2'] in (None, [0, 1, 2]) and list(bound['perm_y']) == [1, 0, 2]
                    and bound['batch_split_factor'] == 1):
                    return {self._bind(target, proof, name, vrow, extra=dict(actual_weight=metadata(weight)))}
        if axis != 0:
            return set()
        if name == 'npu.npu_rms_norm.default':
            bound = _arguments(func, args, kwargs)
            if (set(bound) == {'input', 'gamma', 'epsilon'} and len(outputs) == 2
                and self.graph._broadcast_parameter(bound['gamma'], tuple(first.shape)) and outputs[0].shape == first.shape):
                return {self._bind(outputs[0], proof, name, row)}
        if name == '_C_ascend.npu_add_rms_norm_bias.default':
            bound = _arguments(func, args, kwargs)
            other = self.graph._lookup(bound.get('x2')) if isinstance(bound.get('x2'), torch.Tensor) else None
            if (set(bound) == {'x1', 'x2', 'gamma', 'beta', 'epsilon'} and other is not None
                and other['node']['row_axis'] == 0 and other['rows'] == proof['rows'] and len(outputs) == 3
                and bound['x2'].shape == first.shape and self.graph._broadcast_parameter(bound['gamma'], tuple(first.shape))
                and (bound['beta'] is None or self.graph._broadcast_parameter(bound['beta'], tuple(first.shape)))
                and outputs[0].shape == outputs[2].shape == first.shape):
                return {self._bind(outputs[i], proof, name, row, extra=dict(second_parent_node=other['node']['node_id'])) for i in (0, 2)}
        if name == 'npu.npu_swiglu.default':
            bound = _arguments(func, args, kwargs)
            if (set(bound) == {'self', 'dim'} or set(bound) == {'input', 'dim'}) and bound['dim'] % first.ndim != 0:
                if len(outputs) == 1 and outputs[0].shape[0] == first.shape[0]:
                    return {self._bind(outputs[0], proof, name, row)}
        rope = self._parent('mla_rope')
        if name == 'npu.npu_interleave_rope.default' and rope is not None:
            bound = _arguments(func, args, kwargs)
            preprocess = self._parent('mla_preprocess')
            if (set(bound) in ({'x', 'cos', 'sin'}, {'input', 'cos', 'sin'}) and len(outputs) == 1
                and bound['cos'] is rope['input']['cos'] and bound['sin'] is rope['input']['sin']
                and outputs[0].shape == first.shape and preprocess is not None
                and self.metadata_producers is not None):
                decode = preprocess['input']['attn_metadata'].decode
                require(decode.cos is bound['cos'] and decode.sin is bound['sin'], 'ACTUAL_ROTARY_METADATA_OPERANDS_REQUIRED')
                producer = self.metadata_producers.consumer(decode, self.graph)
                require(producer['actual_rows'] == proof['rows'], 'ROTARY_PRODUCER_REQUEST_ORDER_CHANGED')
                return {self._bind(outputs[0], proof, name, rope, extra=dict(actual_rotary_producer=producer))}
        decode = self._parent('mla_decode')
        if name == 'npu.npu_fused_infer_attention_score_v2.default' and decode is not None:
            return self._decode_attention(state, outputs, proof, decode)
        return set()

    def _decode_attention(self, state, outputs, proof, row):
        bound = _arguments(state['func'], state['args'], state['kwargs'])
        owner, attention = row['input']['self'], row['input']['attn_metadata']
        decode = attention.decode
        count = len(proof['rows']['request_ids'])
        require(self.scope is not None and proof['rows'] == self.scope and count == 8
                and attention.num_actual_tokens == attention.num_decode_tokens == attention.num_decodes == count
                and attention.num_prefills == 0 and attention.prefill is None and decode is not None,
                'COMPLETE_PURE_B8_MLA_DECODE_REQUIRED')
        require(not owner.fa_quant_layer and owner.speculative_config is None
                and row['input'].get('dequant_scale_q_nope') is None,
                'UNREVIEWED_QUANT_OR_SPECULATIVE_MLA')
        current = row['input']['self']
        context = row['input'].get('_extra_context') if self.simulation else self.bindings[
            type(current)._forward_decode.__code__]['function'].__globals__['_EXTRA_CTX']
        require(context is not None and not context.capturing and not context.is_draft_model,
                'EAGER_PRIMARY_MLA_ONLY')
        require(self.metadata_producers is not None, 'ACTUAL_EARLY_METADATA_PRODUCER_REQUIRED')
        producer = self.metadata_producers.consumer(decode, self.graph)
        require(producer['actual_rows'] == proof['rows'], 'MLA_METADATA_PRODUCER_REQUEST_ORDER_CHANGED')
        query = bound.get('query')
        query_rope = bound.get('query_rope')
        other = self.graph._lookup(query_rope) if isinstance(query_rope, torch.Tensor) else None
        require(query is not None and query is state['args'][0] and other is not None
                and other['rows'] == proof['rows'] and other['node']['row_axis'] == 0,
                'ACTUAL_MLA_QUERY_AND_ROPE_ROWS_REQUIRED')
        layout = 'BSND_NBSD' if owner.enable_kv_nz else 'BNSD_NBSD'
        require(bound.get('input_layout') == layout and bound.get('sparse_mode') == 0
                and bound.get('atten_mask') is None and bound.get('actual_seq_qlen') is None
                and bound.get('block_table') is decode.block_table
                and list(bound.get('actual_seq_kvlen') or ()) == list(decode.seq_lens_list)
                and list(decode.seq_lens_list) == [position + 1 for position in self.scope['row_token_positions']]
                and bound.get('block_size') == row['input']['block_size']
                and bound.get('num_query_heads') == owner.num_heads
                and bound.get('num_key_value_heads') == owner.num_kv_heads
                and bound.get('softmax_scale') == owner.scale,
                'MLA_CURRENT_QUERY_KV_LAYOUT_OR_SEQUENCE_BINDING_CHANGED')
        require(len(outputs) == 2 and tuple(outputs[0].shape) == (owner.num_heads, count, 1, owner.kv_lora_rank),
                'MLA_ACTUAL_NBSD_OUTPUT_REQUIRED')
        # Unexpected optional quantization, masks, offsets or compression are
        # not silently included in this reviewed nonquantized eager branch.
        known = {'query', 'key', 'value', 'query_rope', 'key_rope', 'num_query_heads', 'num_key_value_heads',
            'input_layout', 'atten_mask', 'sparse_mode', 'softmax_scale', 'block_table', 'block_size',
            'actual_seq_qlen', 'actual_seq_kvlen'}
        require(len(state['args']) == 3 and not set(state['kwargs']).difference(known), 'UNREVIEWED_MLA_OPTION')
        return {self._bind(outputs[0], proof, str(state['func']), row, axis=1,
            extra=dict(actual_dispatch_sequence=state['event']['sequence'], input_layout=layout,
                query=metadata(query), query_rope=metadata(query_rope), block_table=metadata(decode.block_table),
                actual_seq_kvlen=list(decode.seq_lens_list), actual_metadata_producer=producer))}

    def report(self):
        rows = [{key: value for key, value in row.items() if key not in
                 ('input', 'snapshots', 'mode', 'parent_operator')} for row in self.rows]
        return snapshot(dict(schema='demoncat.quant.source_bound_rows.v1', simulation=self.simulation,
            sources=self.sources, boundaries=rows, errors=self.errors, active=self.active,
            shared_moe_adapter=dict(enabled=self.moe is not None,
                frozen_model_owners=len(self.moe.owners) if self.moe is not None else 0,
                routed_output_row_qualified=False),
            inflight=len(self.pending), native_qualified=False,
            owned_nested_dispatch_modes=len(self.owned_modes), owned_profile_callbacks=int(not self.profile_released),
            owned_scheduler_input_refs=int(self.input_ids is not None),
            cleanup_pending=self.active and self.ending,
            complete=not self.active and not self.owned_modes and self.input_ids is None and not self.errors and not self.pending
                and all(row['status'] == 'OBSERVED' for row in rows)))
