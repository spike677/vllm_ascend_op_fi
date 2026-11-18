"""Observe the reviewed, serial shared-expert path inside the actual MoE call.

The public custom op declares hidden_states mutable. Its usual invalidation is
retained. A suspended input proof can resume only at the first reviewed runner
frame, before any child operator, after checking the actual dispatcher owner,
layer lookup and original objects. Every subsequent write still invalidates it.
Routed expert outputs and permutation indices receive no request-row proof.
"""
import ast
import hashlib
from pathlib import Path
import sys
import threading

import torch

from .quant_row_provenance import digest, metadata, snapshot
from .serving_context import require


RUNNER = 'vllm.model_executor.layers.fused_moe.runner.default_moe_runner'
ASCEND = 'vllm_ascend.ops.fused_moe.fused_moe'
REGISTER = 'vllm.utils.torch_utils'
SCHEMA = ('vllm::moe_forward_shared(Tensor(a0!) hidden_states, Tensor router_logits, '
          'Tensor? shared_experts_input, str layer_name) -> (Tensor, Tensor)')
REVIEWED = {
    RUNNER: ('e58081076104b325695cf5d95f0f3523eb67ea53dfc6d6fda6424ade664e9f79', {
        '_moe_forward_shared': 'moe_opaque', 'get_layer_from_name': 'moe_resolve',
        '_resolve_layer_name': 'moe_name', 'DefaultMoERunner.forward': 'moe_public_forward'}),
    ASCEND: ('bc170911dea7ff5c154ac966c9a046771f3afa79f0edd5e5d871525e911f25c0', {
        'AscendMoERunner.forward_impl': 'moe_runner',
        'AscendSharedFusedMoE.forward_impl': 'moe_layer',
        'AscendSharedFusedMoE._forward_shared_experts': 'moe_shared',
        'AscendSharedFusedMoE._shared_experts_part1': 'moe_part1',
        'AscendSharedFusedMoE._shared_experts_part2': 'moe_part2'}),
    REGISTER: ('b8c0a763d52c0eabb7b34c14b648b6f5e43f67d806f2819755b0ea23f4250221', {
        'direct_register_custom_op': 'moe_register'}),
}


def has_runtime_owner(model):
    # This selects which pinned sources to load, never grants row qualification.
    for _, module in model.named_modules():
        function = getattr(getattr(module, 'forward_impl', None), '__func__', None)
        if (getattr(function, '__module__', None) == ASCEND
                and getattr(function, '__qualname__', None) == 'AscendSharedFusedMoE.forward_impl'):
            return True
    return False


def _path_owner(model, path):
    for part in path.split('.') if path else ():
        model = getattr(model, '_modules', {}).get(part)
        if model is None:
            return None
    return model


def _disjoint(a, b):
    return a['device'] != b['device'] or (a['storage_address'] + a['storage_bytes'] <= b['storage_address']
        or b['storage_address'] + b['storage_bytes'] <= a['storage_address'])


class SharedMoERows:
    def __init__(self, boundary, *, simulation_library=None):
        self.boundary, self.graph = boundary, boundary.graph
        require(simulation_library is None or boundary.simulation, 'MOE_SIMULATION_LIBRARY_IS_CPU_ONLY')
        self.simulation_library = simulation_library
        self.owners = {}
        self.roles = {item['role']: item for item in boundary.bindings.values() if item['role'].startswith('moe_')}
        paths = {id(module): path for path, module in boundary.model.named_modules()}
        for path, layer in boundary.model.named_modules():
            method = getattr(getattr(layer, 'forward_impl', None), '__func__', None)
            if 'moe_layer' not in self.roles or method is not self.roles['moe_layer']['function']:
                continue
            runner, shared = getattr(layer, 'runner', None), getattr(layer, '_shared_experts', None)
            if runner is None or shared is None or id(shared) not in paths:
                continue
            self.owners[id(layer)] = dict(layer=layer, path=path, runner=runner, shared=shared,
                shared_path=paths[id(shared)], config=getattr(runner, 'moe_config', None))

    def _method(self, owner, name, role):
        item = self.roles.get(role)
        return item is not None and getattr(getattr(owner, name, None), '__func__', None) is item['function']

    def _owner(self, layer, runner):
        record = self.owners.get(id(layer))
        require(record is not None and record['layer'] is layer and record['runner'] is runner
            and _path_owner(self.boundary.model, record['path']) is layer
            and _path_owner(self.boundary.model, record['shared_path']) is record['shared']
            and layer.runner is runner and layer._modules.get('_shared_experts') is record['shared']
            and runner.shared_experts is record['shared'], 'MOE_ACTUAL_MODEL_RUNNER_SHARED_OWNER_CHANGED')
        for owner, name, role in ((runner, 'forward', 'moe_public_forward'), (runner, 'forward_impl', 'moe_runner'),
            (layer, 'forward_impl', 'moe_layer'), (layer, '_forward_shared_experts', 'moe_shared'),
            (layer, '_shared_experts_part1', 'moe_part1'), (layer, '_shared_experts_part2', 'moe_part2')):
            require(self._method(owner, name, role), 'MOE_ACTUAL_BOUND_METHOD_CHANGED:' + name)
        config = record['config']
        require(config is not None and runner.moe_config is config and layer.moe_config is config
            and type(layer.layer_name) is str and runner.layer_name == layer.layer_name
            and all(type(getattr(config, name, None)) is int and getattr(config, name) == 1
                    for name in ('tp_size', 'dp_size', 'ep_size', 'sp_size', 'pcp_size'))
            and all(type(getattr(config, name, None)) is int and getattr(config, name) == 0
                    for name in ('tp_rank', 'dp_rank', 'ep_rank', 'pcp_rank'))
            and config.is_sequence_parallel is False and runner.enable_dbo is False
            and runner.routed_input_transform is None and layer._routed_input_transform is None
            and layer.multistream_overlap_shared_expert is False and layer.multistream_overlap_gate is False
            and layer.shared_expert_stream is None and runner.shared_experts_stream is None
            and getattr(record['shared'], 'expert_gate', None) is None,
            'MOE_SERIAL_UNTRANSFORMED_TP1_SHARED_BRANCH_REQUIRED')
        require(not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
            and not self.boundary.model.training and not layer.training and not record['shared'].training,
            'MOE_ACTUAL_EVAL_INFERENCE_THREAD_REQUIRED')
        return record

    def _registration(self, func):
        """Check the current registration, including inactive overridden kernels."""
        fixture = self.simulation_library is not None
        if fixture:
            library, module, assignment, key = self.simulation_library, self.roles['moe_opaque']['module'], '_LIB', 'CPU'
            expected_schema = SCHEMA.replace('vllm::moe_forward_shared', library.ns + '::shared')
        else:
            require(str(torch.__version__) == '2.9.0+cpu', 'MOE_UNREVIEWED_DISPATCH_BUILD')
            module, assignment, key = sys.modules.get(REGISTER), 'vllm_lib', 'PrivateUse1'
            library = getattr(module, assignment, None)
            expected_schema = SCHEMA
            require(self.roles['moe_opaque']['module'].direct_register_custom_op
                    is self.roles['moe_register']['function'], 'MOE_REGISTER_FUNCTION_OWNER_CHANGED')
        require(isinstance(func, torch._ops.OpOverload) and str(func._schema) == expected_schema
            and isinstance(library, torch.library.Library), 'MOE_ACTUAL_SCHEMA_OR_LIBRARY_CHANGED')
        operator = str(func._schema.name)
        require(operator in library._op_defs and operator.replace('::', '/') + '/' + key in library._op_impls,
                'MOE_REGISTERED_LIBRARY_OWNERSHIP_CHANGED')
        path = Path(module.__file__).resolve()
        raw = path.read_bytes()
        expected = self.boundary.sources[module.__name__]['sha256']
        require(hashlib.sha256(raw).hexdigest() == expected, 'MOE_REGISTRATION_SOURCE_CHANGED')
        calls = [node.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == assignment for target in node.targets)]
        require(len(calls) == 1 and isinstance(calls[0], ast.Call)
            and len(calls[0].args) == 2 and all(isinstance(arg, ast.Constant) for arg in calls[0].args)
            and [arg.value for arg in calls[0].args] == [library.ns, 'DEF' if fixture else 'FRAGMENT']
            and not calls[0].keywords, 'MOE_LIBRARY_SOURCE_CONSTRUCTION_CHANGED')
        dump = torch._C._dispatch_dump(operator)
        lines = dump.splitlines()
        current = [line for line in lines if line.startswith(key)]
        definitions = [line for line in lines if line.startswith('debug: registered at ')]
        require(len(current) == len(definitions) == 1 and current[0].startswith(key + ': registered at ')
            and current[0].endswith(' :: (none) [ boxed ]') and 'alias analysis kind: FROM_SCHEMA' in lines
            and 'schema: ' + expected_schema in lines, 'MOE_CURRENT_KERNEL_SHADOWED_OR_UNPROVEN')
        origin = current[0].removeprefix(key + ': registered at ').removesuffix(' :: (none) [ boxed ]')
        filename, sep, line = origin.rpartition(':')
        require(definitions[0] == 'debug: registered at ' + origin and sep and line.isdecimal()
            and Path(filename).resolve() == path and calls[0].lineno <= int(line) <= calls[0].end_lineno,
            'MOE_CURRENT_KERNEL_ORIGIN_CHANGED')
        return dict(operator=operator, schema=expected_schema, torch_version=str(torch.__version__),
            source_sha256=expected, source_path=str(path), actual_dispatch_dump=dump,
            active_origin=origin, registration_key=key, simulation=fixture,
            guard='CURRENT_SINGLE_KERNEL_NO_SHADOW_EXACT_SOURCE_LIBRARY_ORIGIN')

    def before_operator(self, state):
        # Unknown/missing row proofs are retained as explicit unqualified calls.
        state['moe'] = dict(resumed=False, reason=None, resolved=None, shared_output=None,
                            input=None, proof=None, registration=None)
        row = state['moe']
        try:
            names = ('hidden_states', 'router_logits', 'shared_experts_input', 'layer_name')
            args, kwargs = state['args'], state['kwargs']
            require(len(args) <= 4 and not set(kwargs).difference(names)
                and not set(names[:len(args)]).intersection(kwargs), 'MOE_ACTUAL_ARGUMENT_BINDING_CHANGED')
            bound = dict(zip(names, args)); bound.update(kwargs)
            require(set(bound) == set(names), 'MOE_ALL_ACTUAL_ARGUMENTS_REQUIRED')
            x = bound['hidden_states']
            require(isinstance(x, torch.Tensor) and bound['shared_experts_input'] is x
                and isinstance(bound['router_logits'], torch.Tensor) and type(bound['layer_name']) is str
                and x.ndim == 2 and x.shape[0] == len(self.boundary.scope['request_ids']) == 8
                and _disjoint(metadata(x), metadata(bound['router_logits'])),
                'MOE_ACTUAL_SAME_SHARED_INPUT_REQUIRED')
            require(self.boundary.simulation or x.dtype == torch.bfloat16 and x.device.type == 'npu',
                    'MOE_REVIEWED_NPU_BF16_INPUT_REQUIRED')
            row.update(input=x, bound=bound, metadata=metadata(x), forward_id=self.graph.forward_id,
                       input_epoch=self.graph.epochs.get(self.graph.storage_key(metadata(x)), 0),
                       registration=self._registration(state['func']))
            proof = self.graph._lookup(x)
            require(proof is not None and proof['node']['row_axis'] == 0 and proof['rows'] == self.boundary.scope,
                    'MOE_CURRENT_REQUEST_INPUT_ROWS_UNPROVEN')
            row['proof'] = dict(rows=snapshot(proof['rows']), node=snapshot(proof['node']), node_sha256=proof['node_sha256'])
        except (RuntimeError, AttributeError, KeyError, TypeError) as error:
            self._refuse(state, error)
        # Never set state['written']: the parent mutable op still invalidates
        # every operand and every overlapping storage through the usual graph.

    def _refuse(self, state, error):
        row = state['moe']
        row['reason'] = row['reason'] or str(error)
        if row['resumed'] and 'revoked_nodes' not in row:
            # A later source/owner/return failure cannot leave the temporarily
            # resumed input, or a descendant alias, available to later calls.
            # Revoke proofs, not write epochs: this is not an observed write.
            revoked = {row['resumed_node']}
            for node in self.graph.nodes:
                if any(parent in revoked for parent in node['parents']):
                    revoked.add(node['node_id'])
            for key, proof in tuple(self.graph.proofs.items()):
                if proof['node']['node_id'] in revoked:
                    self.graph.proofs.pop(key)
            row['revoked_nodes'] = sorted(revoked)
        state['event']['shared_moe_row_diagnostic'] = self.diagnostic(state)

    def diagnostic(self, state):
        row = state['moe']
        return snapshot(dict(status='SHARED_INPUT_OBSERVED' if row['resumed'] else 'UNVERIFIED',
            reason=row['reason'], registration=row.get('registration'), input=row.get('metadata'),
            parent_sequence=state['event']['sequence'], parent_mutable_invalidation_preserved=True,
            resumed=row['resumed'], resumed_node=row.get('resumed_node'),
            revoked_proof_nodes=row.get('revoked_nodes', []),
            actual_owner=row.get('owner_facts'), shared_output=row.get('shared_output_metadata'),
            routed_output_row_proof=False, native_qualified=False))

    def state(self):
        return next((state for state in reversed(self.boundary.operator_stack) if 'moe' in state), None)

    def _current(self, state, *, require_input=True):
        row = state['moe']
        require(row['reason'] is None and row['resumed'] and self.boundary.active and not self.boundary.ending
            and self.graph.active and not self.graph.ending
            and not self.boundary.errors and not self.graph.errors and not self.graph.overflow
            and threading.get_ident() == self.boundary.thread and self.graph.forward_id == row['forward_id']
            and sys.getprofile() is self.boundary.callback, 'MOE_CURRENT_SOURCE_OBSERVER_REQUIRED')
        self.boundary._verify_bindings()
        self._owner(row['owner']['layer'], row['owner']['runner'])
        resolved = row['resolved']
        require(resolved['name'] == row['owner']['layer'].layer_name
            and resolved['context'].no_compile_layers.get(resolved['name']) is row['owner']['layer'],
                'MOE_CURRENT_FORWARD_CONTEXT_LAYER_OWNER_CHANGED')
        require(self._registration(state['func']) == row['registration'], 'MOE_CURRENT_REGISTRATION_CHANGED')
        if require_input:
            x, before = row['input'], row['metadata']
            require(metadata(x) == before
                and self.graph.epochs.get(self.graph.storage_key(before), 0) == row['input_epoch'] + 1,
                'MOE_INPUT_WRITTEN_OR_REPLACED_AFTER_PARENT_INVALIDATION')
            proof = self.graph._lookup(x)
            require(proof is not None and proof['rows'] == row['proof']['rows']
                and proof['node']['row_axis'] == 0, 'MOE_CURRENT_SHARED_INPUT_ROWS_LOST')
        return row

    def called(self, row, frame):
        state = self.state()
        if state is None:
            return
        record = state['moe']
        if record['reason'] is not None:
            return
        try:
            role, values = row['role'], row['input']
            if role == 'moe_opaque':
                require(row['parent_operator'] is state and all(values.get(name) is value
                    for name, value in record['bound'].items() if name != 'layer_name')
                    and values.get('layer_name') == record['bound']['layer_name'], 'MOE_OPAQUE_ARGUMENTS_CHANGED')
            elif role == 'moe_runner':
                resolved = record['resolved']
                require(not record['resumed'] and resolved is not None and resolved['layer'] is values.get('layer')
                    and self.graph.total == state['event']['sequence'], 'MOE_RESOLUTION_OR_PRE_RESUME_CHILD_CALL_CHANGED')
                owner = self._owner(values['layer'], values['self'])
                require(resolved['name'] == owner['layer'].layer_name
                    and resolved['context'].no_compile_layers.get(resolved['name']) is owner['layer'],
                    'MOE_RESOLVED_LAYER_NAME_OR_CONTEXT_CHANGED')
                bound = record['bound']
                require(values['hidden_states'] is bound['hidden_states'] and values['router_logits'] is bound['router_logits']
                    and values['shared_input'] is bound['shared_experts_input'], 'MOE_RUNNER_INPUT_OBJECT_CHANGED')
                x, before, proof = record['input'], record['metadata'], record['proof']
                require(metadata(x) == before and proof is not None and digest(proof['node']) == proof['node_sha256']
                    and digest(proof['rows']) == proof['node']['rows_sha256']
                    and self.graph.epochs.get(self.graph.storage_key(before), 0) == record['input_epoch'] + 1,
                    'MOE_INPUT_CHANGED_BEFORE_REVIEWED_RUNNER')
                require(self._registration(state['func']) == record['registration'], 'MOE_REGISTRATION_CHANGED_BEFORE_RESUME')
                node = self.graph._bind(x, rows=proof['rows'], parents=(proof['node']['node_id'],),
                    operator='ACTUAL_MOE_SHARED_INPUT_BEFORE_CHILD_OPERATIONS',
                    evidence=dict(parent_sequence=state['event']['sequence'], source_boundary=row['sequence'],
                        owner_path=owner['path'], source_sha256=row['source_sha256'],
                        previous_write_epoch=record['input_epoch'], declaration_invalidation_retained=True))
                record.update(resumed=True, owner=owner, resumed_node=node['node_id'],
                    owner_facts=dict(layer_path=owner['path'], layer_object_identity=id(owner['layer']),
                        runner_object_identity=id(owner['runner']), shared_path=owner['shared_path'],
                        shared_object_identity=id(owner['shared']), current_forward=self.graph.forward_id,
                        thread=threading.get_ident(), serial_tp1=True))
            elif role in ('moe_layer', 'moe_shared', 'moe_part1', 'moe_part2'):
                record = self._current(state)
                require(values.get('self') is record['owner']['layer']
                    and values.get('hidden_states') is record['input'], 'MOE_ACTUAL_SHARED_FRAME_INPUT_OR_OWNER_CHANGED')
                if role == 'moe_part2':
                    gate_up = values.get('shared_gate_up')
                    require(gate_up is record.get('gate_up') and isinstance(gate_up, torch.Tensor)
                        and self.graph._lookup(gate_up) is not None, 'MOE_SHARED_GATE_UP_ACTUAL_PRODUCER_REQUIRED')
        except (RuntimeError, AttributeError, KeyError, TypeError) as error:
            self._refuse(state, error)

    def returned(self, row, values, result):
        state = self.state()
        if state is None or state['moe']['reason'] is not None:
            return
        record = state['moe']
        try:
            role = row['role']
            if role == 'moe_resolve':
                context, name = values.get('forward_context'), values.get('layer_name')
                require(context is not None and type(name) is str
                    and context.no_compile_layers.get(name) is result and id(result) in self.owners,
                    'MOE_ACTUAL_FORWARD_CONTEXT_LAYER_LOOKUP_REQUIRED')
                require(record['resolved'] is None, 'MOE_DUPLICATE_LAYER_RESOLUTION')
                record['resolved'] = dict(layer=result, context=context, name=name)
            elif role in ('moe_part1', 'moe_part2', 'moe_shared'):
                record = self._current(state)
                proof = self.graph._lookup(result) if isinstance(result, torch.Tensor) else None
                require(proof is not None and proof['node']['row_axis'] == 0 and proof['rows'] == record['proof']['rows'],
                        'MOE_SHARED_RETURN_HAS_NO_COMPLETE_ACTUAL_OPERATOR_CHAIN')
                if role == 'moe_part1':
                    record['gate_up'] = result
                elif role == 'moe_shared':
                    record['shared_output'] = result
                    record['shared_output_metadata'] = metadata(result)
            elif role == 'moe_opaque':
                record = self._current(state)
                require(isinstance(result, tuple) and len(result) == 2 and all(isinstance(t, torch.Tensor) for t in result)
                    and result[0] is record['shared_output'] and self.graph._lookup(result[0]) is not None
                    and _disjoint(metadata(result[0]), metadata(result[1]))
                    and all(_disjoint(record['metadata'], metadata(t)) for t in result),
                    'MOE_SHARED_OUTPUT_MUST_BE_CURRENT_DISJOINT_ACTUAL_RETURN')
                state['certified'].add(id(result[0]))
        except (RuntimeError, AttributeError, KeyError, TypeError) as error:
            self._refuse(state, error)

    def permits(self, state):
        try:
            record = self._current(state)
            # Only actual shared projection frames can borrow this proof. The
            # routed/expert pipeline remains outside the request-row scope.
            shared = next((row for row in reversed(self.boundary.stack)
                if row['role'] in ('moe_part1', 'moe_part2')), None)
            return (shared is not None and shared['status'] == 'INFLIGHT'
                and shared['input'].get('self') is record['owner']['layer'])
        except (RuntimeError, AttributeError, KeyError, TypeError):
            return False

    def after_operator(self, state):
        state['event']['shared_moe_row_diagnostic'] = self.diagnostic(state)
        if state['moe']['reason'] is not None:
            state['certified'].clear()
        # Only the operator stack holds tensor references; an event/report is
        # a detached metadata snapshot. No suspended tensor survives return.
        state.pop('moe')
