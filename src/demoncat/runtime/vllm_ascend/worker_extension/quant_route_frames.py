"""Bounded Python routed-call evidence. Never binds rows or admits a fault."""
from collections import Counter
import dataclasses
import enum
import json
import math
import sys
import threading
import time
import types

import torch

from .quant_row_provenance import digest, metadata, snapshot
from .quant_token_routes import _formals
from .serving_context import require


MLP = 'vllm_ascend.ops.fused_moe.moe_mlp'
ARGS = 'vllm_ascend.ops.fused_moe.moe_runtime_args'
CONTRACTS = 'vllm_ascend.ops.fused_moe.moe_stage_contracts'
PARAMS = 'vllm_ascend.ops.fused_moe.moe_stage_params'
DEVICE = 'vllm_ascend.device.device_op'
REVIEWED = {
    'vllm.model_executor.models.deepseek_v2': (
        'bad4921f88f8343a965e333a399edf6b73f9d3031f6297e0dbf56afe37a31b01',
        {'DeepseekV2MoE.forward': 'rchain_model'}),
    MLP: ('1a65440a1357af84641a87fda630cf84efcbc8b34fa66ceedd2c403893fd840d', {
        'unified_apply_mlp': 'rchain_entry', 'quant_apply_mlp': 'rchain_quant',
        'cumsum_group_list': 'rchain_cumsum'}),
    ARGS: ('41bb24c3fe464e5fc99792ad86afdd2d44ad71a5992d106ee4209cf982bde3f5',
        {'build_mlp_compute_input': 'rchain_builder'}),
    CONTRACTS: ('f9ae3e9d5d55e7c172fb7d7a3f93856447514ebc9bf60c2213eb77805357cdd8', {}),
    PARAMS: ('77faab96fc89478528b18c12ba18b044d16c2d2e5bd16b0ecf0946373fcf2555', {
        'MoEQuantParams.is_quant': 'rchain_property', 'MoEQuantParams.is_mxfp': 'rchain_property',
        'MoEQuantParams.is_int_quant': 'rchain_property'}),
    DEVICE: ('fe5262b76b3c4841d735b6c06c0fe02035aeecc8562f81e78b07d0e691c6cd17', {
        'BaseDeviceAdaptor.npu_grouped_matmul_swiglu_quant': 'rchain_gmm1',
        'BaseDeviceAdaptor.npu_grouped_matmul_gmm2': 'rchain_gmm2'}),
    'vllm_ascend.ops.fused_moe.prepare_finalize': (
        'f46b378b2ba197fafd744c409735aa9a011d93e073507c1f87cf1a60ef2ecae4', {
            'PrepareAndFinalizeWithAllGather.prepare': 'rchain_prepare',
            'PrepareAndFinalizeWithAllGather.finalize': 'rchain_finalize'}),
}
TYPE_NAMES = ('MoEPrepareOutput', 'MoEWeights', 'MoEFusedExpertsInput', 'MoETokenDispatchInput',
              'MoETokenDispatchOutput', 'MoEAllGatherCombineMetadata', 'MoEMlpComputeInput')
REQUIRED = ('rchain_prepare', 'route_pipeline', 'route_dispatch', 'rchain_builder', 'route_mlp',
            'rchain_entry', 'rchain_quant', 'rchain_cumsum', 'rchain_gmm1', 'rchain_gmm2',
            'route_combine', 'rchain_finalize')
NATIVE_GATES = (
    'ACTUAL_ROUTING_GMM1_GMM2_UNPERMUTE_LOADED_IMPLEMENTATION_OWNERS_UNPROVEN',
    'GMM_CUMULATIVE_AND_COUNT_GROUP_ROW_SEMANTICS_UNPROVEN',
    'UNPERMUTE_INVERSE_INDEX_AND_PROBABILITY_REDUCTION_SEMANTICS_UNPROVEN',
)


class RouteFrameObserver:
    """32 model calls, 48 frame pairs and 512 KiB metadata per call; zero copies."""
    def __init__(self, boundary, *, type_bindings=None, max_parents=32, max_frames=48,
                 max_bytes=512 * 1024):
        self.boundary, self.graph = boundary, boundary.graph
        require(0 < max_parents <= 32 and 0 < max_frames <= 48 and 0 < max_bytes <= 512 * 1024,
                'ROUTED_FRAME_BUDGET_REQUIRED')
        self.max_parents, self.max_frames, self.max_bytes = max_parents, max_frames, max_bytes
        require(type_bindings is None or boundary.simulation, 'ROUTED_FRAME_TYPES_ARE_FIXTURES_ONLY')
        self.types = type_bindings or {
            name: getattr(sys.modules[CONTRACTS], name) for name in TYPE_NAMES}
        if type_bindings is None:
            self.types.update({name: getattr(sys.modules[PARAMS], name)
                               for name in ('MoEQuantParams', 'MoERoutingParams')})
        self.frozen_types = {}
        for name, cls in self.types.items():
            fields = tuple(field.name for field in dataclasses.fields(cls))
            require(dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
                and all(isinstance(vars(cls).get(field), types.MemberDescriptorType) for field in fields),
                'ROUTED_FROZEN_SLOTTED_PAYLOAD_REQUIRED:' + name)
            self.frozen_types[name] = (cls, fields)
        self.owners = {}
        model_role = next(item for item in boundary.bindings.values() if item['role'] == 'rchain_model')
        for path, owner in boundary.model.named_modules():
            if getattr(getattr(owner, 'forward', None), '__func__', None) is model_role['function']:
                scalar = getattr(owner, 'routed_scaling_factor', None)
                require(type(scalar) in (float, int) and math.isfinite(scalar), 'ROUTED_SCALAR_CONFIG_REQUIRED')
                self.owners[id(owner)] = (owner, path or '<root>', model_role['function'], scalar)
        self.reset()

    def reset(self):
        require(not getattr(self, 'live', None), 'ROUTED_FRAME_OWNER_STILL_LIVE')
        self.live = None
        self.records = []
        self.total_parents = self.overflow = self.callbacks = self.elapsed_ns = 0

    def _fail(self, ctx, error):
        ctx['record']['first_failure'] = ctx['record']['first_failure'] or str(error)

    def _current(self, ctx):
        b, g = self.boundary, self.graph
        require(b.active and not b.ending and g.active and not g.ending
            and threading.get_ident() == b.thread and sys.getprofile() is b.callback
            and ctx['record']['forward_id'] == g.forward_id and not b.errors and not g.errors and not g.overflow,
            'ROUTED_FRAME_CURRENT_OBSERVER_REQUIRED')
        require(torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            and not b.model.training and not ctx['owner'].training, 'ROUTED_FRAME_INFERENCE_REQUIRED')
        owner, path, function, scalar = self.owners[id(ctx['owner'])]
        require(getattr(owner.forward, '__func__', None) is function
            and dict(b.model.named_modules()).get('' if path == '<root>' else path) is owner
            and type(owner.routed_scaling_factor) is type(scalar) and owner.routed_scaling_factor == scalar,
            'ROUTED_FRAME_CURRENT_MODEL_OR_SCALAR_OWNER_CHANGED')
        b._verify_bindings()
        if not b.simulation:
            require(sys.modules[DEVICE].DeviceOperator is sys.modules[DEVICE].BaseDeviceAdaptor
                and sys.modules[MLP].DeviceOperator is sys.modules[DEVICE].BaseDeviceAdaptor,
                'ROUTED_FRAME_BASE_DEVICE_ADAPTOR_REQUIRED')
            for name, (cls, _) in self.frozen_types.items():
                module = PARAMS if name in ('MoEQuantParams', 'MoERoutingParams') else CONTRACTS
                require(getattr(sys.modules[module], name) is cls, 'ROUTED_FRAME_PAYLOAD_CLASS_CHANGED:' + name)

    def _typed(self, value, name):
        cls, fields = self.frozen_types[name]
        require(type(value) is cls and tuple(f.name for f in dataclasses.fields(cls)) == fields
            and cls.__dataclass_params__.frozen
            and all(isinstance(vars(cls).get(field), types.MemberDescriptorType) for field in fields),
            'ROUTED_FRAME_ACTUAL_PAYLOAD_TYPE_CHANGED:' + name)
        return value

    def _tensor(self, value):
        require(isinstance(value, torch.Tensor), 'ROUTED_FRAME_ACTUAL_TENSOR_REQUIRED')
        meta = metadata(value)
        return dict(tensor=meta, write_epoch=self.graph.epochs.get(self.graph.storage_key(meta), 0))

    def _describe(self, value, depth=0):
        require(depth < 6, 'ROUTED_FRAME_ARGUMENT_DEPTH_EXHAUSTED')
        if isinstance(value, torch.Tensor):
            return self._tensor(value)
        if value is None or type(value) in (int, float, bool, str):
            require(type(value) is not float or math.isfinite(value), 'ROUTED_FRAME_NONFINITE_CONFIG')
            return value
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        if isinstance(value, enum.Enum):
            return dict(enum_class=type(value).__module__ + '.' + type(value).__qualname__, name=value.name)
        if type(value) in (list, tuple) or isinstance(value, torch.Size):
            require(len(value) <= 64, 'ROUTED_FRAME_CONTAINER_BUDGET_EXHAUSTED')
            return [self._describe(item, depth + 1) for item in value]
        for name, (cls, fields) in self.frozen_types.items():
            if type(value) is cls:
                self._typed(value, name)
                return dict(payload_type=name, object_identity=id(value), fields={
                    key: self._describe(getattr(value, key), depth + 1) for key in fields})
        return dict(object_identity=id(value), object_type=type(value).__module__ + '.' + type(value).__qualname__)

    def _append(self, ctx, destination, value):
        size = len(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        require(ctx['record']['metadata_bytes'] + size <= self.max_bytes, 'ROUTED_FRAME_METADATA_BUDGET_EXHAUSTED')
        ctx['record']['metadata_bytes'] += size
        destination.append(value)

    def _update(self, ctx, destination, **values):
        # Charge each immutable packet once, including return metadata. Never
        # serialize the entire growing history during a callback.
        size = len(json.dumps(values, sort_keys=True, allow_nan=False).encode())
        require(ctx['record']['metadata_bytes'] + size <= self.max_bytes, 'ROUTED_FRAME_METADATA_BUDGET_EXHAUSTED')
        ctx['record']['metadata_bytes'] += size
        destination.update(values)

    def _hold(self, ctx, name, value):
        ctx['held'][name] = (value, self._tensor(value))
        self.graph.watched_storages.append(snapshot(ctx['held'][name][1]['tensor']))

    def _use(self, ctx, name, value):
        require(name in ctx['held'], 'ROUTED_FRAME_CURRENT_PRODUCER_MISSING:' + name)
        held, before = ctx['held'][name]
        require(value is held and self._tensor(value) == before,
                'ROUTED_FRAME_OBJECT_OR_EPOCH_CHANGED:' + name)
        return value

    def called(self, row):
        started = time.monotonic_ns()
        try:
            if row['role'] == 'rchain_model':
                self.total_parents += 1
                require(self.live is None, 'ROUTED_FRAME_REENTRANT_MODEL_CALL')
                if len(self.records) >= self.max_parents:
                    self.overflow += 1
                    row['routed_frame_refusal'] = 'ROUTED_FRAME_PARENT_BUDGET_EXHAUSTED'
                    return
                owner = row['input']['self']
                require(id(owner) in self.owners, 'ROUTED_FRAME_UNREGISTERED_MODEL_OWNER')
                record = dict(schema='demoncat.quant.routed_call_frames.v1', forward_id=self.graph.forward_id,
                    scope='PYTHON_FRAME_OBJECT_SCALAR_EPOCH_OBSERVATION_ONLY', status='OBSERVING',
                    model_boundary_sequence=row['sequence'], model_path=row['model_path'],
                    owner_object_identity=id(owner), request_scope_sha256=digest(self.boundary.scope),
                    frames=[], consumers=[], first_failure=None, metadata_bytes=0,
                    python_chain_complete=False, completed=False, native_registration_proven=False,
                    routed_output_row_proof=False, native_qualified=False, native_gates=list(NATIVE_GATES))
                self.records.append(record)
                self.live = dict(record=record, owner=owner, held={}, objects={}, frames={}, model_row=row)
                self._hold(self.live, 'model_input', row['input']['hidden_states'])
                self._update(self.live, record, input_request_rows=snapshot(self.graph.query(row['input']['hidden_states'])))
            ctx = self.live
            if ctx is None or row['role'] == 'rchain_property':
                return
            if row['role'] not in REQUIRED + ('rchain_model',):
                return
            self._current(ctx)
            require(len(ctx['record']['frames']) < self.max_frames, 'ROUTED_FRAME_CALL_BUDGET_EXHAUSTED')
            require(len(row['input']) <= 32, 'ROUTED_FRAME_ARGUMENT_COUNT_EXHAUSTED')
            event = dict(role=row['role'], boundary_sequence=row['sequence'], model_path=row['model_path'],
                function=row['function'], source_sha256=row['source_sha256'], status='ENTERED',
                input={key:self._describe(value) for key,value in row['input'].items()}, original_calls=1)
            self._append(ctx, ctx['record']['frames'], event)
            ctx['frames'][row['sequence']] = event
            self._call(ctx, row)
        except BaseException as error:
            if self.live is not None:
                self._fail(self.live, error)
            else:
                row['routed_frame_refusal'] = str(error)
        finally:
            self.callbacks += 1
            self.elapsed_ns += time.monotonic_ns() - started

    def _call(self, ctx, row):
        role, args, objects = row['role'], row['input'], ctx['objects']
        if role == 'route_pipeline':
            value = self._typed(args['fused_experts_input'], 'MoEFusedExpertsInput')
            self._use(ctx, 'prepared', value.hidden_states)
            objects['fused'] = value
        elif role == 'route_dispatch':
            value = self._typed(args['token_dispatch_input'], 'MoETokenDispatchInput')
            fused = objects['fused']
            require(all(getattr(value, name) is getattr(fused, name)
                        for name in ('hidden_states', 'topk_ids', 'topk_weights', 'routing', 'quant')),
                    'ROUTED_FRAME_DISPATCH_PAYLOAD_CHANGED')
        elif role == 'rchain_builder':
            require(args['fused_experts_input'] is objects['fused']
                and args['token_dispatch_output'] is objects['dispatch'], 'ROUTED_FRAME_BUILDER_INPUT_CHANGED')
        elif role in ('route_mlp', 'rchain_entry'):
            require(args['mlp_compute_input'] is objects['built'], 'ROUTED_FRAME_MLP_PAYLOAD_REPLACED')
        elif role == 'rchain_quant':
            built = self._typed(objects['built'], 'MoEMlpComputeInput')
            require(all(args[key] is getattr(built, key) for key in ('hidden_states','group_list','dynamic_scale'))
                and args['group_list_type'] == built.group_list_type
                and all(args[key] is getattr(built.weights, key)
                        for key in ('w1','w2','w1_scale','w2_scale','w1_scale_bias','w2_scale_bias','w1_offset','w2_offset')),
                    'ROUTED_FRAME_QUANT_PAYLOAD_CHANGED')
            require(args['fusion'] is True and args['dynamic_eplb'] is False and args['use_mxfp_quant'] is False
                and args['group_list_type'] == 1 and args['dynamic_scale'] is not None
                and all(args[key] is None for key in ('w1_scale_bias','w2_scale_bias','w1_offset','w2_offset')),
                    'ROUTED_FRAME_ONLY_OBSERVED_W8A8_FUSION_BRANCH')
            objects['quant'] = args
        elif role == 'rchain_cumsum':
            require(args['group_list'] is objects['quant']['group_list']
                and args['src_list_type'] == 1 and args['dst_list_type'] == 0,
                    'ROUTED_FRAME_CUMULATIVE_GROUP_SOURCE_CHANGED')
        elif role == 'rchain_gmm1':
            quant = objects['quant']
            single = lambda value: value[0] if type(value) is list and len(value) == 1 else value
            require(args['x'] is quant['hidden_states'] and args['x_scale'] is quant['dynamic_scale']
                and args['weight'] is single(quant['w1']) and args['weight_scale'] is single(quant['w1_scale'])
                and args['bias'] is None and args['use_mxfp_quant'] is False,
                    'ROUTED_FRAME_GMM1_CURRENT_ARGUMENTS_CHANGED')
            self._use(ctx, 'cumulative', args['group_list'])
        elif role == 'rchain_gmm2':
            quant = objects['quant']
            self._use(ctx, 'gmm1', args['hidden_states']); self._use(ctx, 'gmm1_scale', args['per_token_scale'])
            require(args['weight'] is quant['w2'] and args['weight_scale'] is quant['w2_scale']
                and args['group_list'] is quant['group_list'] and args['group_list_type'] == 1
                and args['bias'] is None and args['use_mxfp_quant'] is False,
                    'ROUTED_FRAME_GMM2_CURRENT_ARGUMENTS_CHANGED')
            ctx['held'].pop('gmm1'); ctx['held'].pop('gmm1_scale')
        elif role == 'route_combine':
            self._use(ctx, 'mlp_return', args['hidden_states'])
            require(args['combine_metadata'] is objects['combine'], 'ROUTED_FRAME_COMBINE_METADATA_REPLACED')
        elif role == 'rchain_finalize':
            self._use(ctx, 'combined', args['hidden_states'])

    def returned(self, row, values, result):
        ctx = self.live
        if ctx is None:
            return
        started = time.monotonic_ns()
        try:
            event = ctx['frames'].pop(row['sequence'], None)
            if event is not None:
                self._current(ctx)
                self._update(ctx, event, status='RETURNED', output=self._describe(result),
                    operator_sequences=list(row['operator_sequences']))
                self._return(ctx, row, result)
            if row is ctx['model_row']:
                require(isinstance(result, torch.Tensor), 'ROUTED_FRAME_MODEL_RETURN_REQUIRED')
                self._use(ctx, 'model_return', result)
                counts = Counter(event['role'] for event in ctx['record']['frames'] if event['status'] == 'RETURNED')
                require(all(counts[role] == 1 for role in REQUIRED), 'ROUTED_FRAME_COMPLETE_CALL_CHAIN_REQUIRED')
                require(ctx['record'].get('integer_route_evidence', {}).get('integer_map_valid') is True
                    and len(ctx['record']['consumers']) == 2
                    and all(event['status'] == 'RETURNED' for event in ctx['record']['consumers']),
                    'ROUTED_FRAME_COMPLETE_PARENT_AND_CONSUMERS_REQUIRED')
                ctx['record']['python_chain_complete'] = ctx['record']['first_failure'] is None
        except BaseException as error:
            if event is not None:
                event['status'] = 'REFUSED'
            self._fail(ctx, error)
        finally:
            if row is ctx['model_row']:
                ctx['record']['completed'] = True
                ctx['record']['status'] = ('PYTHON_OBJECT_CHAIN_COMPLETE_NATIVE_UNVERIFIED'
                    if ctx['record']['python_chain_complete'] else 'PYTHON_CHAIN_REFUSED')
                ctx['held'].clear(); ctx['objects'].clear(); ctx['frames'].clear()
                self.live = None
            self.callbacks += 1
            self.elapsed_ns += time.monotonic_ns() - started

    def _return(self, ctx, row, result):
        role, args, objects = row['role'], row['input'], ctx['objects']
        if role == 'rchain_prepare':
            prepared = self._typed(result, 'MoEPrepareOutput')
            require(prepared.hidden_states is args['hidden_states']
                and metadata(prepared.hidden_states) == row['snapshots'][id(args['hidden_states'])],
                    'ROUTED_FRAME_NONIDENTITY_PREPARATION_UNPROVEN')
            self._hold(ctx, 'prepared', prepared.hidden_states)
        elif role == 'route_dispatch':
            objects['dispatch'] = self._typed(result, 'MoETokenDispatchOutput')
            objects['combine'] = self._typed(result.combine_metadata, 'MoEAllGatherCombineMetadata')
        elif role == 'rchain_builder':
            built = self._typed(result, 'MoEMlpComputeInput'); dispatch = objects['dispatch']; fused = objects['fused']
            require(all(getattr(built, key) is getattr(dispatch, key)
                        for key in ('hidden_states','group_list','dynamic_scale','topk_scales'))
                and built.group_list_type == dispatch.group_list_type and built.weights is fused.weights
                and built.quant is fused.quant and built.fusion is True and args['use_fusion_ops'] is True,
                    'ROUTED_FRAME_BUILDER_OUTPUT_CHANGED')
            objects['built'] = built
        elif role == 'rchain_cumsum':
            self._hold(ctx, 'cumulative', result)
        elif role == 'rchain_gmm1':
            require(type(result) is tuple and len(result) == 3, 'ROUTED_FRAME_GMM1_RETURN_CHANGED')
            self._hold(ctx, 'gmm1', result[0]); self._hold(ctx, 'gmm1_scale', result[1])
        elif role == 'rchain_gmm2':
            self._hold(ctx, 'gmm2', result)
        elif role in ('rchain_quant', 'rchain_entry', 'route_mlp'):
            self._use(ctx, 'gmm2', result)
            if role == 'route_mlp':
                self._hold(ctx, 'mlp_return', result)
                for key in ('quant', 'built', 'dispatch', 'fused'):
                    objects.pop(key, None)
        elif role == 'route_combine':
            self._hold(ctx, 'combined', result)
            objects.pop('combine', None)
        elif role == 'rchain_finalize':
            self._use(ctx, 'combined', result)
            self._hold(ctx, 'finalized', result)

    def before_operator(self, state):
        started = time.monotonic_ns()
        try:
            self._before_operator(state)
        finally:
            self.callbacks += 1
            self.elapsed_ns += time.monotonic_ns() - started

    def _before_operator(self, state):
        ctx = self.live
        if ctx is None or self.boundary.routes.helper_active:
            return
        name = self.boundary.operator_aliases.get(state['func'], str(state['func']))
        try:
            if name == 'vllm.moe_forward_shared.default':
                require('parent' not in ctx['objects'], 'ROUTED_FRAME_DUPLICATE_PUBLIC_PARENT')
                ctx['objects']['parent'] = state
                state['routed_frame_parent'] = True
            elif name == 'aten.alias.default' and state['args']:
                for key in ('post_shared', 'post_routed'):
                    if key in ctx['held'] and state['args'][0] is ctx['held'][key][0]:
                        self._use(ctx, key, state['args'][0]); state['routed_frame_alias'] = key
            elif name == 'aten.view.default' and 'post_routed' in ctx['held'] and state['args']:
                if state['args'][0] is ctx['held']['post_routed'][0]:
                    self._use(ctx, 'post_routed', state['args'][0])
                    state['routed_frame_final_view'] = True
            elif name in ('aten.mul_.Tensor', 'aten.mul_.Scalar', 'aten.add_.Tensor') and 'post_routed' in ctx['held']:
                self._current(ctx)
                require(len(ctx['record']['consumers']) < 4, 'ROUTED_FRAME_CONSUMER_BUDGET_EXHAUSTED')
                bound, origins = _formals(state['func'], state['args'], state['kwargs'])
                self._use(ctx, 'post_routed', bound['self'])
                event = dict(sequence=state['event']['sequence'], schema=str(state['func']._schema),
                    formals={key:self._describe(value) for key,value in bound.items()}, argument_origins=origins,
                    scalar_owner_object_identity=id(ctx['owner']), scalar_config=ctx['owner'].routed_scaling_factor,
                    target_before=self._tensor(bound['self']), status='ENTERED', original_calls=0,
                    admission_granted=False, request_rows_proven=False)
                self._append(ctx, ctx['record']['consumers'], event)
                state['routed_frame_consumer'] = (event, bound['self'])
                if name.startswith('aten.mul_'):
                    require(type(bound['other']) in (float, int) and math.isfinite(bound['other'])
                        and bound['other'] == ctx['owner'].routed_scaling_factor,
                        'ROUTED_FRAME_ACTUAL_SCALAR_ARGUMENT_UNPROVEN')
                else:
                    self._use(ctx, 'post_shared', bound['other'])
                    require(bound['alpha'] == 1, 'ROUTED_FRAME_ACTUAL_ADD_ALPHA_UNSUPPORTED')
        except BaseException as error:
            self._fail(ctx, error)

    def after_operator(self, state, output=None, error=None):
        started = time.monotonic_ns()
        try:
            self._after_operator(state, output, error)
        finally:
            self.callbacks += 1
            self.elapsed_ns += time.monotonic_ns() - started

    def _after_operator(self, state, output, error):
        ctx = self.live
        if ctx is None or state is None:
            return
        try:
            if error is not None:
                self._fail(ctx, 'ROUTED_FRAME_OBSERVED_ORIGINAL_OPERATOR_FAILED:' + type(error).__name__)
                if 'failed_operator_sequence' not in ctx['record']:
                    ctx['record']['failed_operator_sequence'] = state['event']['sequence']
            if state.get('routed_frame_parent'):
                require(error is None and type(output) is tuple and len(output) == 2,
                        'ROUTED_FRAME_PUBLIC_PARENT_FAILED_OR_RETURN_CHANGED')
                self._use(ctx, 'finalized', output[1])
                self._hold(ctx, 'post_shared', output[0]); self._hold(ctx, 'post_routed', output[1])
                link = state['event'].get('integer_route_evidence')
                require(isinstance(link, dict) and type(link.get('record_index')) is int,
                        'ROUTED_FRAME_ACTUAL_INTEGER_PARENT_LINK_REQUIRED')
                raw = self.boundary.routes.records[link['record_index']]
                require(raw['parent_sequence'] == state['event']['sequence'] and raw['forward_id'] == self.graph.forward_id
                    and link['sha256'] == digest(raw), 'ROUTED_FRAME_INTEGER_PARENT_OR_DIGEST_CHANGED')
                stage_links = {}
                for stage, role in (('gmm1','rchain_gmm1'), ('gmm2','rchain_gmm2'), ('unpermute','route_combine')):
                    frames = [frame for frame in ctx['record']['frames'] if frame['role'] == role]
                    require(len(frames) == 1 and frames[0]['status'] == 'RETURNED'
                        and raw['stages'][stage]['original_calls'] == 1
                        and raw['stages'][stage]['sequence'] in frames[0]['operator_sequences'],
                            'ROUTED_FRAME_ACTUAL_OPERATOR_NOT_IN_CURRENT_SOURCE_FRAME:' + stage)
                    stage_links[stage] = dict(operator_sequence=raw['stages'][stage]['sequence'],
                        source_boundary_sequence=frames[0]['boundary_sequence'])
                self._update(ctx, ctx['record'], integer_route_evidence=snapshot(link),
                    actual_stage_frame_links=stage_links, public_return=self._describe(output))
                ctx['objects'].pop('parent', None)
            if 'routed_frame_alias' in state:
                key = state['routed_frame_alias']; old = ctx['held'][key][1]['tensor']; new = metadata(output)
                require(all(old[k] == new[k] for k in ('storage_address','storage_identity','storage_bytes',
                    'offset','shape','stride','dtype','device')), 'ROUTED_FRAME_PUBLIC_RETURN_ALIAS_CHANGED')
                self._hold(ctx, key, output)
            if state.get('routed_frame_final_view'):
                before = ctx['held']['post_routed'][1]['tensor']
                after = metadata(output)
                require(error is None and all(before[k] == after[k] for k in ('storage_address',
                    'storage_identity','storage_bytes','offset','shape','stride','dtype','device'))
                    and after['shape'] == ctx['held']['model_input'][1]['tensor']['shape'],
                    'ROUTED_FRAME_FINAL_VIEW_EXTENT_CHANGED')
                self._hold(ctx, 'model_return', output)
                self._update(ctx, ctx['record'], final_view=dict(sequence=state['event']['sequence'],
                    original_calls=state['event']['original_calls'], input=before, output=after))
            if 'routed_frame_consumer' in state:
                event, target = state['routed_frame_consumer']
                self._update(ctx, event, original_calls=state['event']['original_calls'],
                    observed_invalidations=snapshot(state['event'].get('invalidated_storages', [])),
                    target_after=self._tensor(target))
                require(error is None and output is target and event['original_calls'] == 1,
                        'ROUTED_FRAME_CONSUMER_FAILED_OR_RETURN_CHANGED')
                before, after = event['target_before'], event['target_after']
                require(after['write_epoch'] == before['write_epoch'] + 1
                    and all(value == after['tensor'][key] for key,value in before['tensor'].items() if key != 'version'),
                    'ROUTED_FRAME_CONSUMER_GEOMETRY_OR_WRITE_EPOCH_CHANGED')
                event['status'] = 'RETURNED'
                self._hold(ctx, 'post_routed', target)
        except BaseException as caught:
            self._fail(ctx, caught)

    def report(self):
        return snapshot(dict(schema='demoncat.quant.routed_frame_census.v1', records=self.records,
            total_parents=self.total_parents, overflow=self.overflow, callbacks=self.callbacks,
            observer_ns=self.elapsed_ns,
            timing_scope='CALLBACK_BODY_EXCLUDES_SHARED_PROFILE_AND_GRAPH_OVERHEAD',
            live_parents=int(self.live is not None),
            live_tensor_holds=len(self.live['held']) if self.live is not None else 0,
            live_payload_holds=len(self.live['objects']) if self.live is not None else 0,
            metadata_bytes=sum(row['metadata_bytes'] for row in self.records),
            max_parents=self.max_parents, max_frames_per_parent=self.max_frames,
            max_metadata_bytes_per_parent=self.max_bytes, tensor_readbacks=0,
            graph_row_bindings=0, native_registration_proven=False, routed_output_row_proof=False,
            native_qualified=False, c1=False, c2=False, c3=False))
