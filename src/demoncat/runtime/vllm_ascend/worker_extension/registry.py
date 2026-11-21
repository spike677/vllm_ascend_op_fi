"""One public Worker route inventory; an entry never grants target qualification."""
from copy import deepcopy
import importlib

_TENSOR = (
    ('8.restore_offset', ['torch_npu.npu_moe_token_unpermute']),
    ('15.alias_detach', ['aten.as_strided', 'aten.copy_', 'base_consumer']),
    ('17.spatial_padding', ['torch.nn.Conv3d']),
    ('18.api_half', ['torch.nn.RMSNorm']),
    ('22.wrong_row', ['vllm::npu_rotary_embedding.positions', 'RoPE_consumer']),
)
_NATIVE = (
    ('39.output_proxy', 'fault39_output', 'create_single_request_provider', ['aten.mm', 'aten.matmul', 'aten.linear'],
     'reviewed_public_acl_native_span', 'trusted_single_site_or_complete_set_fresh_prefill_real_enqueue'),
    ('41.tail_column', 'fault41', 'Fault41ServingProvider', ['torch_npu.npu_fused_infer_attention_score'],
     'reviewed_public_acl_native_span', 'trusted_first_decode_live_kv_group_real_enqueue_required'),
    ('26.residual_stomp', 'fault26', 'Fault26ResidualProvider', ['aclnnQuantMatmulWeightNz'],
     'reviewed_process_preload', 'trusted_serial_fresh_prefill_live_residual_required'),
    ('quant.batch_shared_scale', 'quant', 'QuantProvider', ['aclnnDynamicQuantV2'],
     'reviewed_v2_only_vendor', 'trusted_first_b8_decode_embedding_norm_lineage_required'),
    ('26.fia_destination', 'fault26', 'Fault26FiaProvider', ['aclnnFusedInferAttentionScoreV3'],
     'reviewed_process_preload', 'trusted_serial_fresh_prefill_deployment_required'),
    ('20.buffer_substitute', 'fault20', 'Fault20Provider', ['aclnnMatmulWeightNz', 'aclnnQuantMatmulWeightNz'],
     'process_preload', 'trusted_single_worker_replay'),
    ('39.matmul_range', 'fault39', 'Fault39Provider', ['aclnnMatmul'],
     'process_preload', 'trusted_single_worker_scheduler_replay'),
)
_SEPARATE = {
    '28.fa.legal_tiling': 'native_execution_strategy.StrategySession',
    '28.fia.legal_tiling': 'fault_injection.fia',
    '28.fia.denominator_replay': 'fault_injection.fia',
    '33.paged_decode': 'demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd',
}
_ATTENTION = (
    ('28.fa.legal_tiling', 'FaLegalTilingServingProvider', ['torch_npu.npu_fusion_attention_v3']),
    ('28.fia.legal_tiling', 'FiaLegalTilingServingProvider', ['torch_npu.npu_fused_infer_attention_score']),
    ('28.fia.denominator_replay', 'FiaDenominatorReplayServingProvider', ['torch_npu.npu_fused_infer_attention_score']),
    ('33.paged_decode', 'PagedDecodeServingProvider', ['torch_npu.npu_fused_infer_attention_score']),
)

def rpc_provider_capabilities():
    """Preserve the v1 callable-provider list; no import or native load occurs."""
    rows = [dict(mechanism=name, implementation_version='1', operators=ops,
        attachment='owned_python_hooks', execution='trusted_synchronous_tensor_replay_required',
        execution_boundary='python_hook', natural_model_verified=False) for name, ops in _TENSOR]
    for row in rows:
        if row['mechanism'] in ('8.restore_offset', '18.api_half'):
            row.update(execution='reviewed_natural_serving_group_real_enqueue_required',
                       legacy_replay_fallback=False, historical_qualification_inherited=False)
        if row['mechanism'] == '18.api_half':
            row.update(execution='typed_ordinary_single_target_or_explicit_validation_group',
                       ordinary_binding='install_rms18_task_binding', ordinary_scope='one_actual_rms_return',
                       ordinary_npu_validation='SOURCE126_BLUE_ZONE_DECLARED_SCOPE_ACCEPTED',
                       current_worker_qualification_inherited=False, default_census=False)
        if row['mechanism'] == '17.spatial_padding':
            row.update(optional_two_healthy_mm_census=True,
                       observation_binding='_demoncat_vision17_observation_binding',
                       group_binding='_demoncat_vision17_group_binding',
                       group_scope='actual_single_projection_fresh_mm_requests',
                       group_native_qualification=False,
                       census_grants_fault_admission=False)
        if row['mechanism'] == '22.wrong_row':
            row.update(execution='reviewed_fused_serving_group_real_enqueue_required',
                       variant_selection='actual_source_bound_fused_calls', legacy_lookup_fallback=False)
    rows += [dict(mechanism=name, implementation_version='1', operators=ops,
        attachment=attachment, execution=execution) for name, _, _, ops, attachment, execution in _NATIVE]
    rows += [dict(mechanism=name, implementation_version='1', operators=ops,
        attachment='owner_scoped_actual_model_module_api',
        execution='trusted_actual_attention_census_and_matched_native_baseline',
        natural_model_verified=False, fault_execution_ready=False,
        fault_admission='new_actual_stock_private_bridge_required',
        executable_stages=['actual_call_observation', 'matched_native_baseline'])
        for name, _, ops in _ATTENTION]
    from fault_injection.delivery_scope import describe_delivery
    for row in rows:
        row['current_delivery'] = describe_delivery(row['mechanism'])
        from fault_injection.request_plan import request_plan
        row['request_plan'] = request_plan(row['mechanism'])
        row['research_entry'] = row['current_delivery'].get('research_entry')
        from .ordinary_admission import ordinary_route_report
        row['ordinary_task_prerequisites'] = ordinary_route_report(row['mechanism'])
        # A deployed single-site driver exists; this does not grant a fresh
        # Worker's live normal/input/ownership admission or main-host wiring.
        if row['mechanism'] == '33.paged_decode':
            row.update(fault_execution_ready=True, execution_scope='SOURCE105_SINGLE_SITE_ONLY',
                       fault_admission='CURRENT_LIVE_SINGLE_SITE_NORMAL_CONTROLS_REQUIRED',
                       executable_stages=['actual_call_observation', 'matched_native_baseline',
                                          'single_site_fault', 'recovery_off'])
        if row['mechanism'] == '39.output_proxy':
            row['execution'] = 'trusted_single_site_fresh_prefill_real_enqueue'
        from .ordinary_admission import ORDINARY
        row['historical_execution_description'] = row['execution']
        row['execution'] = ('ordinary_task_binding_and_live_call_required'
                            if row['mechanism'] in ORDINARY else 'public_route_closed')
        row['fault_execution_ready'] = False  # Static declaration cannot attest this Worker.
        row['public_protocol'] = ('six_actual_batches' if row['mechanism']=='quant.batch_shared_scale'
                                  else 'six_actual_requests_with_calibration' if row['mechanism']=='41.tail_column'
                                  else 'four_actual_requests' if row['mechanism'] in ORDINARY else None)
    return deepcopy(rows)

def describe_worker_provider(mechanism):
    from fault_injection.catalog import resolve
    capability = resolve(mechanism)
    from fault_injection.delivery_scope import describe_delivery
    delivery = describe_delivery(mechanism)
    route = next((row for row in rpc_provider_capabilities() if row['mechanism'] == mechanism), None)
    return dict(mechanism=mechanism, implementation_version='1', rpc_available=route is not None,
        entrypoint=('demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension' if route else _SEPARATE.get(mechanism)),
        route=route, status=('EXCLUDED_CURRENT_DELIVERY' if delivery['state'] == 'EXCLUDED_CURRENT_DELIVERY' else
                            'UNSUPPORTED' if delivery['state'] == 'UNSUPPORTED' else
                            'ORDINARY_ADAPTER_AND_ACCEPTED_DEPLOYMENT_REQUIRED' if route else 'WORKER_RPC_ADAPTATION_PENDING'),
        current_delivery=delivery,
        mechanism_boundary=capability.boundary, target_selection='actual_objects_and_calls',
        qualification_required=True, qualification_granted=False,
        diagnostic_limits=['A catalog or structural candidate is not an observed eligible call.',
                           'A registered factory is not proof that its deployment matches this Worker.',
                           'C1, C2, C3, Recovery and E5 remain independently reported.'])

def worker_core_manifest():
    from fault_injection.catalog import CAPABILITIES
    rows = [describe_worker_provider(item.name) for item in CAPABILITIES]
    return dict(schema='demoncat.worker_core.manifest.v1', mechanisms=rows,
        mechanism_count=len(rows), rpc_provider_count=sum(row['rpc_available'] for row in rows),
        worker_lifecycle_owner='serving_runtime', creates_or_restarts_workers=False,
        native_runtime='fault_injection.native_build.NativeBuildManager',
        recovery_authority='existing WorkerFaultRuntime and mechanism-specific native/owned-resource readback',
        coverage_authority='fault_injection.coverage_session.CoverageSession', qualification_granted=False)

def create_research_provider(mechanism, version, worker, model, identity):
    from fault_injection.catalog import resolve
    resolve(mechanism)
    if mechanism == '25.reduce_scatter':
        raise RuntimeError('FAULT25_UNSUPPORTED_NO_SUBSTITUTE')
    if version != '1':
        raise RuntimeError('WORKER_PROVIDER_NOT_ADAPTED')
    if (mechanism == '17.spatial_padding'
            and hasattr(worker, '_demoncat_vision17_group_binding')):
        if hasattr(worker, '_demoncat_vision17_observation_binding'):
            raise RuntimeError('VISION17_DEPLOYMENT_BINDINGS_CONFLICT')
        module = importlib.import_module(__package__ + '.providers.vision17_group')
        if type(worker._demoncat_vision17_group_binding) is not module.Vision17GroupBinding:
            raise RuntimeError('VISION17_TYPED_GROUP_BINDING_REQUIRED')
        return module.Vision17GroupProvider(worker, model, identity)
    if (mechanism == '17.spatial_padding'
            and hasattr(worker, '_demoncat_vision17_observation_binding')):
        module = importlib.import_module(__package__ + '.providers.vision17')
        return module.Vision17ObservationProvider(worker, model, identity)
    if mechanism == '22.wrong_row':
        module = importlib.import_module(__package__ + '.providers.fault22')
        return module.Fault22ServingProvider(worker, model, identity)
    if mechanism in ('8.restore_offset', '18.api_half'):
        if mechanism == '8.restore_offset':
            from .serving_tensor_binding import _ATTRIBUTE, TensorServingBinding
            bindings = getattr(worker, _ATTRIBUTE, None)
            binding = bindings.get(mechanism) if type(bindings) is dict else None
            if type(binding) is TensorServingBinding and binding.runtime_only:
                from .restore8_task import Restore8TaskProvider
                return Restore8TaskProvider(worker, model, identity)
        if mechanism == '18.api_half':
            from .serving_tensor_binding import _ATTRIBUTE, TensorServingBinding
            bindings = getattr(worker, _ATTRIBUTE, None)
            binding = bindings.get(mechanism) if type(bindings) is dict else None
            if type(binding) is TensorServingBinding and binding.runtime_only:
                module = importlib.import_module(__package__ + '.providers.rms18_task')
                return module.Rms18TaskProvider(worker, model, identity)
        module = importlib.import_module(__package__ + '.providers.tensor_serving')
        return module.TensorServingProvider(worker, model, identity, mechanism)
    if mechanism in {name for name, _ in _TENSOR}:
        module = importlib.import_module(__package__ + '.providers.tensor_modules')
        return module.TensorModuleProvider(worker, model, identity, mechanism)
    for name, module_name, class_name, *_ in _NATIVE:
        if name == mechanism:
            module = importlib.import_module(__package__ + '.providers.' + module_name)
            return getattr(module, class_name)(worker, model, identity)
    for name, class_name, _ in _ATTENTION:
        if name == mechanism:
            if mechanism == '33.paged_decode':
                module = importlib.import_module(__package__ + '.providers.paged33_s2')
                return module.resolve_current_paged_decode_provider(worker, model, identity)
            module = importlib.import_module(__package__ + '.providers.attention_serving')
            return getattr(module, class_name)(worker, model, identity)
    raise RuntimeError('WORKER_PROVIDER_NOT_ADAPTED')


def create_worker_provider(mechanism, version, worker, model, identity):
    """Public route; research protocols require explicit direct construction."""
    from fault_injection.catalog import resolve
    from .ordinary_admission import require_ordinary_route
    resolve(mechanism)
    if version != '1':
        raise RuntimeError('WORKER_PROVIDER_NOT_ADAPTED')
    admission=require_ordinary_route(mechanism,worker,model)
    from .continuous_binding import current_binding
    if current_binding(worker,model,mechanism) is not None:
        from .providers.continuous import ContinuousProvider
        provider=ContinuousProvider(worker,model,identity,mechanism)
        provider._deployment_admission=(worker,model,admission['deployment_acceptance'])
        return provider
    tensor_binding=getattr(worker,'_demoncat_natural_tensor_serving_bindings',{}).get(mechanism)
    if mechanism in ('18.api_half','22.wrong_row') and getattr(tensor_binding,'execution_profile',None) is not None:
        from .new_stack_rms18 import PROFILE,NewStackRms18Provider,NewStackTP2Rms18Provider
        if tensor_binding.execution_profile!=PROFILE:raise RuntimeError('UNREVIEWED_NEWSTACK_EXECUTION_PROFILE')
        if mechanism=='18.api_half':cls=NewStackTP2Rms18Provider if tensor_binding.group_context is not None else NewStackRms18Provider
        else:
            from .new_stack_rope22 import NewStackRope22Provider,NewStackTP2Rope22Provider
            cls=NewStackTP2Rope22Provider if tensor_binding.group_context is not None else NewStackRope22Provider
        provider=cls(worker,model,identity)
    elif mechanism=='41.tail_column':
        from .providers.kv41_task import KV41TaskProvider
        if getattr(getattr(worker,'_demoncat_kv41_serving_binding',None),'group_context',None) is not None:
            from .kv41_tp2 import TP2KV41TaskProvider
            provider=TP2KV41TaskProvider(worker,model,identity)
        else:provider=KV41TaskProvider(worker,model,identity)
    elif mechanism=='18.api_half' and getattr(getattr(worker,'_demoncat_natural_tensor_serving_bindings',{}).get(mechanism),'group_context',None) is not None:
        from .rms18_tp2 import TP2Rms18Provider
        provider=TP2Rms18Provider(worker,model,identity)
    elif mechanism=='22.wrong_row':
        if getattr(getattr(worker,'_demoncat_natural_tensor_serving_bindings',{}).get(mechanism),'group_context',None) is not None:
            from .rope22_tp2 import TP2Rope22Provider
            provider=TP2Rope22Provider(worker,model,identity)
        else:
            from .rope22_task import Rope22TaskProvider
            provider=Rope22TaskProvider(worker,model,identity)
    elif mechanism=='17.spatial_padding':
        from .providers.vision17_task import Vision17TaskProvider
        provider=Vision17TaskProvider(worker,model,identity)
    else:provider=create_research_provider(mechanism,version,worker,model,identity)
    if mechanism=='quant.batch_shared_scale':
        provider.ordinary_only=True
    provider._deployment_admission=(worker,model,admission['deployment_acceptance'])
    return provider
