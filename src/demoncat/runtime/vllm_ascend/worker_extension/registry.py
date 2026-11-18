"""One public Worker route inventory; an entry never grants target qualification."""
from copy import deepcopy
import importlib

_TENSOR = (
    ('8.restore_offset', ['torch_npu.npu_moe_token_unpermute']),
    ('15.alias_detach', ['aten.as_strided', 'aten.copy_', 'base_consumer']),
    ('17.spatial_padding', ['torch.nn.Conv3d']),
    ('18.api_half', ['torch.nn.RMSNorm']),
    ('22.wrong_row', ['vllm::npu_rotary_embedding.positions', 'RoPE_consumer']),
    ('25.reduce_scatter', ['torch.distributed.reduce_scatter_tensor']),
)
_NATIVE = (
    ('39.output_proxy', 'fault39_output', 'Fault39OutputServingProvider', ['aten.mm', 'aten.matmul', 'aten.linear'],
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
        if row['mechanism'] == '17.spatial_padding':
            row.update(optional_two_healthy_mm_census=True,
                       observation_binding='_demoncat_vision17_observation_binding',
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
    return deepcopy(rows)

def describe_worker_provider(mechanism):
    from fault_injection.catalog import resolve
    capability = resolve(mechanism)
    route = next((row for row in rpc_provider_capabilities() if row['mechanism'] == mechanism), None)
    return dict(mechanism=mechanism, implementation_version='1', rpc_available=route is not None,
        entrypoint=('demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension' if route else _SEPARATE[mechanism]),
        route=route, status='DEPLOYMENT_AND_TARGET_QUALIFICATION_REQUIRED' if route else 'WORKER_RPC_ADAPTATION_PENDING',
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

def create_worker_provider(mechanism, version, worker, model, identity):
    from fault_injection.catalog import resolve
    resolve(mechanism)
    if version != '1':
        raise RuntimeError('WORKER_PROVIDER_NOT_ADAPTED')
    if (mechanism == '17.spatial_padding'
            and hasattr(worker, '_demoncat_vision17_observation_binding')):
        module = importlib.import_module(__package__ + '.providers.vision17')
        return module.Vision17ObservationProvider(worker, model, identity)
    if mechanism == '22.wrong_row':
        module = importlib.import_module(__package__ + '.providers.fault22')
        return module.Fault22ServingProvider(worker, model, identity)
    if mechanism in ('8.restore_offset', '18.api_half'):
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
            module = importlib.import_module(__package__ + '.providers.attention_serving')
            return getattr(module, class_name)(worker, model, identity)
    raise RuntimeError('WORKER_PROVIDER_NOT_ADAPTED')
