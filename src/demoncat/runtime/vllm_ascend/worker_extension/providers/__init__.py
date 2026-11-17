"""Route Worker actions to the existing provider catalog, without loading native code."""

def provider_capabilities():
    tensor = [dict(mechanism=mechanism, implementation_version='1',
                   operators=operators, attachment='owned_python_hooks',
                   execution='trusted_synchronous_tensor_replay_required',
                   execution_boundary='python_hook', natural_model_verified=False)
              for mechanism, operators in (
                  ('8.restore_offset', ['torch_npu.npu_moe_token_unpermute']),
                  ('15.alias_detach', ['aten.as_strided', 'aten.copy_', 'base_consumer']),
                  ('17.spatial_padding', ['torch.nn.Conv3d']),
                  ('18.api_half', ['torch.nn.RMSNorm']),
                  ('22.wrong_row', ['aten.index_select', 'RoPE_consumer']),
                  ('25.reduce_scatter', ['torch.distributed.reduce_scatter_tensor']))]
    return tensor + [
        {
            'mechanism': '26.residual_stomp',
            'implementation_version': '1',
            'operators': ['aclnnQuantMatmulWeightNz'],
            'attachment': 'reviewed_process_preload',
            'execution': 'trusted_serial_fresh_prefill_live_residual_required',
        },
        {
            'mechanism': 'quant.batch_shared_scale',
            'implementation_version': '1',
            'operators': ['aclnnDynamicQuantV2'],
            'attachment': 'reviewed_v2_only_vendor',
            'execution': 'trusted_first_b8_decode_embedding_norm_lineage_required',
        },
        {
            'mechanism': '26.fia_destination',
            'implementation_version': '1',
            'operators': ['aclnnFusedInferAttentionScoreV3'],
            'attachment': 'reviewed_process_preload',
            'execution': 'trusted_serial_fresh_prefill_deployment_required',
        },
        {
            'mechanism': '20.buffer_substitute',
            'implementation_version': '1',
            'operators': ['aclnnMatmulWeightNz', 'aclnnQuantMatmulWeightNz'],
            'attachment': 'process_preload',
            'execution': 'trusted_single_worker_replay',
        },
        {
            'mechanism': '39.matmul_range',
            'implementation_version': '1',
            'operators': ['aclnnMatmul'],
            'attachment': 'process_preload',
            'execution': 'trusted_single_worker_scheduler_replay',
        },
    ]


def resolve_provider(mechanism, implementation_version, worker, model, identity):
    from fault_injection.catalog import resolve

    resolve(mechanism)
    if implementation_version != '1' or mechanism not in (
        '20.buffer_substitute', '39.matmul_range', '26.fia_destination', 'quant.batch_shared_scale',
        '15.alias_detach', '17.spatial_padding', '18.api_half', '8.restore_offset', '22.wrong_row',
        '25.reduce_scatter', '26.residual_stomp'
    ):
        raise RuntimeError('WORKER_PROVIDER_NOT_ADAPTED')
    if mechanism in ('15.alias_detach', '17.spatial_padding', '18.api_half', '8.restore_offset', '22.wrong_row', '25.reduce_scatter'):
        from .tensor_modules import TensorModuleProvider
        return TensorModuleProvider(worker, model, identity, mechanism)
    if mechanism == '20.buffer_substitute':
        from .fault20 import Fault20Provider

        return Fault20Provider(worker, model, identity)
    if mechanism == '26.fia_destination':
        from .fault26 import Fault26FiaProvider
        return Fault26FiaProvider(worker, model, identity)
    if mechanism == '26.residual_stomp':
        from .fault26 import Fault26ResidualProvider
        return Fault26ResidualProvider(worker, model, identity)
    if mechanism == 'quant.batch_shared_scale':
        from .quant import QuantProvider
        return QuantProvider(worker, model, identity)
    from .fault39 import Fault39Provider

    return Fault39Provider(worker, model, identity)
