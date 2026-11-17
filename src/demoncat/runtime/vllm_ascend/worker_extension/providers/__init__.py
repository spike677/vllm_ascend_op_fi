"""Route Worker actions to the existing provider catalog, without loading native code."""

def provider_capabilities():
    return [
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
        '20.buffer_substitute', '39.matmul_range', '26.fia_destination', 'quant.batch_shared_scale'
    ):
        raise RuntimeError('WORKER_PROVIDER_NOT_ADAPTED')
    if mechanism == '20.buffer_substitute':
        from .fault20 import Fault20Provider

        return Fault20Provider(worker, model, identity)
    if mechanism == '26.fia_destination':
        from .fault26 import Fault26FiaProvider
        return Fault26FiaProvider(worker, model, identity)
    if mechanism == 'quant.batch_shared_scale':
        from .quant import QuantProvider
        return QuantProvider(worker, model, identity)
    from .fault39 import Fault39Provider

    return Fault39Provider(worker, model, identity)
