"""Route Worker actions to the existing provider catalog, without loading native code."""

def provider_capabilities():
    return [
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
        '20.buffer_substitute', '39.matmul_range'
    ):
        raise RuntimeError('WORKER_PROVIDER_NOT_ADAPTED')
    if mechanism == '20.buffer_substitute':
        from .fault20 import Fault20Provider

        return Fault20Provider(worker, model, identity)
    from .fault39 import Fault39Provider

    return Fault39Provider(worker, model, identity)
