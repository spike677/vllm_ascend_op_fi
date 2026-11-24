"""Test-only trusted registration; caller must own the isolated worker session."""

def trusted_bootstrap(wrapper, action, options):
    from experiments.worker_core_v11_20261118.product_mode_probe import trusted_bootstrap as existing
    if action != 'external_configure':
        return existing(wrapper, action, options)
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import authority_for, configure_deployment_acceptance
    from demoncat.runtime.vllm_ascend.worker_extension.external_provider import ExternalSite, register_external_site, factory_sha256
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import observe_deployment
    from demoncat.runtime.vllm_ascend.worker_extension.product_binding import configure_product_profiles
    import importlib
    # A supplement may contain one reviewed family only. Never import another
    # unreviewed factory merely to register the selected variant.
    modules={'row9':'quant_scale','row10':'quant_scale','row7':'hidden_rows','row8':'hidden_rows',
        'row12':'logits_mapping','row3':'swiglu','row16':'vision_precision','row32':'vision_precision',
        'row19':'matmul_order','row34':'matmul_order'}
    assert set(options) == {'variant', 'canary_tokens'}
    variants=options['variant'] if isinstance(options['variant'],list) else [options['variant']]
    assert variants and len(variants)==len(set(variants))
    worker = wrapper.worker if isinstance(wrapper, WorkerWrapperBase) else wrapper
    authority = authority_for(worker)
    if authority is None:
        authority = configure_deployment_acceptance(worker, source='BLUE_COMBINATION',
            authorize_operator=lambda principal, operation: principal == 'product-acceptance')
    results=[];profiles={}
    for variant in variants:
        factory=importlib.import_module('reference_adapters.'+modules[variant.split('_')[0]]).FACTORIES[variant]
        stage='sampler' if variant.startswith('row12_') else 'vision' if variant.startswith(('row16_','row32_','row34_')) else 'model'
        mechanism = 'external.' + variant
        spec = ExternalSite(mechanism, factory, factory_sha256(factory),
            observe_deployment(worker, worker.model_runner.model), stage=stage, region_kind='transient')
        results.append(register_external_site(worker, spec, authority=authority, operator='product-acceptance'))
        profiles[mechanism]=dict(mechanism=mechanism,mode='PRODUCT_ACCEPTANCE_ONLY',canary_prompts=[options['canary_tokens']])
    configure_product_profiles(worker, profiles=profiles,
        authority=authority, operator='product-acceptance')
    return results
