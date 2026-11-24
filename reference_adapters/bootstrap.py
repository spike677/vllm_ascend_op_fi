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
    from reference_adapters.quant_scale import FACTORIES as quant
    from reference_adapters.hidden_rows import FACTORIES as hidden
    from reference_adapters.logits_mapping import FACTORIES as logits
    from reference_adapters.swiglu import FACTORIES as swiglu
    from reference_adapters.vision_precision import FACTORIES as vision
    from reference_adapters.matmul_order import FACTORIES as matmul
    factories = {**quant, **hidden, **logits, **swiglu, **vision, **matmul}
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
        factory=factories[variant]
        stage='sampler' if variant in logits else 'vision' if variant in vision or variant=='row34_bf16_before_bias' else 'model'
        mechanism = 'external.' + variant
        spec = ExternalSite(mechanism, factory, factory_sha256(factory),
            observe_deployment(worker, worker.model_runner.model), stage=stage, region_kind='transient')
        results.append(register_external_site(worker, spec, authority=authority, operator='product-acceptance'))
        profiles[mechanism]=dict(mechanism=mechanism,mode='PRODUCT_ACCEPTANCE_ONLY',canary_prompts=[options['canary_tokens']])
    configure_product_profiles(worker, profiles=profiles,
        authority=authority, operator='product-acceptance')
    return results
