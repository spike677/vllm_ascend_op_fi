"""Acceptance of the product facade. Not the main-project service entry point."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


def trusted_bootstrap(wrapper, action, options):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    worker = wrapper.worker if isinstance(wrapper, WorkerWrapperBase) else wrapper
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import (
        authority_for, configure_deployment_acceptance, install_deployment_acceptance_permit,
        remove_deployment_acceptance_permit)
    authority = authority_for(worker)
    if action == 'external_configure':
        from demoncat.runtime.vllm_ascend.worker_extension.external_provider import ExternalSite, register_external_site, factory_sha256
        from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import observe_deployment
        from demoncat.runtime.vllm_ascend.worker_extension.product_binding import configure_product_profiles
        from external_gain_example import ExternalGainSite
        assert set(options)=={'canary_tokens'}
        if authority is None:
            authority=configure_deployment_acceptance(worker,source='BLUE_COMBINATION',
                authorize_operator=lambda principal,operation:principal=='product-acceptance')
        mechanism='external.gain_three_quarters'
        spec=ExternalSite(mechanism,ExternalGainSite,factory_sha256(ExternalGainSite),observe_deployment(worker,worker.model_runner.model))
        registration=register_external_site(worker,spec,authority=authority,operator='product-acceptance')
        configure_product_profiles(worker,profiles={mechanism:dict(mechanism=mechanism,mode='PRODUCT_ACCEPTANCE_ONLY',
            canary_prompts=[options['canary_tokens']])},authority=authority,operator='product-acceptance')
        return registration
    if action == 'audit':
        runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
        return dict(rank=worker.rank,leases=len(getattr(runtime,'_leases',{})),owners=len(getattr(runtime,'_tenants',{})),
                    bindings=len(getattr(worker,'_demoncat_continuous_bindings',{})),
                    wrappers=sum(n in vars(worker.model_runner) for n in ('execute_model','sample_tokens','_preprocess')))
    if action == 'configure':
        from demoncat.runtime.vllm_ascend.worker_extension.product_bootstrap import configure_product_bootstrap, product_bootstrap_rpc
        # Isolated acceptance only. Production supplies its own authenticated callback.
        authorize = authority.authorize_operator if authority else lambda principal, operation: principal == 'product-acceptance'
        configure_product_bootstrap(worker, source='BLUE_COMBINATION', authorize_operator=authorize, **options)
        return product_bootstrap_rpc(worker, 'configure', {'operator':'product-acceptance'})
    if action == 'registration_identity':
        from demoncat.runtime.vllm_ascend.worker_extension.product_registration import observe_identity
        return observe_identity(worker, options['mechanism'])
    if action == 'registered_cleanup':
        assert not hasattr(worker,'_demoncat_blue_acceptance_permit')
        runtime=worker._demoncat_worker_fault_runtime
        return dict(status='NO_PERMIT_USED',permit_remaining=False,leases=len(runtime._leases),owners=len(runtime._tenants))
    if action == 'negative':
        from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report
        from demoncat.runtime.vllm_ascend.worker_extension.product_binding import product_binding_rpc
        if options['mechanism'].startswith('external.'):
            from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import deployment_acceptance_report
            route=deployment_acceptance_report(options['mechanism'],worker,worker.model_runner.model)
        else:route=ordinary_route_report(options['mechanism'],worker,worker.model_runner.model)
        rejected={}
        for label,payload in [('unknown_profile',dict(operator='product-acceptance',profile='not-a-configured-profile')),
                              ('user_native_path',dict(operator='product-acceptance',profile=options['mechanism'],native_library='/not-allowed.so'))]:
            try:product_binding_rpc(worker,'capabilities',payload)
            except RuntimeError as error:rejected[label]=str(error)
            else:raise AssertionError('PRODUCT_NEGATIVE_WAS_ACCEPTED:'+label)
        # Real Worker-side public configuration rejection; this does not claim a
        # yellow CLI/daemon integration test.
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        try: FaultConfig(99, 'unknown')
        except (ValueError, RuntimeError) as error: rejected['unknown_mechanism'] = str(error)
        else: raise AssertionError('PRODUCT_UNKNOWN_MECHANISM_ACCEPTED')
        runtime = worker._demoncat_worker_fault_runtime
        assert not route['can_prepare'] and not hasattr(worker,'_demoncat_blue_acceptance_permit')
        assert len(runtime._leases) == len(runtime._tenants) == 0
        return dict(rank=worker.rank,missing_permit=route,permit_remaining=False,
                    resources=dict(leases=0,owners=0),**rejected)
    if action == 'permit':
        return install_deployment_acceptance_permit(worker, authority=authority, operator='product-acceptance', **options)
    if action == 'remove_permit':
        remove_deployment_acceptance_permit(worker, authority=authority, operator='product-acceptance')
        runtime = worker._demoncat_worker_fault_runtime
        return dict(status='PERMIT_REMOVED', leases=len(runtime._leases), owners=len(runtime._tenants))
    raise ValueError('UNKNOWN_BOOTSTRAP')


def main():
    # The earlier direct-LLM harness is preserved in commit 9ec051ea.
    # Current product acceptance must use real inference HTTP requests.
    from registered_http_probe import main as http_main
    return http_main()

if __name__ == '__main__':raise SystemExit(main())
