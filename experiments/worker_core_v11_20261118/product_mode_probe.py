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
    if action == 'configure':
        if authority is None:
            authority = configure_deployment_acceptance(worker, source='BLUE_COMBINATION',
                authorize_operator=lambda principal, operation: principal == 'product-acceptance')
        from demoncat.runtime.vllm_ascend.worker_extension.product_config import configure_product_deployments
        configure_product_deployments(worker, authority=authority, operator='product-acceptance', **options)
        return dict(status='TRUSTED_PROFILES_CONFIGURED')
    if action == 'negative':
        from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report
        from demoncat.runtime.vllm_ascend.worker_extension.product_binding import product_binding_rpc
        route=ordinary_route_report(options['mechanism'],worker,worker.model_runner.model)
        rejected={}
        for label,payload in [('unknown_profile',dict(operator='product-acceptance',profile='not-a-configured-profile')),
                              ('user_native_path',dict(operator='product-acceptance',profile=options['mechanism'],native_library='/not-allowed.so'))]:
            try:product_binding_rpc(worker,'capabilities',payload)
            except RuntimeError as error:rejected[label]=str(error)
            else:raise AssertionError('PRODUCT_NEGATIVE_WAS_ACCEPTED:'+label)
        assert not route['can_prepare'] and not hasattr(worker,'_demoncat_blue_acceptance_permit')
        return dict(rank=worker.rank,missing_permit=route,permit_remaining=False,**rejected)
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
    from product_http_probe import main as http_main
    return http_main()

if __name__ == '__main__':raise SystemExit(main())
