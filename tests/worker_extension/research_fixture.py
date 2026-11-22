"""Explicit CPU-only legacy factory injection; never a public admission grant."""
from unittest.mock import patch


def install_research_resolver(case, mechanism):
    from demoncat.runtime.vllm_ascend.worker_extension import runtime,registry
    original=runtime._resolve_provider
    def resolve(name,version,worker,model,identity):
        if name!=mechanism:return original(name,version,worker,model,identity)
        parameters=list(model.parameters())
        if not parameters or any(p.device.type!='cpu' for p in parameters):
            raise AssertionError('RESEARCH_FIXTURE_REQUIRES_CPU_MODEL')
        if name=='39.output_proxy':
            from demoncat.runtime.vllm_ascend.worker_extension.providers.fault39_output import Fault39OutputServingProvider
            return Fault39OutputServingProvider(worker,model,identity)
        return registry.create_research_provider(name,version,worker,model,identity)
    case.enterContext(patch.object(runtime,'_resolve_provider',side_effect=resolve))
