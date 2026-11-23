"""Explicit CPU research resolver. Never installed by public production code."""
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension import runtime,registry

def install(test):
    def resolve(mechanism,version,worker,model,identity):
        if mechanism not in ('41.tail_column','26.residual_stomp','26.fia_destination'):
            raise AssertionError('CPU_RESEARCH_MECHANISM_NOT_ALLOWED')
        if not type(worker).__module__.startswith('test_') or any(p.device.type!='cpu' for p in model.parameters()):
            raise AssertionError('CPU_RESEARCH_FIXTURE_ONLY')
        return registry.create_research_provider(mechanism,version,worker,model,identity)
    test.enterContext(patch.object(runtime,'_resolve_provider',side_effect=resolve))
