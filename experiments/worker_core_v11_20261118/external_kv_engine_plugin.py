"""Read-only dedicated-instance page boundary probe via vLLM general plugin.

Opt-in only in the isolated example. No third-party source file is edited.
Do not install this experimental plugin in a production environment.
"""
import os
from functools import wraps


def register():
    if os.environ.get('DCAT_V2_KV_PAGE_PROBE')!='1':return
    from vllm.v1.engine.core import EngineCore
    if getattr(EngineCore.step,'_dcat_page_probe',False):return
    original=EngineCore.step
    @wraps(original)
    def step(engine,*args,**kwargs):
        if not hasattr(engine,'_dcat_kv_page_probe'):
            from demoncat.runtime.vllm_ascend.worker_extension.persistent_kv_release import KVReleaseBarrier
            from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
            groups=engine.scheduler.kv_cache_config.kv_cache_groups;selected=[];layers=[]
            for i,group in enumerate(groups):
                spec=group.kv_cache_spec
                specs=list(spec.kv_cache_specs.values()) if type(spec).__name__=='UniformTypeKVCacheSpecs' else [spec]
                if specs and all(type(s).__name__=='FullAttentionSpec' for s in specs):
                    selected.append(i);layers.extend(group.layer_names)
            require(len(layers)==16,'V2_EXAMPLE_EXPECTS_16_ACTUAL_FULL_ATTENTION_LAYERS')
            def notify(action,payload):
                return engine.model_executor.collective_rpc('demoncat_kv_page_evidence',timeout=120,
                    args=(action,dict(payload,full_attention_layers=layers)))
            barrier=KVReleaseBarrier(engine,notify=notify,world_size=engine.vllm_config.parallel_config.tensor_parallel_size,
                                     group_ids=tuple(selected))
            barrier.install();engine._dcat_kv_page_probe=barrier
        return original(engine,*args,**kwargs)
    step._dcat_page_probe=True;EngineCore.step=step
