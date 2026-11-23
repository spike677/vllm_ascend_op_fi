"""V2-Lite AllGather output-row proxy. BLUE product rehearsal only.

This preserves producer metadata and original unpermute. It is not MC2/EP
qualification, nor the historical restore-index mutation variant.
"""
from torch.utils._python_dispatch import _disable_current_modes
from .moe8 import MoE8Site,CombineMode,require,fingerprint
from .mc2_output8 import MC2OutputConsumer


class AllGatherOutput8Site(MoE8Site):
    def install(self):
        result=super().install()
        self.output_mode=MC2OutputConsumer(self.region_leases)
        self.output_mode.events=self.events
        self.mode_entered=False
        def enter(*args):
            require(not self.mode_entered,'ALLGATHER_OUTPUT_OBSERVER_REENTRANT')
            self.output_mode.__enter__();self.mode_entered=True
        def leave(*args):
            if self.mode_entered:self.output_mode.__exit__(None,None,None);self.mode_entered=False
        self.hooks.extend([self.model.register_forward_pre_hook(enter),self.model.register_forward_hook(leave,always_call=True)])
        return result
    def combine(self,original,dispatcher,h,metadata,bias):
        require(len(self.parents)==1 and bias is None,'ALLGATHER_REAL_MODEL_PARENT_REQUIRED')
        name,mod=self.parents[0];ids=list(self.runner.input_batch.req_ids);produced=self.pipelines.get(id(dispatcher))
        require(len(ids)==1 and produced is not None and produced['dispatcher'] is dispatcher and produced['metadata'] is metadata
            and produced['mlp_output'] is h and produced['site']==name and produced['request_ids']==ids,'ALLGATHER_CURRENT_ROUTING_JOIN_UNPROVEN')
        i,p=metadata.expanded_row_idx,metadata.topk_weights
        require(h.ndim==2 and h.shape[0]==i.numel() and p.shape[1]==mod.top_k and tuple(metadata.restore_shape)==(p.shape[0],h.shape[1]),'ALLGATHER_GEOMETRY')
        from vllm.forward_context import get_forward_context
        actual=get_forward_context().attn_metadata;rows=list(actual.values()) if isinstance(actual,dict) else [actual]
        require({int(m.num_actual_tokens) for m in rows if hasattr(m,'num_actual_tokens')}=={p.shape[0]},'ALLGATHER_REQUEST_ROW_MAPPING')
        with _disable_current_modes():before=[fingerprint(i),fingerprint(p)]
        proof=CombineMode(h,i,p)
        with proof:out=original(dispatcher,h,metadata,bias)
        require(proof.calls==1 and out is proof.output,'ALLGATHER_NATIVE_UNPERMUTE_NOT_OBSERVED')
        with _disable_current_modes():require(before==[fingerprint(i),fingerprint(p)],'ALLGATHER_METADATA_WAS_MODIFIED')
        self.output_mode.watch(out,[ids[0]]*p.shape[0],site=name,fault=self.fault)
        e=self.events[-1];e.update(operator='npu::npu_moe_token_unpermute',reference_variant='allgather_output_row_proxy',
            request_ids=ids,actual_producer_join=True,topk_ids_sha256=produced['topk_ids_sha256'],
            token_rows=p.shape[0],row_request_ids=[ids[0]]*p.shape[0],native_calls=proof.calls,metadata_unchanged=True)
        calls=self.forward_calls[-1];calls[name]=calls.get(name,0)+1;del self.pipelines[id(dispatcher)]
        return out
    def close(self):
        if getattr(self,'mode_entered',False):self.output_mode.__exit__(None,None,None);self.mode_entered=False
        self.output_mode.close();super().close()


def validate_product_scope(*, family, source, tp, ep, soc):
    require(family=='deepseek_v2' and source=='BLUE_COMBINATION' and tp==2 and ep is False
            and soc is not None and soc['family']=='A2','ALLGATHER8_BLUE_A2_TP2_EP_OFF_ONLY')
