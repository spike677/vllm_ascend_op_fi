"""Network-isolated A2 acceptance only; uses the public persistent host facade."""
import hashlib,inspect,os
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from experiments.worker_core_v11_20261118.external_persistent_product_probe import PersistentExampleExtension
from reference_adapters.persistent_probe import main
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
from fault_injection.sites.mla28 import fingerprint
from reference_adapters.lm_head_rows import TOKEN_IDS,VARIANTS,is_weight_reader

PINS={
 'vllm.model_executor.layers.vocab_parallel_embedding':'60b116e260d5438513c9bb60370306eed5e9b0ce612bfbe0a3885e64fdf0587b',
 'vllm.model_executor.layers.utils':'8d732d33cbc9ebcf18eba9ec3ef3ecedd46d0fc1405beff97d2854fadb7532fd',
 'vllm_ascend.ops.vocab_parallel_embedding':'a4a8517abc3a4f8494f27cc4cc3a876264ef8fea8cfc6de9abdbaf75d42c3b24'}


def inspect_head(model):
    import importlib,torch_npu
    for name,sha in PINS.items():
        module=importlib.import_module(name)
        require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==sha,'ROW12_SOURCE_CHANGED:'+name)
    candidates=[m for n,m in model.named_modules() if n.split('.')[-1]=='lm_head']
    require(len(candidates)==1,'ROW12_UNIQUE_LM_HEAD_REQUIRED');m=candidates[0]
    method=m.quant_method
    require(type(method).__name__=='UnquantizedEmbeddingMethod'
        and inspect.getmodule(type(method)).__name__=='vllm.model_executor.layers.vocab_parallel_embedding',
        'ROW12_UNREVIEWED_HEAD_METHOD:'+type(method).__name__)
    require(not vars(method) and 'apply' not in vars(method),'ROW12_PREPACKED_OR_PATCHED_METHOD')
    require(m.bias is None and set(dict(m.named_parameters()))=={'weight'}
        and not list(m.named_buffers()),'ROW12_DERIVED_OR_BIAS_PARAMETERS')
    require(not any(isinstance(v,torch.Tensor) for v in vars(m).values()),'ROW12_UNREGISTERED_TENSOR_COPY')
    require(torch_npu.get_npu_format(m.weight)==2,'ROW12_ND_FORMAT_REQUIRED')
    require(sum(p is m.weight for mod in model.modules() for p in mod._parameters.values())==1,
        'ROW12_TIED_WEIGHT_NOT_SUPPORTED')
    s=m.shard_indices
    require(s.added_vocab_start_index==s.added_vocab_end_index,'ROW12_ADDED_VOCAB_UNSUPPORTED')
    admission=dict(vocabulary_size=m.org_vocab_size,vocabulary_start=s.org_vocab_start_index,
        vocabulary_end=s.org_vocab_end_index,padded_start=s.padded_org_vocab_start_index,
        padded_end=s.padded_org_vocab_end_index,derived_copies=(),tied_embedding=False,
        source_sha256=PINS['vllm.model_executor.layers.vocab_parallel_embedding'],
        actual_weight_consumer='aten::mm:lm_head_weight')
    return m,admission


class LmHeadExtension(PersistentExampleExtension):
    def demoncat_persistent_example(self,action,options):
        if action=='configure':
            from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import configure_deployment_acceptance
            from demoncat.runtime.vllm_ascend.worker_extension.external_persistent_runtime import install_persistent_runtime,register_persistent_site,PersistentSite
            from demoncat.runtime.vllm_ascend.worker_extension.external_provider import factory_sha256
            from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import observe_deployment
            from reference_adapters.lm_head_provider import LmHeadCollapseProvider,LmHeadSwapProvider
            variant=os.environ['DCAT_REFERENCE_ROW12'];require(variant in VARIANTS,'ROW12_FIXED_VARIANT_REQUIRED')
            factory=LmHeadSwapProvider if variant.endswith('swap') else LmHeadCollapseProvider
            model=self.model_runner.model;m,admission=inspect_head(model)
            require('compute_logits' not in vars(model),'ROW12_COMPUTE_LOGITS_ALREADY_OWNED')
            self._v2_example_boundary=dict(dedicated_instance=True,accepting=False,inflight=0)
            context=dict(module=m,admission=admission,provider=None,reader_calls=0,
                boundary=lambda:dict(self._v2_example_boundary));self._reference_lm_context=context
            authority=configure_deployment_acceptance(self,source='BLUE_COMBINATION',
                authorize_operator=lambda p,o:p=='v2-example')
            install_persistent_runtime(self,authority=authority,operator='v2-example')
            actual=observe_deployment(self,model)
            require(actual['tp']==2 and actual['eager'] is True and actual['max_num_seqs']==1,'ROW12_RANGE')
            self._v2_example_mechanism='external.'+variant
            register_persistent_site(self,PersistentSite(self._v2_example_mechanism,'persistent_weight',factory,
                factory_sha256(factory),actual),authority=authority,operator='v2-example')
            self._row12_rows=[];self._row12_pending=[];self._row12_all=[]
            original=model.compute_logits;worker=self
            class Reader(TorchDispatchMode):
                def __init__(self):super().__init__();self.matches=[];self.names=[]
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    name=func._schema.name;self.names.append(name)
                    match=is_weight_reader(name,args,m.weight)
                    result=func(*args,**(kwargs or {}))
                    if match:self.matches.append(name)
                    return result
            def compute(hidden,*args,**kwargs):
                trace=Reader()
                with trace:result=original(hidden,*args,**kwargs)
                # A read-only census must not kill inference merely because a
                # candidate was absent. Refuse at provider construction later.
                if len(trace.matches)==1:
                    context['reader_calls']+=1
                    context['admission']['actual_weight_consumer']=trace.matches[0]+':lm_head_weight'
                else:context['reader_failure']=dict(reason='ROW12_DIRECT_WEIGHT_READER_NOT_OBSERVED',operators=trace.names)
                ids=list(worker.model_runner.input_batch.req_ids)
                require(len(ids)==1,'ROW12_SERIAL_REQUEST_REQUIRED')
                provider=context.get('provider')
                if provider:
                    require(len(trace.matches)==1,'ROW12_ADMITTED_READER_CHANGED')
                    provider.consumed(ids[0])
                row=dict(hidden_sha256=fingerprint(hidden),logits_sha256=None if result is None else fingerprint(result),
                    finite=bool(torch.isfinite(hidden).all().item()) and (result is None or bool(torch.isfinite(result).all().item())),
                    tokens=list(TOKEN_IDS),target_logits=None,target_probabilities=None,sampler_consumed=False,
                    weight_readers=trace.matches)
                if result is not None:
                    require(result.ndim==2 and result.shape[-1]==admission['vocabulary_size'],
                            'ROW12_FULL_LOGITS_REQUIRED_REDUCE_SAMPLE_UNSUPPORTED')
                    row.update(target_logits=result[:,list(TOKEN_IDS)].float().cpu().tolist(),
                        target_probabilities=result.double().softmax(-1)[:,list(TOKEN_IDS)].cpu().tolist())
                    worker._row12_pending.append((result,row))
                worker._row12_rows.append(row);worker._row12_all.append(dict(request_id=ids[0],row=row))
                return result
            model.compute_logits=compute
            sampler=self.model_runner.sampler
            def before(mod,args,kwargs):
                logits=args[0] if args else kwargs['logits']
                require(len(worker._row12_pending)==1,'ROW12_SAMPLER_REQUEST_ASSOCIATION')
                value,row=worker._row12_pending[0]
                require(logits is value and fingerprint(logits)==row['logits_sha256'],'ROW12_SAMPLER_LOGITS_IDENTITY')
            def after(mod,args,kwargs,result):
                torch.npu.current_stream().synchronize()
                value,row=worker._row12_pending.pop(0);row['sampler_consumed']=True
            hooks=[sampler.register_forward_pre_hook(before,with_kwargs=True),sampler.register_forward_hook(after,with_kwargs=True)]
            class Cleanup:
                def remove(self):
                    require(not worker._row12_pending,'ROW12_SAMPLER_PENDING')
                    require(vars(model).get('compute_logits') is compute,'ROW12_OBSERVER_OWNER_CHANGED')
                    del model.compute_logits
                    for hook in hooks:hook.remove()
            self._v2_canary_hook=Cleanup()
            return dict(rank=self.rank,status='ROW12_CONFIGURED_NOT_QUALIFIED',deployment=actual,admission=admission)
        if action=='canary_start':self._row12_rows=[];return dict(rank=self.rank,status='ARMED')
        if action=='canary_end':
            require(len(self._row12_rows)==4 and not self._row12_pending,'ROW12_FOUR_COMPLETED_STEPS_REQUIRED')
            return dict(rank=self.rank,all_finite=all(r['finite'] for r in self._row12_rows),
                output=dict(logits_observations=self._row12_rows))
        return super().demoncat_persistent_example(action,options)


if __name__=='__main__':
    main('reference_adapters.lm_head_product.LmHeadExtension',os.environ['DCAT_REFERENCE_ROW12'])
