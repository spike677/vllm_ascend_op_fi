"""Bounded Fault22 input variant for the actual fused RoPE reader.

Only the final legal position is replaced with the first legal position in a
new positions allocation. The original fused implementation executes once.
This is not the older lookup-output proxy and does not reproduce uint32 wrap.
No C2/C3 or E5 acceptance is issued by this local adapter.
"""
from collections import Counter

from ._scoped_dispatch import ScopedDispatch
from .conv3d_natural import tensor_identity
from .core import UnsupportedCall
from .rope_census import NaturalRoPETrace, same_view
from .rope_sites import ModuleSites
from .tensor_observer import tensors


class FusedWrongRowSession(ScopedDispatch):
    mechanism='22.wrong_row.fused_positions'

    def __init__(self,model,*,admission=None,references=None,_runtime=None,**kwargs):
        super().__init__(model,**kwargs)
        if _runtime is None:
            if self.simulation:raise UnsupportedCall('EXPLICIT_CPU_ROPE_FIXTURE_REQUIRED')
            _runtime=NaturalRoPETrace.from_runtime(model)
        if bool(_runtime.simulation)!=self.simulation:
            raise UnsupportedCall('ROPE_RUNTIME_SIMULATION_MISMATCH')
        if _runtime.outer_operator is None:raise UnsupportedCall('ACTUAL_OPAQUE_ROPE_BOUNDARY_REQUIRED')
        self.runtime=_runtime;self.sites=ModuleSites(model)
        self.owners=self.sites.owner_sites(_runtime.owners)
        self.admission=admission;self.references={} if references is None else dict(references)
        self.healthy_calls={};self.site_counts=Counter();self.records=[]
        self.request_id=None;self.positions=None;self.expected_positions=None

    def bind_request(self,request_id,positions,expected_positions):
        if self.stack or not request_id or not isinstance(expected_positions,list):
            raise UnsupportedCall('IDLE_ACTUAL_SCHEDULER_REQUEST_REQUIRED')
        self.request_id=request_id;self.positions=positions;self.expected_positions=list(expected_positions)

    def _before(self,path,module):
        def enter(actual,args):
            actual_path=self.sites.resolve(actual,self.stack)
            if actual_path is None:return
            if actual is self.model:self.site_counts.clear()
            super(FusedWrongRowSession,self)._before(actual_path,module)(actual,args)
        return enter

    def dispatch(self,func,args,kwargs):
        import torch
        schema=getattr(func,'_schema',None)
        if getattr(schema,'name',None)!=self.runtime.outer_operator:
            return func(*args,**kwargs)
        path,module=self.stack[-1]
        if self.owners.get(path) is not module or self.request_id is None:
            raise UnsupportedCall('ACTUAL_LIVE_ROPE_SITE_AND_REQUEST_REQUIRED')
        guard=self.runtime._guard_by_owner.get(id(module))
        if guard is not None:guard.verify()
        values={arg.name:arg.default_value for arg in schema.arguments}
        values.update(zip((arg.name for arg in schema.arguments),args));values.update(kwargs)
        event=self.event(str(func));ordinal=self.site_counts[path];self.site_counts[path]+=1
        event.update(site_ordinal=ordinal,request_id=self.request_id)
        p,q,k,cache=(values[name] for name in ('positions','query','key','cos_sin_cache'))
        with self.diagnostics():
            if (not same_view(p,self.positions) or cache is not module.cos_sin_cache
                or p.ndim!=1 or p.dtype not in (torch.int32,torch.int64) or not p.is_contiguous()
                or q.ndim!=2 or k.ndim!=2 or q.shape[0]!=p.numel() or k.shape[0]!=p.numel()
                or not q.is_contiguous() or not k.is_contiguous() or cache.ndim!=2
                or q.dtype!=k.dtype or q.dtype!=cache.dtype or q.device!=k.device or q.device!=p.device
                or q.device!=cache.device or not self.simulation and q.device.type!='npu'
                or values.get('offsets') is not None or values['rotary_dim']!=values['head_size']):
                raise UnsupportedCall('REVIEWED_FULL_DIM_CONTIGUOUS_FUSED_ROPE_INPUTS_REQUIRED')
            before_p=self.cpu(p);before_cache=self.cpu(cache);before_q=self.cpu(q);before_k=self.cpu(k)
            positions=before_p.tolist()
            if (positions!=self.expected_positions or len(positions)<2 or positions[0]==positions[-1]
                or min(positions)<0 or max(positions)>=cache.shape[0]
                or torch.equal(before_cache[positions[0]],before_cache[positions[-1]])):
                raise UnsupportedCall('DISTINCT_ACTUAL_LEGAL_ROPE_ROWS_REQUIRED')
            selected=self.arm=='fault' and bool(self.admission(event) if self.admission else self.is_selected(event))
            reference=self.references.get((path,ordinal))
            if selected and reference is None:raise UnsupportedCall('HEALTHY_SITE_REFERENCE_REQUIRED')
            if selected and (not torch.equal(before_q[:-1],reference['query_before'][:-1])
                    or not torch.equal(before_k[:-1],reference['key_before'][:-1])
                    or tensor_identity(before_cache)!=reference['cache']):
                raise UnsupportedCall('NON_TARGET_INPUT_OR_CACHE_CHANGED_FROM_HEALTHY_SITE')
            native=dict(values)
            if selected:
                replacement=p.clone();replacement[-1].copy_(p[0]);self.synchronize()
                expected=before_p.clone();expected[-1]=expected[0]
                if (replacement.untyped_storage()._cdata==p.untyped_storage()._cdata
                        or not torch.equal(self.cpu(replacement),expected)):
                    raise UnsupportedCall('INDEPENDENT_WRONG_POSITION_READBACK_REQUIRED')
                self.temps.append(replacement);native['positions']=replacement
        output=func(**native)
        with self.diagnostics():
            outputs=tuple(tensors(output))
            if len(outputs)!=2 or not torch.equal(self.cpu(p),before_p) or not torch.equal(self.cpu(cache),before_cache):
                raise UnsupportedCall('ORIGINAL_ROPE_POSITIONS_OR_CACHE_CHANGED')
            after_q,after_k=(self.cpu(t) for t in outputs)
            non_target=(not selected or torch.equal(after_q[:-1],reference['query_after'][:-1])
                        and torch.equal(after_k[:-1],reference['key_after'][:-1]))
            if not non_target:raise UnsupportedCall('FUSED_ROPE_NON_TARGET_OUTPUT_CHANGED')
            if not selected:
                self.healthy_calls[(path,ordinal)]=dict(query_before=before_q,key_before=before_k,
                    query_after=after_q,key_after=after_k,cache=tensor_identity(before_cache))
            event.update(status='applied' if selected else 'baseline',applied=selected,native_calls=1,
                boundary=str(func)+'.positions',semantics='LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW',
                correct_source_row=positions[-1],replacement_source_row=positions[0],target_token=len(positions)-1,
                positions_unchanged=True,cache_unchanged=True,non_target_output_unchanged=non_target,
                actual_qk_before=[tensor_identity(before_q),tensor_identity(before_k)],
                actual_qk_after=[tensor_identity(after_q),tensor_identity(after_k)],
                same_complete_inputs_as_healthy=bool(selected and torch.equal(before_q,reference['query_before'])
                    and torch.equal(before_k,reference['key_before'])),
                changed_from_healthy=bool(selected and (not torch.equal(after_q,reference['query_after'])
                    or not torch.equal(after_k,reference['key_after']))),
                propagated_inputs_possible=bool(selected and (not torch.equal(before_q,reference['query_before'])
                    or not torch.equal(before_k,reference['key_before']))),
                consumer_proof='REQUIRES_INDEPENDENT_NATURAL_ROPE_TRACE',
                final_launch_proof='REQUIRES_INDEPENDENT_NATURAL_ROPE_TRACE')
        return output

    def close(self):
        super().close()
        self.positions=None;self.request_id=None;self.expected_positions=None

    def report(self):
        result=super().report()
        result.update(variant='LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW',
            original_lookup_output_proxy=False,integer_overflow_reproduced=False,
            standard_worker_qualified=False,c2=False,c3=False,
            structural_sites=list(self.owners),native_launch_qualification=False,
            limits=['Runtime observer and actual group recovery are mandatory independent evidence.',
                    'Changed later inputs are propagation; comparisons are not isolated causal trials.'])
        return result
