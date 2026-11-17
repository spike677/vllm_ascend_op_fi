"""Fault8 paired contribution-group remapping before real MoE unpermute.

The original dispatcher receives one new indices tensor and matching probs.
No integer offset is added to indices, and permuted hidden values are untouched.
"""
from ._scoped_dispatch import ScopedDispatch
from .core import UnsupportedCall


class RestoreOffsetSession(ScopedDispatch):
    mechanism = '8.restore_offset'

    def __init__(self, model, *, selected_calls=None, admission=None, strict_group_oracle=False, **kwargs):
        super().__init__(model, **kwargs)
        self.private_observations = {}
        self.selected_calls = None if selected_calls is None else frozenset(selected_calls)
        self.admission, self.strict_group_oracle = admission, strict_group_oracle

    def is_selected(self, event):
        if self.admission is not None:
            return bool(self.admission(event))
        return ((event['path'], event['ordinal']) in self.selected_calls
                if self.selected_calls is not None else super().is_selected(event))

    def dispatch(self, func, args, kwargs):
        import torch
        schema = getattr(func, '_schema', None)
        name = getattr(schema, 'name', '')
        if name != 'npu::npu_moe_token_unpermute' or getattr(schema, 'overload_name', ''):
            return func(*args, **kwargs)
        names = ('permuted_tokens','sorted_indices','probs','padded_mode','restore_shape')
        if len(args)>len(names) or set(kwargs)-set(names) or set(names[:len(args)]) & set(kwargs):
            raise UnsupportedCall('Unreviewed MoE unpermute signature')
        values = dict(probs=None,padded_mode=False,restore_shape=None)
        values.update(zip(names,args)); values.update(kwargs)
        hidden, indices, probs = (values.get(k) for k in names[:3])
        event = self.event(str(func))
        with self.diagnostics():
            if (any(not isinstance(x,torch.Tensor) for x in (hidden,indices,probs))
                or hidden.ndim!=2 or indices.ndim!=1 or probs.ndim!=2
                or indices.dtype!=torch.int32 or not hidden.is_floating_point()
                or probs.dtype!=hidden.dtype or hidden.device!=indices.device or hidden.device!=probs.device
                or values['padded_mode'] is not False or values['restore_shape'] is not None
                or not self.simulation and hidden.device.type!='npu'):
                event.update(status='unsupported',reason='UNPADDED_COMPLETE_TOKEN_GROUP_CONTRACT_REQUIRED')
                return func(*args,**kwargs)
            tokens, topk = probs.shape
            if tokens<=1:
                event.update(status='passthrough',reason='SINGLE_TOKEN_HAS_NO_NEXT_GROUP')
                return func(*args,**kwargs)
            if topk<1 or indices.numel()!=tokens*topk or hidden.shape[0]!=tokens*topk:
                raise UnsupportedCall('Partial MoE expert/token groups are not admitted')
            original_i, original_p, original_h = self.cpu(indices), self.cpu(probs), self.cpu(hidden)
            if not torch.equal(original_i.long().sort().values,torch.arange(tokens*topk)):
                event.update(status='unsupported', reason='COMPLETE_CONTRIBUTION_PERMUTATION_REQUIRED',
                    complete_permutation_value_verified=False, native_calls=0,
                    token_groups=tokens, topk=topk, unique_indices=int(original_i.unique().numel()))
                raise UnsupportedCall('Healthy restore indices must cover the complete contribution permutation')
            if not bool(torch.isfinite(original_h).all() and torch.isfinite(original_p).all()):
                raise UnsupportedCall('Healthy MoE input must be finite')
            injecting = self.arm=='fault' and self.is_selected(event)
            preserve_actual_arguments = self.strict_group_oracle and not injecting
            if preserve_actual_arguments:
                # Natural baseline and rejected/exhausted admissions must send
                # the original objects to the real consumer, not equal copies.
                group_cpu = torch.arange(tokens, dtype=torch.long)
                remapped_i, remapped_p = indices, probs
            else:
                groups = torch.arange(tokens,device=indices.device)
                if injecting: groups=(groups+1).clamp_max(tokens-1)
                remapped_i=indices.reshape(tokens,topk).index_select(0,groups).reshape(-1).contiguous()
                remapped_p=probs.index_select(0,groups).contiguous()
                self.temps.extend((groups,remapped_i,remapped_p))
                group_cpu=self.cpu(groups).long()
                if (not torch.equal(self.cpu(remapped_i),original_i.reshape(tokens,topk)[group_cpu].reshape(-1))
                        or not torch.equal(self.cpu(remapped_p),original_p[group_cpu])):
                    raise UnsupportedCall('Paired group transformation failed device readback')
        # This is the only call to the actual selected dispatcher operator.
        result=(func(*args,**kwargs) if preserve_actual_arguments else
                func(**dict(values,sorted_indices=remapped_i,probs=remapped_p)))
        with self.diagnostics():
            actual=self.cpu(result)
            if tuple(actual.shape)!=(tokens,hidden.shape[1]):
                raise UnsupportedCall('Native unpermute output geometry differs')
            if (not torch.equal(self.cpu(indices),original_i) or not torch.equal(self.cpu(probs),original_p)
                    or not torch.equal(self.cpu(hidden),original_h)):
                raise UnsupportedCall('Original MoE inputs changed')
            # The CPU oracle describes the intended grouped consumer. It is
            # auxiliary evidence, never a substitute for the original NPU call.
            unshifted=(original_h.float()[original_i.long()].reshape(tokens,topk,-1)
                       *original_p.float().unsqueeze(-1)).sum(1)
            reference=unshifted[group_cpu]
            max_abs=float((actual.float()-reference).abs().max())
            oracle_exact=torch.equal(actual, reference.to(actual.dtype))
            if self.strict_group_oracle and not oracle_exact:
                expected = reference.to(actual.dtype)
                event.update(status='unsupported', reason='COMPLETE_GROUP_ORACLE_NOT_EXACT',
                    original_execution_verified=True, native_calls=1, token_groups=tokens, topk=topk,
                    output_shape=list(actual.shape), dtype=str(actual.dtype),
                    complete_permutation_value_verified=True, complete_native_output_oracle_exact=False,
                    rounded_oracle_mismatches=int((actual != expected).sum()),
                    rounded_oracle_max_abs=float((actual.float()-expected.float()).abs().max()))
                raise UnsupportedCall('NATURAL_NATIVE_UNPERMUTE_COMPLETE_GROUP_ORACLE_NOT_EXACT')
            event.update(status='applied' if injecting else 'baseline',applied=injecting,
                original_execution_verified=True,native_calls=1,paired_group_readback_exact=True,
                input_storage_unchanged=True,token_groups=tokens,topk=topk,
                source_groups=group_cpu.tolist(),output_shape=list(actual.shape),dtype=str(actual.dtype),
                native_inputs_original_objects=remapped_i is indices and remapped_p is probs,
                native_inputs_original_storage=(remapped_i.untyped_storage()._cdata == indices.untyped_storage()._cdata
                    and remapped_p.untyped_storage()._cdata == probs.untyped_storage()._cdata),
                consumer_operator=str(func),consumer_returned=True,auxiliary_fp32_oracle_max_abs=max_abs,
                boundary='npu_moe_token_unpermute.inputs',
                semantics='PAIRED_TOKEN_GROUP_REMAP_NOT_ELEMENTWISE_INDEX_OFFSET')
            if self.strict_group_oracle:
                event.update(complete_permutation_value_verified=True,
                    complete_native_output_oracle_exact=True, return_readback_exact=True,
                    output_changed=not torch.equal(actual, unshifted.to(actual.dtype)),
                    consumers=[dict(operator=str(func), returned=True, actual_native_input_consumer=True)])
            self.private_observations[(event['path'],event['ordinal'])] = dict(
                inputs=(original_h,original_i,original_p),output=actual,source_groups=group_cpu)
        return result
