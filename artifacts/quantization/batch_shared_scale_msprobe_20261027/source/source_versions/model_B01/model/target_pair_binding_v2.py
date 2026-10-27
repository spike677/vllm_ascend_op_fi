"""Target-only calibration with unchanged native healthy observation.

The original native producer still runs once. Healthy reference calibration,
API restoration, one producer, exact consumer pair identity, and unchanged
upstream input retain the old binder's checks. No installation on import.
"""


def make_binding(base):
    class TargetOnlyBinding(base):
        def __init__(self,*args,experiment_mode='healthy',**kwargs):
            if experiment_mode not in ('healthy','per_token','shared'):
                raise ValueError('unsupported experiment mode')
            self.experiment_mode=experiment_mode
            super().__init__(*args,**kwargs)

        def _produce(self,x,*args,**kwargs):
            state=self._state_for_thread()
            if state is None:
                return self._producer(x,*args,**kwargs)
            import torch
            from ascend_fi.quantization.batch_scale_core import compute_pair
            from ascend_fi.quantization.batch_binding import BindingError
            from ascend_fi.quantization.batch_binding import NativePairBinding
            event=state['event']
            requested_mode=event['mode']
            old_observer=self.observer
            original_producer=self._producer
            captured=[]
            def capture_native(value,*a,**k):
                pair=original_producer(value,*a,**k)
                captured.append(pair)
                return pair
            event['mode']='healthy'
            self.observer=None
            self._producer=capture_native
            try:
                try:
                    NativePairBinding._produce(self,x,*args,**kwargs)
                except BindingError as error:
                    # The base has already validated native shape/dtype/device,
                    # finite X/scales, signature and unchanged input snapshot.
                    # Only this specific reference-comparison failure is
                    # converted into evidence for an unchanged native run.
                    if str(error)!='HEALTHY_REPLAY_DOES_NOT_MATCH_NATIVE':
                        raise
                    event['full_reference_calibration_failed']=True
            finally:
                event['mode']=requested_mode
                self.observer=old_observer
                self._producer=original_producer
            if len(captured)!=1:
                raise BindingError('EXACTLY_ONE_NATIVE_PRODUCER_REQUIRED')
            native_q,native_s=captured[0]
            valid=torch.tensor(state['rows'].valid_rows,device=x.device,dtype=torch.bool)
            target=torch.tensor([ok and rid==event['target_request_id']
                                 for ok,rid in zip(state['rows'].valid_rows,state['rows'].request_ids)],
                                device=x.device,dtype=torch.bool)
            if int(target.sum().item())!=1:
                raise BindingError('EXACTLY_ONE_REAL_DECODE_TARGET_REQUIRED')
            reference_pair=compute_pair(x,shared=False,valid_rows=list(state['rows'].valid_rows),reference=self.reference)
            nonzero=x.detach().abs().amax(dim=-1)>0
            rows=[]
            for i,rid in enumerate(state['rows'].request_ids):
                q_exact=bool(torch.equal(native_q[i],reference_pair.q[i]))
                scale_close=bool(torch.allclose(native_s[i],reference_pair.scale[i],rtol=self.scale_rtol,atol=0)) if nonzero[i].item() else True
                rows.append({'row':i,'request_id':rid,'is_target':bool(target[i].item()),'q_exact':q_exact,
                    'q_mismatch_count':int((native_q[i]!=reference_pair.q[i]).sum().item()),
                    'q_max_abs_delta':int((native_q[i].to(torch.int16)-reference_pair.q[i].to(torch.int16)).abs().max().item()),
                    'scale_close':scale_close,'scale_exact':bool(torch.equal(native_s[i],reference_pair.scale[i])),
                    'scale_abs_delta':float((native_s[i]-reference_pair.scale[i]).abs().item())})
            target_calibration=next(r for r in rows if r['is_target'])
            target_valid=target_calibration['q_exact'] and target_calibration['scale_close']
            event['calibration']={'q_exact':all(r['q_exact'] for r in rows),'scale_close':all(r['scale_close'] for r in rows),
                'scale_rtol':self.scale_rtol,'scale_atol':0,'rows':rows,
                'max_scale_abs_delta':max(r['scale_abs_delta'] for r in rows),
                'zero_row_policy':'native Q and scale preserved on zero target'}
            event['target_calibration']={**target_calibration,'qualified':target_valid,'scope':'target row only',
                                         'scale_rtol':self.scale_rtol,'scale_atol':0}
            event['calibration_domain_note']='Only P0 is replaced; all actual peer rows still participate in batch absmax and retain native Q/scale.'
            if self.experiment_mode!='healthy' and not target_valid:
                raise BindingError('TARGET_REFERENCE_CALIBRATION_FAILED_NO_INJECTION')
            rowmax=x.detach().float().abs().amax(dim=-1)
            external=bool((rowmax[valid].max()>rowmax[target].max()).item())
            q,scale=native_q,native_s
            applied=False
            if self.experiment_mode=='per_token':
                ref=compute_pair(x,shared=False,valid_rows=list(state['rows'].valid_rows),reference=self.reference)
                # Zero rows retain the native zero-scale convention.
                replace=target & (rowmax>0)
                q=torch.where(replace[:,None],ref.q,native_q)
                scale=torch.where(replace,ref.scale,native_s)
                applied=True
            elif self.experiment_mode=='shared' and external:
                ref=compute_pair(x,shared=True,valid_rows=list(state['rows'].valid_rows),reference=self.reference)
                q=torch.where(target[:,None],ref.q,native_q)
                scale=torch.where(target,ref.scale,native_s)
                applied=True
            peer=valid & ~target
            preserved=torch.equal(q[peer],native_q[peer]) and torch.equal(scale[peer],native_s[peer])
            if not preserved:
                raise BindingError('NON_TARGET_PAIR_CHANGED')
            state['pair']=(q,scale)
            state['pair_snapshot']=(q.detach().clone(),scale.detach().clone())
            self._record_effect(x,native_q,native_s,q,scale,valid,state)
            a=x.detach().float()[target]
            native_dequant=native_q[target].float()*native_s[target,None]
            returned_dequant=q[target].float()*scale[target,None]
            native_error=(native_dequant-a).square().mean()
            returned_error=(returned_dequant-a).square().mean()
            energy=a.square().mean()
            target_metrics={'changed_codes':int((q[target]!=native_q[target]).sum().item()),
                            'changed_dequant_values':int((returned_dequant!=native_dequant).sum().item()),
                            'native_mse':float(native_error.item()),'returned_mse':float(returned_error.item()),
                            'native_nmse':float((native_error/energy).item()) if energy.item()>0 else None,
                            'returned_nmse':float((returned_error/energy).item()) if energy.item()>0 else None,
                            'native_zero_ratio':float((native_q[target]==0).float().mean().item()),
                            'returned_zero_ratio':float((q[target]==0).float().mean().item()),
                            'native_saturation_ratio':float((native_q[target].to(torch.int16).abs()>=127).float().mean().item()),
                            'returned_saturation_ratio':float((q[target].to(torch.int16).abs()>=127).float().mean().item()),
                            'saturation_definition':'abs(INT16(code)) >= 127; includes -128',
                            'target_q_native_exact':bool(torch.equal(q[target],native_q[target])),
                            'target_scale_native_exact':bool(torch.equal(scale[target],native_s[target])),
                            'scale_ratio':float((scale[target]/native_s[target]).item()) if native_s[target].item()>0 else None}
            event.update(experiment_mode=self.experiment_mode,substitution_applied=applied,
                         target_metrics=target_metrics,
                         replacement_scope='one target row only; rank-local feature domain',
                         non_target_pair_preserved=bool(preserved),
                         skipped_reason='NO_EXTERNAL_RANGE_INCREASE' if self.experiment_mode=='shared' and not external else None,
                         target_row_indices=target.nonzero().flatten().cpu().tolist(),
                         artificial_multiplier=1.0)
            self._observe('pair',state,input_x=x,native_q=native_q,native_scale=native_s,
                          returned_q=q,returned_scale=scale,reference_q=reference_pair.q,reference_scale=reference_pair.scale)
            return q,scale
    return TargetOnlyBinding
