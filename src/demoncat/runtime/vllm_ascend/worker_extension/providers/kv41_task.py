"""Ordinary single-site KV41 with explicit healthy calibration and recovery.

Actual root callbacks complete requests. Query never collects, advances or
executes inference. Historical all-target C2/C3 remains a separate provider.
"""
from copy import deepcopy
from .fault41 import Fault41ServingProvider, require
from ._scope import reject_unknown
from ..first_decode_health import FirstDecodeHealth

from fault_injection.request_plan import KV41 as PHASES
_NATIVE=('observe','observe','lossless','fault','off')


class KV41TaskProvider(Fault41ServingProvider):
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        require(self.binding.runtime_only, 'KV41_ORDINARY_TASK_BINDING_REQUIRED')
        self.rows=[]
        self.health=None
        self.first_error=None
        self.recovery_phase=None

    @property
    def phase(self):
        return self.recovery_phase or (PHASES[self.index] if self.index<len(PHASES) else 'complete')

    def _failure(self,error,source):
        self.first_error=self.first_error or repr(error)
        super()._failure(error,source)

    def _wrap_callback(self,kind,callback):
        import threading
        provider,epoch=self.p,self.p.install_epoch
        def actual(*args,**kwargs):
            before,after=kind=='root_pre',kind=='root_post'
            try:
                self._context()
                from .. import postwrite41
                require(self._registered and self.p is provider and provider.install_epoch==epoch
                        and provider.installed and postwrite41._ACTIVE is provider,
                        'KV41_TASK_STALE_OR_FOREIGN_CALLBACK')
                if kind not in ('root_post','attention_post'):
                    require(not self.error,'KV41_TASK_FAILED_CALLBACK_REQUIRES_RECOVERY')
                if before:
                    require(not self.root_active and self.pending is not None,'KV41_TASK_EXPECTED_SERIAL_ROOT_REQUIRED')
                    self.root_active=True;self.root_unwound=False
                    require(next(reversed(args[0]._forward_pre_hooks.values())) is actual,
                            'KV41_TASK_FINAL_ROOT_INPUT_OBSERVER_REQUIRED')
                elif after:
                    require(self.root_active,'KV41_TASK_ROOT_POST_WITHOUT_PRE')
                    require(next(reversed(args[0]._forward_hooks.values())) is actual,
                            'KV41_TASK_FINAL_ROOT_RETURN_OBSERVER_REQUIRED')
                elif provider.current is not None:
                    require(self.root_active,'KV41_TASK_CALLBACK_OUTSIDE_ROOT')
                value=callback(*args,**kwargs)
            except BaseException as error:
                self._failure(error,kind);raise
            finally:
                if after and self.p is provider and provider.install_epoch==epoch and threading.get_ident()==self.thread:
                    self.root_active=False;self.root_unwound=True
            if after and self.pending and provider.pending_run is not None:
                try:self._complete_request()
                except BaseException as error:self._failure(error,'complete_actual_root');raise
            return value
        return actual

    def dispatch(self,action,settings,*,owner,generation):
        if action not in ('advance','finish','recover'):
            return super().dispatch(action,settings,owner=owner,generation=generation)
        self.verify(owner,generation);self._context()
        reject_unknown(settings,('request_id',) if action=='advance' else ())
        try:
            if action=='advance':
                require(self.attempted and not self.stopped and not self.error,'KV41_TASK_ACTIVE_OR_RECOVERY_REQUIRED')
                require(set(settings)=={'request_id'} and self.pending is None,'KV41_TASK_PREVIOUS_REQUEST_NOT_COMPLETED')
                rid=settings['request_id']
                require(isinstance(rid,str) and 0<len(rid)<=256 and rid not in self.seen and len(self.seen)<8,
                        'KV41_TASK_BOUNDED_FRESH_REQUEST_REQUIRED')
                phase=self.phase
                require(phase!='complete','KV41_TASK_PHASES_EXHAUSTED')
                self.seen.add(rid);self.pending=(rid,phase,phase)
                if phase=='health':
                    reference=dict(prefixes=self.p.history[0]['actual_scheduler']['prefixes'],
                                   model_output_sha256=self.p.history[0]['model_output_sha256'])
                    self.health=FirstDecodeHealth(self.p,rid,reference,self._health_complete)
                    self.health.install()
                else:
                    native='off' if self.recovery_phase else _NATIVE[self.index]
                    self.p.expect(native,rid,self.binding.prompt)
            elif action=='finish':
                require(self.complete and self.recovered and not self.error,'KV41_TASK_OFF_AND_HEALTH_REQUIRED')
                self._cleanup()
            elif action=='recover':
                if self.health:
                    if self.health.error:self._failure(RuntimeError(self.health.error),'actual_health')
                    self.health.close()
                    self.health=None
                self._cancel_returned_request()
                self.error=None
                if self.p is not None and self.p.fault_attempted and not self.recovered:
                    self.recovery_phase='off' if self.p.installed else 'health'
                    self.stopped=False
                else:self._cleanup()
        except BaseException as error:
            self._failure(error,'rpc_'+action);raise
        return self.query(owner=owner,generation=generation)

    def _complete_request(self):
        if self.pending is None:return
        require(not self.error and not self.root_active and self.p.current is None,
                'KV41_TASK_ACTUAL_REQUEST_NOT_RETURNED')
        rid,phase,_=self.pending
        require(self.p.pending_run and rid in self.p.pending_run['request_ids'],
                'KV41_TASK_ACTUAL_REQUEST_ID_MISMATCH')
        record=self.p.finish_request()
        interventions=[row for row in record['rows'] if row.get('intervention')]
        effect=None
        guard_fault=phase=='fault' and self.binding.group_context is not None and not self.p.fault_injector()
        if guard_fault:
            guarded=[r['guard_boundary'] for r in record['rows'] if r.get('guard_boundary')]
            require(not interventions and len(guarded)==1 and guarded[0]['unchanged']
                and guarded[0]['native_writes']==0,'TP2_KV41_GUARD_BOUNDARY_REQUIRED')
            effect=deepcopy(guarded[0])
        if phase=='lossless' or phase=='fault' and not guard_fault:
            require(len(interventions)==1,'KV41_TASK_ONE_ACTUAL_TARGET_REQUIRED')
            selected=interventions[0]; effect=deepcopy(selected['intervention'])
            require(effect['original_calls']==1 and effect['consumer_returned']
                    and effect['readback_exact'] and effect['restored'] and effect['released'],
                    'KV41_TASK_NATIVE_CONSUMER_AND_RECOVERY_REQUIRED')
            if phase=='fault':require(effect['native_applied'] and effect['changed_elements']>0,
                                      'KV41_TASK_ACTUAL_NATIVE_EFFECT_REQUIRED')
        self.rows.append(dict(phase=phase,request_id=rid,model_output_sha256=record['model_output_sha256'],
            hidden_changed=not record['exact_healthy_output'],exact_healthy_output=record['exact_healthy_output'],
            actual_calls=len(record['rows']),effect=effect,new_fault_hits=record['new_fault_hits'],
            recovery_confirmed=record['recovery_confirmed']))
        self.pending=None
        if phase=='off':
            self.p.close()
            require(not self.p.handles and not self.p.patches and self.p.lease is None and not self.p.installed,
                    'KV41_TASK_OFF_RESOURCES_REMAIN')
        if self.recovery_phase:self.recovery_phase='health'
        else:self.index+=1

    def _health_complete(self,row):
        self._context()
        require(self.pending and row['request_id']==self.pending[0] and self.phase=='health',
                'KV41_TASK_HEALTH_OWNER_OR_REQUEST_CHANGED')
        self.rows.append(deepcopy(row));self.pending=None
        self.recovered=self.complete=True
        self.recovery_phase=None;self.index=len(PHASES)

    def _cleanup(self):
        if self.health:self.health.close()
        require(not self.root_active and (self.p is None or self.p.current is None),
                'KV41_TASK_ROOT_STILL_ACTIVE')
        require(self.p is None or not self.p.fault_attempted or self.recovered,
                'KV41_TASK_REAL_OFF_AND_HEALTH_REQUIRED')
        self._cancel_returned_request()
        if self.p is not None:self.p.close()
        self.stopped=True

    def query(self,*,owner,generation):
        state=super().query(owner=owner,generation=generation)
        health=self.health.readback() if self.health else dict(owned_hooks=0,active=False,error=None)
        error=self.error or health['error']
        current=self.rows[-1] if self.rows and self.pending is None else None
        fault=next((row for row in reversed(self.rows) if row['phase']=='fault'),None)
        selected=None if self.p is None or self.p.target is None else dict(
            module_path=self.p.target[0],selector='FIRST_ACTUAL_STABLE_WRITER_CACHE_CONSUMER',
            target_key=deepcopy(self.p.target),all_matching_calls_covered=False)
        evidence=dict(actual_requests=deepcopy(self.rows),selection=selected,
            complete=self.complete,matched_off_and_health_confirmed=self.recovered,
            last_fault_effect=deepcopy(fault['effect']) if fault else None,
            first_error=self.first_error,failures=deepcopy(self.failures),
            logits_observation='NOT_CAPTURED_BY_PROVIDER_ROOT_HASH_ONLY')
        state.update(ordinary_task=True,simulation=self.binding.simulation,
            owned_hooks=state['owned_hooks']+health['owned_hooks'],
            inflight=int(bool(state['inflight'] or health['active'])),
            replay_failed=bool(error and not self.stopped),
            recovery_validation_pending=bool(self.p and self.p.fault_attempted and not self.recovered),
            current_request_phase=self.phase if self.pending else current['phase'] if current else None,
            current_request_effect=deepcopy(current.get('effect')) if current else None,
            last_fault_effect=deepcopy(fault['effect']) if fault else None,
            serving=dict(next_phase=self.phase,pending_request=self.pending,complete=self.complete,
                         recovered=self.recovered,error=error),
            evidence=evidence,provider=None,coverage=None,
            C1=False,C2=False,C3=False,E5=False,default_dump=False,default_msprobe=False,
            scope='ONE_ACTUAL_KV_TAIL_FIRST_DECODE_WITH_HEALTHY_CALIBRATION',
            next_task='NEW_RUNTIME_OWNER_AND_PROVIDER_REQUIRED')
        return state
