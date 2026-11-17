"""Mechanism-specific capabilities behind four execution-family contracts.

The first two guarded framework routes execute. Other routes preserve their
legacy backend but explicitly refuse unified execution until its bridge exists.
"""
from copy import deepcopy

from fault_registry import definitions, resolve
from .config import example_config, validate_config
from .protocol import AdapterError, Request


LIFECYCLE=('discover','inspect','configure','arm','observe','inject','verify','disarm','restore','audit','status','cleanup')


class MechanismAdapter:
    contract_version='1'
    lifecycle=LIFECYCLE
    evidence_schema='demoncat.unified_evidence.v1'

    def __init__(self,mechanism,config=None):
        self.definition=resolve(mechanism)
        self.config=validate_config(example_config(mechanism) if config is None else config,mechanism)
        self.capabilities=dict(execution=self.definition['adapter_status'],
            reason=self.definition['adapter_reason'],backends=list(self.definition['backends']),
            qualification_granted=False,trigger_modes=['first','nth','every_nth','every_match','probability','condition'])
        self.capabilities['execution_trigger_modes']=(list(self.capabilities['trigger_modes'])
            if self.definition['adapter_status']=='CPU_EXECUTABLE' else [])

    def validate_config(self,config):return validate_config(config,self.definition['legacy_id'])
    def configure(self,config):
        if getattr(self,'model',None) is not None or self.status()['resources_acquired']:
            raise AdapterError('INVALID_STATE','Configure before binding a model or acquiring resources')
        self.config=self.validate_config(config)
        return dict(config=deepcopy(self.config),configured=True)
    def _unavailable(self):
        raise AdapterError('NOT_IMPLEMENTED',self.definition['adapter_reason'])
    def discover(self,model,request):return self._unavailable()
    def inspect(self):return deepcopy(self.definition)
    def arm(self,admit):return self._unavailable()
    def observe(self):return self._unavailable()
    def inject(self):return self._unavailable()
    def verify(self):return self._unavailable()
    def disarm(self):return dict(admission_stopped=True,resources_acquired=False)
    def restore(self):return dict(restored=True,resources_acquired=False)
    def audit(self):return dict(schema=self.evidence_schema,status='NO_EXECUTION',qualification_granted=False)
    def status(self):return dict(capabilities=deepcopy(self.capabilities),resources_acquired=False)
    def cleanup(self):return dict(cleanup_confirmed=True,resources_acquired=False)


class NativeAdapter(MechanismAdapter):pass
class TilingAdapter(MechanismAdapter):pass
class KernelAdapter(MechanismAdapter):pass


def _walk(value):
    import torch
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,dict):
        for item in value.values():yield from _walk(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from _walk(item)


def _snapshot(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {k:_snapshot(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_snapshot(v) for v in value)
    return deepcopy(value)


def _compare(left,right):
    import torch
    if isinstance(left,torch.Tensor) and isinstance(right,torch.Tensor):
        if left.shape!=right.shape or left.dtype!=right.dtype:return False
        if left.is_floating_point() or left.is_complex():
            return bool(torch.allclose(left,right,rtol=0,atol=0,equal_nan=True))
        return bool(torch.equal(left,right))
    if type(left) is not type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(_compare(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return len(left)==len(right) and all(_compare(a,b) for a,b in zip(left,right))
    return left==right


class FrameworkAdapter(MechanismAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model=self.request=self.lease=None
        self.enabled=False
        self.phase=None
        self.records={}
        self.values={}
        self.output=None
        self.hits=0
        self.verification=None
        self.recovery=None

    def discover(self,model,request):
        if self.definition['adapter_status']!='CPU_EXECUTABLE':return self._unavailable()
        from fault_injection.module_faults import discover_modules
        if not isinstance(request,Request) or not callable(getattr(model,'named_modules',None)):
            raise AdapterError('INVALID_INPUT','Actual model and typed Request required')
        if self.config['execution']['cann']!='cpu':
            raise AdapterError('NOT_IMPLEMENTED','This unified module bridge is CPU-scoped; use existing qualified NPU routes.')
        if self.config['execution']['operator']!=self.definition['operator_paths'][0]['family']:
            raise AdapterError('UNSUPPORTED_OPERATOR','Configured operator does not match this module adapter')
        if not all(self.config['evidence'].values()):
            raise AdapterError('UNSUPPORTED_CONFIGURATION','This adapter requires independent B/F/OFF capture')
        tensors=list(model.parameters())+list(model.buffers())+list(_walk((request.args,request.kwargs)))
        if any(t.device.type!='cpu' for t in tensors):
            raise AdapterError('UNSUPPORTED_ENVIRONMENT','CPU bridge must not execute NPU or CUDA tensors')
        candidates=discover_modules(model,self.definition['legacy_id'])
        if not candidates:raise AdapterError('NO_TARGET','No structurally supported module')
        if model.training:raise AdapterError('UNSUPPORTED_MODEL','Inference eval mode required')
        self.model,self.request=model,request
        return dict(candidates=[p for p,_ in candidates],status='STRUCTURAL_CANDIDATES',actual_calls_observed=False)

    def _run(self,phase):
        if self.model is None:raise AdapterError('INVALID_STATE','discover must bind model/request first')
        from fault_injection.tensor_observer import NaturalTensorObserver
        import torch
        observer=NaturalTensorObserver(self.model,record_all_operators=True)
        self.values.pop(phase,None)
        if phase=='fault':self.verification=None
        self.phase=phase
        try:
            with observer,torch.no_grad():
                self.output=self.model(*self.request.args,**self.request.kwargs)
            self.values[phase]=_snapshot(self.output)
            self.records.setdefault(phase,[]).append(observer.report())
        except Exception:
            self.records.setdefault(phase,[]).append(observer.report())
            raise
        finally:self.phase=None
        return dict(phase=phase,observation_complete=self.records[phase][-1]['complete'],
                    observed_calls=len(self.records[phase][-1]['events']),npu_calls=0)

    def observe(self):return self._run('baseline' if 'baseline' not in self.values else 'observe')

    def arm(self,admit):
        if self.definition['adapter_status']!='CPU_EXECUTABLE':return self._unavailable()
        if self.model is None:raise AdapterError('INVALID_STATE','discover required')
        if self.lease is not None:raise AdapterError('RECOVERY_REQUIRED','Prior lease still held')
        if 'baseline' not in self.values:self._run('baseline')
        from fault_injection.module_faults import ModuleFaultSession
        def eligible(event):
            if not self.enabled or self.phase!='fault':return False
            selected=admit(event,self.request)
            if selected:self.hits+=1
            return selected
        self.lease=ModuleFaultSession(self.model,self.definition['legacy_id'],_admit_call=eligible)
        self.lease.__enter__()
        self.enabled=True
        return dict(armed=True,resources_acquired=True,scope='CPU_MODULE_PROXY')

    def inject(self):
        if not self.enabled:raise AdapterError('INVALID_STATE','arm required')
        before=self.hits
        result=self._run('fault')
        result.update(selected_calls=self.hits-before,total_selected_calls=self.hits)
        return result

    def verify(self):
        if not {'baseline','fault'}<=self.values.keys():
            if self.verification is not None:return deepcopy(self.verification)
            raise AdapterError('MISSING_EVIDENCE','Baseline and fault outputs required')
        self.verification=dict(effect_observed=not _compare(self.values['baseline'],self.values['fault']),
                    scope='independent returned CPU tensor comparison; no native first-difference claim',
                    comparison='exact with equal NaN; no tolerance-based NPU acceptance',selected_calls=self.hits)
        return deepcopy(self.verification)

    def disarm(self):
        self.enabled=False
        return dict(admission_stopped=True,inflight=int(self.phase is not None))

    def restore(self):
        self.enabled=False
        if self.phase is not None:raise AdapterError('BUSY','Actual forward still inflight')
        if self.lease is not None:
            self.lease.close()
            if not self.lease.closed:raise AdapterError('RECOVERY_FAILED','Lease release unconfirmed')
        if self.model is not None and 'baseline' in self.values:
            self._run('off')
            equal=_compare(self.values['baseline'],self.values['off'])
            if self.config['lifecycle']['require_healthy_replay'] and not equal:
                raise AdapterError('RECOVERY_FAILED','Healthy replay differs; retain session and retry')
        else:equal=None
        self.lease=None
        self.recovery=dict(restored=True,healthy_replay_equal=equal,resources_acquired=False)
        return deepcopy(self.recovery)

    def audit(self):
        return dict(schema=self.evidence_schema,phases=deepcopy(self.records),
                    baseline_fault=self.verify() if 'fault' in self.values or self.verification is not None else None,
                    recovery=deepcopy(self.recovery),
                    raw_msprobe_statistics=False,qualification_granted=False)

    def status(self):
        return dict(capabilities=deepcopy(self.capabilities),resources_acquired=self.lease is not None,
                    admission_enabled=self.enabled,inflight=int(self.phase is not None),selected_calls=self.hits)

    def cleanup(self):
        if self.lease is not None:raise AdapterError('RECOVERY_REQUIRED','restore must release owned lease first')
        if 'fault' in self.values:self.verify()
        self.enabled=False;self.model=self.request=None
        self.values.clear();self.output=None
        return dict(cleanup_confirmed=True,resources_acquired=False)


class AdapterRegistry:
    def resolve(self,mechanism,config=None):
        row=resolve(mechanism)
        config=validate_config(example_config(mechanism) if config is None else config,mechanism)
        selected=config['execution']['backend']
        cls={'framework':FrameworkAdapter,'native':NativeAdapter,'tiling':TilingAdapter,'device_kernel':KernelAdapter}[selected]
        return cls(row['legacy_id'],config)

    def definitions(self):return definitions()


registry=AdapterRegistry()
