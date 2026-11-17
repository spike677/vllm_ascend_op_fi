"""One owner-thread state machine; errors never discard recovery resources."""
from copy import deepcopy
import os
import threading
import uuid

from .adapters import registry
from .protocol import AdapterError, Request, Result
from .trigger import CallIdentity, TriggerEngine


class UnifiedSession:
    def __init__(self,config):
        self._config=deepcopy(config)
        self.adapter=registry.resolve(config['mechanism']['id'],config)
        self.owner=uuid.uuid4().hex
        self.generation=1
        self.owner_thread=threading.get_ident()
        self.state='CONFIGURED'
        self.trigger=TriggerEngine(config)
        self._lock=threading.Lock()
        self._call_sequence=0
        self.history=[]

    @property
    def config(self):return deepcopy(self._config)

    @property
    def output(self):return getattr(self.adapter,'output',None)

    def _result(self,operation,ok=True,code='OK',data=None,message=''):
        return Result(operation,ok,code,self.state,self._config['mechanism']['id'],self.owner,
                      self.generation,deepcopy(data or {}),message)

    def _operation(self,name,allowed,fn,next_state=None):
        if threading.get_ident()!=self.owner_thread:
            return self._result(name,False,'OWNER_MISMATCH',message='Use the configuring owner thread')
        if not self._lock.acquire(blocking=False):
            return self._result(name,False,'BUSY',message='An owned operation or forward is inflight')
        try:
            if self.state not in allowed:
                return self._result(name,False,'INVALID_STATE',message='Operation is unavailable in '+self.state)
            before=self.state
            try:
                data=fn()
                if next_state:self.state=next_state(data) if callable(next_state) else next_state
                result=self._result(name,data=data)
            except Exception as error:
                code=error.code if isinstance(error,AdapterError) else 'EXECUTION_ERROR'
                if name=='restore':self.state='RECOVERY_FAILED'
                elif name in ('arm','inject') and code not in ('NOT_IMPLEMENTED','INVALID_STATE','NO_TARGET'):
                    self.state='ERROR'
                result=self._result(name,False,code,message=str(error))
            self.history.append(dict(operation=name,before=before,after=self.state,code=result.code))
            return result
        finally:self._lock.release()

    def discover(self,model,request):
        return self._operation('discover',{'CONFIGURED'},lambda:self.adapter.discover(model,request),'DISCOVERED')
    def inspect(self):return self._operation('inspect',{'CONFIGURED','DISCOVERED','RESTORED'},self.adapter.inspect)
    def configure(self,config):
        from .config import validate_config, ConfigError
        def update():
            try:validated=validate_config(config,self._config['mechanism']['id'])
            except ConfigError as error:raise AdapterError(error.code,str(error)) from error
            adapter=registry.resolve(validated['mechanism']['id'],validated)
            trigger=TriggerEngine(validated)
            self.adapter,self.trigger,self._config=adapter,trigger,validated
            self.generation+=1
            return dict(config=self.config)
        return self._operation('configure',{'CONFIGURED'},update)

    def _admit(self,event,request):
        self._call_sequence+=1
        identity=CallIdentity(str(self._call_sequence),os.getpid(),request.worker,request.rank,request.device,
                              request.request_id,self._config['execution']['operator'],event['path'])
        return self.trigger.decide(identity,eligible=True).selected

    def arm(self):return self._operation('arm',{'DISCOVERED','CONFIGURED'},lambda:self.adapter.arm(self._admit),'ARMED')
    def observe(self):return self._operation('observe',{'DISCOVERED','ARMED','HIT','OFF','RESTORED'},self.adapter.observe)
    def inject(self):return self._operation('inject',{'ARMED','HIT'},self.adapter.inject,
                                           lambda d:'HIT' if d.get('selected_calls',0) else self.state)
    def verify(self):return self._operation('verify',{'HIT','ARMED','OFF','RESTORED'},self.adapter.verify)
    def disarm(self):
        def stop():
            self.state='DRAIN'
            value=self.adapter.disarm()
            if value.get('inflight',0):raise AdapterError('BUSY','Owned calls have not drained')
            return value
        return self._operation('disarm',{'ARMED','HIT','ERROR','DRAIN','OFF'},stop,'OFF')
    def restore(self):return self._operation('restore',{'OFF','ERROR','RECOVERY_FAILED','DISCOVERED'},self.adapter.restore,'RESTORED')
    def audit(self):return self._operation('audit',{'CONFIGURED','DISCOVERED','ARMED','HIT','OFF','RESTORED','RECOVERY_FAILED','CLOSED'},self.adapter.audit)
    def status(self):
        return self._operation('status',{'CONFIGURED','DISCOVERED','ARMED','HIT','DRAIN','OFF','RESTORED','ERROR','RECOVERY_FAILED','CLOSED'},
            lambda:dict(adapter=self.adapter.status(),trigger=self.trigger.status(),history=deepcopy(self.history)))
    def cleanup(self):return self._operation('cleanup',{'CONFIGURED','DISCOVERED','RESTORED','CLOSED'},self.adapter.cleanup,'CLOSED')
