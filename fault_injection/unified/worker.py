"""Local Worker control; tensor arguments stay in the actual worker process."""
from . import configure
from .protocol import Request


class WorkerControl:
    def __init__(self,model,mechanism,config):
        self.model=model
        self.session=configure(mechanism,config)

    def discover(self,request):
        return self.session.discover(self.model,request)

    def dispatch(self,operation):
        if operation not in ('inspect','arm','observe','inject','verify','disarm','restore','audit','status','cleanup'):
            return self.session._result(operation,False,'UNKNOWN_OPERATION').to_dict()
        return getattr(self.session,operation)().to_dict()
