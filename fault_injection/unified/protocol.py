"""Uniform JSON receipts and local typed execution inputs."""
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Request:
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    request_id: str = 'local-request'
    worker: str = 'local-cpu'
    rank: int = 0
    device: int = 0

    def __post_init__(self):
        if type(self.args) is not tuple or type(self.kwargs) is not dict:
            raise ValueError('REQUEST_ARGS_TUPLE_AND_KWARGS_DICT_REQUIRED')
        if any(type(v) is not str or not v for v in (self.request_id,self.worker)):
            raise ValueError('REQUEST_AND_WORKER_ID_REQUIRED')
        if any(type(v) is not int or v<0 for v in (self.rank,self.device)):
            raise ValueError('NONNEGATIVE_RANK_AND_DEVICE_REQUIRED')


@dataclass(frozen=True)
class Result:
    operation: str
    ok: bool
    code: str
    state: str
    mechanism: str
    owner: str
    generation: int
    data: dict = field(default_factory=dict)
    message: str = ''
    schema: str = 'demoncat.operation_result.v1'
    qualification_granted: bool = False

    def to_dict(self):return asdict(self)


class AdapterError(RuntimeError):
    def __init__(self,code,message):
        super().__init__(message);self.code=code


@runtime_checkable
class AdapterProtocol(Protocol):
    contract_version: str
    capabilities: dict
    lifecycle: tuple
    evidence_schema: str

    def validate_config(self,config): ...
    def configure(self,config): ...
    def discover(self,model,request): ...
    def inspect(self): ...
    def arm(self,admit): ...
    def observe(self): ...
    def inject(self): ...
    def verify(self): ...
    def disarm(self): ...
    def restore(self): ...
    def audit(self): ...
    def status(self): ...
    def cleanup(self): ...
