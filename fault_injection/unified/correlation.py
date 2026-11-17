"""Typed invocation correlations. Addresses and timestamp proximity are not joins."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import threading


@dataclass(frozen=True)
class InvocationEvent:
    event_id: str
    parent_id: str | None
    kind: str
    process: int
    worker: str
    rank: int
    device: int
    thread: int
    request_id: str
    generation: int
    source: str

    def __post_init__(self):
        if self.kind not in ('model','module','dispatcher','native','tiling','kernel','consumer'):
            raise ValueError('UNKNOWN_EVENT_KIND')
        if self.source not in ('runtime','mock','inferred'):
            raise ValueError('UNKNOWN_EVENT_SOURCE')
        if any(type(v) is not str or not v for v in (self.event_id,self.worker,self.request_id)):
            raise ValueError('EVENT_IDENTITY_REQUIRED')
        if any(type(v) is not int or v<0 for v in (self.process,self.rank,self.device,self.thread,self.generation)):
            raise ValueError('EVENT_SCOPE_REQUIRED')

    def scope(self):return (self.process,self.worker,self.rank,self.device,self.thread,self.request_id,self.generation)


class InvocationCorrelator:
    def __init__(self,max_events=100000):
        if type(max_events) is not int or max_events<1:raise ValueError('FINITE_EVENT_BUDGET_REQUIRED')
        self.max_events=max_events;self._events={};self._lock=threading.Lock()

    def record(self,event):
        if not isinstance(event,InvocationEvent):raise TypeError('TYPED_EVENT_REQUIRED')
        with self._lock:
            if event.event_id in self._events:raise ValueError('DUPLICATE_EVENT_ID')
            if len(self._events)>=self.max_events:raise ValueError('CORRELATION_BUDGET_EXHAUSTED')
            if event.parent_id==event.event_id:raise ValueError('SELF_PARENT')
            # Out-of-order arrival is supported. Cycles never become observed edges.
            cursor=event.parent_id
            while cursor in self._events:
                cursor=self._events[cursor].parent_id
                if cursor==event.event_id:raise ValueError('CYCLIC_INVOCATION')
            self._events[event.event_id]=event

    def report(self):
        with self._lock:
            edges=[]
            for event in self._events.values():
                if event.parent_id is None:continue
                parent=self._events.get(event.parent_id)
                if parent is None:status,reason='NOT_VISIBLE','PARENT_NOT_CAPTURED'
                elif parent.scope()!=event.scope():status,reason='REJECTED','IDENTITY_OR_GENERATION_MISMATCH'
                elif parent.source=='runtime' and event.source=='runtime':status,reason='OBSERVED','EXPLICIT_RUNTIME_PARENT_ID'
                elif parent.source=='mock' or event.source=='mock':status,reason='MOCK','TEST_EVENTS_ONLY'
                else:status,reason='INFERRED','UNVERIFIED_CAUSAL_JOIN'
                edges.append(dict(source=event.parent_id,target=event.event_id,status=status,reason=reason))
            return dict(schema='demoncat.invocation_correlation.v1',events=[asdict(e) for e in self._events.values()],
                edges=edges,npu_qualified=False,limitation='Runtime hooks must supply genuine causal IDs; this model cannot establish NPU causality by itself.')
