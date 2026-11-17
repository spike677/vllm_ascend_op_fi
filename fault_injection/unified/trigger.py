"""Atomic admission after eligibility/target filtering; never keyed by addresses."""
from dataclasses import asdict, dataclass
import hashlib
import json
import threading

from .config import validate_config


@dataclass(frozen=True)
class CallIdentity:
    event_id: str
    process: int
    worker: str
    rank: int
    device: int
    request_id: str | None
    operator: str
    layer: str

    def __post_init__(self):
        for name in ('event_id','worker','operator','layer'):
            if type(getattr(self,name)) is not str or not getattr(self,name):
                raise ValueError('CALL_IDENTITY_REQUIRED:'+name)
        for name in ('process','rank','device'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:
                raise ValueError('CALL_IDENTITY_REQUIRED:'+name)
        if self.request_id is not None and (type(self.request_id) is not str or not self.request_id):
            raise ValueError('CALL_IDENTITY_REQUIRED:request_id')


@dataclass(frozen=True)
class TriggerDecision:
    selected: bool
    reason: str
    ordinal: int | None
    hits: int
    scope_key: tuple


class TriggerEngine:
    """One engine per owner/generation; max_hits is per configured scope bucket.

    process scope explicitly shares ranks/requests within this engine; rank
    and request scopes isolate workers and ranks. No implicit cross-process or
    cross-session sharing. Selection reserves budget even if execution fails.
    """
    def __init__(self, config, *, max_events=100000):
        self.config=validate_config(config)
        if type(max_events) is not int or max_events<1:raise ValueError('FINITE_EVENT_BUDGET_REQUIRED')
        self.max_events=max_events
        self._lock=threading.Lock()
        self._seen=set()
        self._counts={}
        self._hits={}
        self._observed=0

    def _key(self, call):
        scope=self.config['trigger']['scope']
        base=(call.process,call.worker,call.rank,call.device)
        if scope=='process':return (call.process,)
        if scope=='worker':return base[:2]
        if scope=='rank':return base[:3]
        if scope=='device':return base
        value=getattr(call,{'request':'request_id','operator':'operator','layer':'layer'}[scope])
        if value is None:raise ValueError('SCOPE_IDENTITY_MISSING:'+scope)
        return base+(value,)

    def decide(self, call, *, eligible):
        if not isinstance(call,CallIdentity) or type(eligible) is not bool:
            raise TypeError('TYPED_CALL_AND_ELIGIBILITY_REQUIRED')
        with self._lock:
            key=self._key(call)
            unique=(call.process,call.worker,call.rank,call.device,call.event_id)
            if unique in self._seen:raise ValueError('DUPLICATE_CALL_EVENT')
            if len(self._seen)>=self.max_events:raise RuntimeError('TRIGGER_EVENT_BUDGET_EXHAUSTED')
            self._seen.add(unique)
            self._observed+=1
            target=self.config['target'];trigger=self.config['trigger']
            matched=(eligible and call.rank in target['rank'] and call.device in target['device']
                and (target['layer'] is None or target['layer']==call.layer)
                and (target['request_scope']=='all' or call.request_id in target['request_ids']))
            hits=self._hits.get(key,0)
            if not matched:return TriggerDecision(False,'INELIGIBLE_OR_TARGET_FILTERED',None,hits,key)
            ordinal=self._counts.get(key,0)+1
            self._counts[key]=ordinal
            mode=trigger['mode']
            selected={'first':lambda:ordinal==1,'nth':lambda:ordinal==trigger['ordinal'],
                'every_nth':lambda:ordinal%trigger['ordinal']==0,'every_match':lambda:True,
                'probability':lambda:self._random(key,ordinal)<trigger['probability'],
                'condition':lambda:self._condition(call)}[mode]()
            if target['operator_call'] is not None:selected &= ordinal==target['operator_call']
            if trigger['max_hits'] is not None and hits>=trigger['max_hits']:
                return TriggerDecision(False,'HIT_BUDGET_EXHAUSTED',ordinal,hits,key)
            if selected:
                hits+=1;self._hits[key]=hits
            return TriggerDecision(bool(selected),'SELECTED' if selected else 'TRIGGER_NOT_MATCHED',ordinal,hits,key)

    def _random(self,key,ordinal):
        raw=json.dumps([self.config['trigger']['seed'],key,ordinal],separators=(',',':')).encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')/2**64

    def _condition(self,call):
        condition=self.config['trigger']['condition']
        actual=getattr(call,condition['field']);expected=condition['value']
        return {'eq':lambda:actual==expected,'ne':lambda:actual!=expected,
                'in':lambda:actual in expected}[condition['op']]()

    def status(self):
        with self._lock:
            return dict(observed_calls=self._observed,eligible_calls=sum(self._counts.values()),
                reserved_hits=sum(self._hits.values()),scope=self.config['trigger']['scope'],
                counting='after eligibility and target filters; before trigger and hit budget',
                buckets=[dict(key=list(k),eligible=n,hits=self._hits.get(k,0)) for k,n in self._counts.items()])
