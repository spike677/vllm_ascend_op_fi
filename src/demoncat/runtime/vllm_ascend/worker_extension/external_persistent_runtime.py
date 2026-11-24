"""Additive v2 route on the existing Runtime tenant/lease/recovery protocol.

Installed only by a trusted dedicated-instance bootstrap before the first RPC.
All non-v2 calls use the unmodified base methods. No global resolver monkeypatch.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
import re
import secrets

from .runtime import (WorkerFaultRuntime, Rejected, _Lease, _TICKET, _TARGET,
                      _check_shape, _text, _json, _provider_error)
from .external_provider import factory_sha256
from .config import FaultConfig
from .serving_context import require

ATTRIBUTE='_demoncat_external_persistent_sites_v2'


@dataclass(frozen=True)
class PersistentFaultConfig(FaultConfig):
    implementation_version: str='2'

    def __post_init__(self):
        if (self.fault_id!='external' or type(self.variant) is not str
                or re.fullmatch(r'[a-z][a-z0-9_]{0,63}',self.variant) is None
                or self.implementation_version!='2'):
            raise ValueError('EXTERNAL_PERSISTENT_NAMED_VARIANT_V2_REQUIRED')


@dataclass(frozen=True)
class PersistentSite:
    mechanism: str
    region_kind: str
    factory: object
    source_sha256: str
    expected_deployment: dict


def register_persistent_site(worker,spec,*,authority,operator):
    from .deployment_authorization import authority_for
    from .deployment_acceptance import observe_deployment
    require(authority is not None and authority_for(worker) is authority
            and authority.authorize_operator(operator,'register_persistent_site') is True,
            'EXTERNAL_V2_TRUSTED_DEPLOYMENT_REQUIRED')
    require(type(spec) is PersistentSite and spec.mechanism.startswith('external.'),
            'EXTERNAL_V2_TYPED_SPEC_REQUIRED')
    PersistentFaultConfig('external',spec.mechanism.removeprefix('external.'))
    require(spec.region_kind in ('persistent_weight','persistent_kv'),'EXTERNAL_V2_REGION_KIND_REQUIRED')
    require(factory_sha256(spec.factory)==spec.source_sha256,'EXTERNAL_V2_SOURCE_SHA_MISMATCH')
    require(type(spec.expected_deployment) is dict and spec.expected_deployment
            and observe_deployment(worker,worker.model_runner.model)==spec.expected_deployment,
            'EXTERNAL_V2_DEPLOYMENT_IDENTITY_MISMATCH')
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    require(type(runtime) is PersistentWorkerRuntime and not runtime._tenants and not runtime._leases
            and runtime._census is None,'EXTERNAL_V2_BOOTSTRAP_BEFORE_ATTACHMENT_REQUIRED')
    sites=getattr(worker,ATTRIBUTE,{})
    require(type(sites) is dict,'EXTERNAL_V2_SITE_TABLE_INVALID')
    require(spec.mechanism not in sites,'EXTERNAL_V2_ALREADY_REGISTERED')
    from .external_provider import external_site
    require(external_site(worker,spec.mechanism) is None,'EXTERNAL_V2_CANNOT_SHADOW_V1')
    sites[spec.mechanism]=(PersistentSite(spec.mechanism,spec.region_kind,spec.factory,spec.source_sha256,
                                        deepcopy(spec.expected_deployment)),worker.model_runner.model)
    setattr(worker,ATTRIBUTE,sites)
    return dict(status='PERSISTENT_SITE_REGISTERED_NOT_QUALIFIED',mechanism=spec.mechanism,
                region_kind=spec.region_kind,source_sha256=spec.source_sha256,qualification_granted=False)


def create_persistent_provider(worker,model,identity,mechanism):
    from .deployment_acceptance import observe_deployment,deployment_acceptance_report
    pair=getattr(worker,ATTRIBUTE,{}).get(mechanism)
    require(pair is not None,'EXTERNAL_V2_NOT_REGISTERED')
    spec,original_model=pair
    require(model is original_model and worker.model_runner.model is model,'EXTERNAL_V2_MODEL_CHANGED')
    require(factory_sha256(spec.factory)==spec.source_sha256,'EXTERNAL_V2_SOURCE_SHA_MISMATCH')
    require(observe_deployment(worker,model)==spec.expected_deployment,'EXTERNAL_V2_DEPLOYMENT_IDENTITY_MISMATCH')
    report=deployment_acceptance_report(mechanism,worker,model)
    require(report['can_prepare'] and report['acceptance_permit'] is not None,
            'EXTERNAL_V2_CURRENT_ACCEPTANCE_PERMIT_REQUIRED')
    provider=spec.factory(worker,model,identity,mechanism=mechanism,region_kind=spec.region_kind)
    require(getattr(provider,'persistent_protocol_version',None)==2
            and getattr(provider,'region_kind',None)==spec.region_kind
            and all(callable(getattr(provider,k,None)) for k in ('dispatch','query')),
            'EXTERNAL_V2_PROVIDER_PROTOCOL_REQUIRED')
    provider._deployment_admission=(worker,model,report)
    return provider


class PersistentWorkerRuntime(WorkerFaultRuntime):
    """Only v2 prepare resolution is extended; _run and cleanup are inherited."""
    def _inert(self,state):
        return WorkerFaultRuntime._inert(state)

    def _query(self,context,payload):
        # Query is a serialized Worker control boundary, not a mutation. Held
        # backups are expected while enabled; the base transient zero-lease
        # predicate would incorrectly quarantine this otherwise healthy owner.
        self._persistent_action='query'
        try:return super()._query(context,payload)
        finally:self._persistent_action=None

    def _run(self,lease,action,settings):
        if getattr(lease.provider,'persistent_protocol_version',None)!=2:
            return super()._run(lease,action,settings)
        # The base state transitions and failure ownership are unchanged. Only
        # the quiescence predicate admits held persistent backups in v2.
        self._persistent_action=action
        try:return super()._run(lease,action,settings)
        finally:self._persistent_action=None

    def _quiet(self,state):
        action=getattr(self,'_persistent_action',None)
        if not action or not state or state.get('persistent_protocol_version')!=2:
            return WorkerFaultRuntime._quiet(state)
        if action not in ('register','check','enable','query'):
            return WorkerFaultRuntime._quiet(state)
        counts=all(type(state.get(k)) is int and state[k]>=0
                   for k in ('inflight','persistent_leases','leases'))
        return (state.get('product_mode') is not True
                and counts
                and state.get('inflight')==0
                and state.get('device_readers_complete') is True
                and (action=='query' or state.get('exclusive_ingress_paused') is True)
                and state.get('persistent_leases')==state.get('leases')
                and (type(state.get('enabled')) is bool if action=='query'
                     else state.get('enabled') is (action=='enable')))

    def invoke(self,worker,action,payload=None):
        result=super().invoke(worker,action,payload)
        if action=='capabilities' and result.get('status')=='CAPABILITIES':
            result['providers'] += [dict(mechanism=name,implementation_version='2',
                region_kind=spec.region_kind,attachment='trusted_persistent_v2_bootstrap',
                source_sha256=spec.source_sha256,qualification_granted=False)
                for name,(spec,_) in sorted(getattr(worker,ATTRIBUTE,{}).items())]
        return result

    def _dispatch(self,worker,context,action,payload):
        if action=='prepare' and any(getattr(x.provider,'persistent_protocol_version',None)==2
                                     for x in self._leases.values()):
            # A broadcast aimed at another rank must remain RPC_NOT_TARGETED.
            self._expect_worker(context,payload.get('worker') if isinstance(payload,dict) else None)
            raise Rejected('EXTERNAL_V2_EXCLUSIVE_INSTANCE_OWNED')
        if not (action=='prepare' and isinstance(payload,dict)
                and payload.get('implementation_version')=='2'
                and isinstance(payload.get('mechanism'),str) and payload['mechanism'].startswith('external.')):
            return super()._dispatch(worker,context,action,payload)
        # Same checks as frozen base prepare; no new owner table or phase machine.
        _check_shape(payload,_TICKET|_TARGET,('settings',))
        settings=payload.get('settings',{})
        if not isinstance(settings,dict):raise Rejected('INVALID_PROVIDER_SETTINGS')
        try:settings=json.loads(_json(settings))
        except (TypeError,ValueError,RecursionError):raise Rejected('INVALID_PROVIDER_SETTINGS') from None
        tenant=self._tenant(context,payload)
        if self._census is not None:raise Rejected('WORKER_CENSUS_IN_PROGRESS')
        mechanism=_text(payload['mechanism']);version=_text(payload['implementation_version']);owner=_text(payload['owner'])
        if mechanism in self._leases:raise Rejected('MECHANISM_ALREADY_OWNED')
        if self._leases:raise Rejected('EXTERNAL_V2_EXCLUSIVE_INSTANCE_REQUIRED')
        if self._quarantined():raise Rejected('WORKER_RECOVERY_REQUIRED')
        if any(item.replay_pending for item in self._leases.values()):raise Rejected('WORKER_REPLAY_IN_PROGRESS')
        try:provider=create_persistent_provider(worker,context.model,deepcopy(context.worker),mechanism)
        except Exception as error:return dict(status='PROVIDER_UNAVAILABLE',**_provider_error(error))
        self._generation+=1
        lease=_Lease(tenant,mechanism,version,owner,self._generation,secrets.token_hex(16),provider)
        self._leases[mechanism]=lease
        return self._run(lease,action,settings)


def install_persistent_runtime(worker,*,authority,operator):
    from .deployment_authorization import authority_for
    require(authority is not None and authority_for(worker) is authority
            and authority.authorize_operator(operator,'install_persistent_runtime') is True,
            'EXTERNAL_V2_TRUSTED_DEPLOYMENT_REQUIRED')
    require(not hasattr(worker,'_demoncat_worker_fault_runtime'),
            'EXTERNAL_V2_BOOTSTRAP_BEFORE_FIRST_RPC_REQUIRED')
    worker._demoncat_worker_fault_runtime=PersistentWorkerRuntime(os.getpid())
    return dict(status='PERSISTENT_RUNTIME_INSTALLED',existing_state_machine='WorkerFaultRuntime',qualification_granted=False)
