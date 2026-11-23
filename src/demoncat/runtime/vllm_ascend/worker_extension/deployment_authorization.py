"""Trusted deployment API; intentionally absent from FaultConfig and public RPC.

The bootstrap installs an authentication/authorization callback from its own
control plane. This does not authenticate arbitrary Python code in this process;
that code already has Worker privileges. Never use a user-supplied callback.
"""
from dataclasses import dataclass
import os
import secrets
import threading
from copy import deepcopy

SOURCES=('BLUE_COMBINATION','YELLOW_TEST_DEPLOYMENT','YELLOW_DEDICATED_INSTANCE')
_ATTRIBUTE='_demoncat_deployment_acceptance_authority'

@dataclass(frozen=True)
class DeploymentAuthority:
    worker: object
    model: object
    source: str
    authorize_operator: object
    pid: int
    thread: int


def authority_for(worker):
    value=getattr(worker,_ATTRIBUTE,None)
    if (type(value) is DeploymentAuthority and value.worker is worker
        and worker.model_runner.model is value.model and value.pid==os.getpid()
        and value.thread==threading.get_ident()):return value
    return None


def configure_deployment_acceptance(worker,*,source,authorize_operator):
    if source not in SOURCES or not callable(authorize_operator):
        raise ValueError('TRUSTED_DEPLOYMENT_AUTHORIZATION_REQUIRED')
    if hasattr(worker,_ATTRIBUTE):raise RuntimeError('DEPLOYMENT_AUTHORITY_ALREADY_BOUND')
    value=DeploymentAuthority(worker,worker.model_runner.model,source,authorize_operator,os.getpid(),threading.get_ident())
    setattr(worker,_ATTRIBUTE,value)
    return value


def install_deployment_acceptance_permit(worker,*,authority,operator,mechanisms,attempt,source_manifest_sha256):
    from .deployment_acceptance import BlueAcceptancePermit,_BLUE
    if authority_for(worker) is not authority:raise RuntimeError('CURRENT_DEPLOYMENT_AUTHORITY_REQUIRED')
    if not isinstance(operator,str) or not 1<=len(operator)<=128:raise ValueError('OPERATOR_REQUIRED')
    if authority.authorize_operator(operator,'install_acceptance_permit') is not True:
        raise PermissionError('DEPLOYMENT_OPERATOR_NOT_AUTHORIZED')
    if hasattr(worker,_BLUE):raise RuntimeError('BLUE_ACCEPTANCE_PERMIT_ALREADY_BOUND')
    if (not isinstance(mechanisms,(tuple,list)) or not mechanisms or len(mechanisms)>17
        or any(type(x) is not str for x in mechanisms) or len(set(mechanisms))!=len(mechanisms)
        or not isinstance(attempt,str) or not 1<=len(attempt)<=128
        or not isinstance(source_manifest_sha256,str) or len(source_manifest_sha256)!=64
        or any(c not in '0123456789abcdef' for c in source_manifest_sha256)):
        raise ValueError('EXPLICIT_BOUNDED_BLUE_ACCEPTANCE_SCOPE_REQUIRED')
    # Pending/unimplemented routes still cannot execute even with a permit.
    value=BlueAcceptancePermit(worker,worker.model_runner.model,tuple(mechanisms),attempt,source_manifest_sha256,
        os.getpid(),threading.get_ident(),authority.source,operator,secrets.token_hex(16))
    setattr(worker,_BLUE,value)
    return dict(mode='DEPLOYMENT_ACCEPTANCE_ONLY',source=authority.source,attempt=attempt,
        permit_id=value.permit_id,operator=operator,acceptance_granted=False)


def acceptance_use(provider,action,*,enforce=False):
    context=getattr(provider,'_deployment_admission',None)
    if context is None:return None  # Research/CPU doubles have separate contracts.
    worker,model,report=context
    data=deepcopy(report)
    permit=data.get('acceptance_permit')
    valid=True
    if data.get('mode')=='PRODUCT_REGISTERED':
        from .continuous_binding import current_binding
        from .product_registration import registered_for
        registration=data.get('product_registration') or {}
        try:
            b=current_binding(worker,model,provider.mechanism)
            if b is None:raise ValueError('PRODUCT_BINDING_NOT_ATTACHED')
            profile=getattr(worker,'_demoncat_product_profiles',{}).get(b.product['profile'])
            registered_for(worker,b.mechanism,profile,expected_sha=registration.get('record_sha256'))
        except (ValueError,RuntimeError,AttributeError):valid=False
    elif permit:
        from .deployment_acceptance import BlueAcceptancePermit,_BLUE
        live=getattr(worker,_BLUE,None)
        valid=(type(live) is BlueAcceptancePermit and live.permit_id==permit['permit_id']
            and live.worker is worker and live.model is model and worker.model_runner.model is model
            and live.pid==os.getpid() and live.thread==threading.get_ident()
            and authority_for(worker) is not None)
    elif data['mode']!='CPU_FIXTURE_ONLY':
        valid=(authority_for(worker) is not None and authority_for(worker).source==data['registration_source']
            and worker.model_runner.model is model)
    if enforce and not valid:raise RuntimeError('DEPLOYMENT_ACCEPTANCE_SCOPE_CHANGED')
    return dict(schema='demoncat.acceptance_use.v1',action=action,scope_valid=valid,
        registration_source=data['registration_source'],mode=data['mode'],accepted=data['accepted'],
        actual=data['actual'],permit=permit,reviewed_combination=data['reviewed_combination'],
        accepted_record_created=False)


def remove_deployment_acceptance_permit(worker, *, authority, operator):
    """Revoke only after owners release; restoration itself never needs a permit."""
    if authority_for(worker) is not authority:
        raise RuntimeError('CURRENT_DEPLOYMENT_AUTHORITY_REQUIRED')
    if authority.authorize_operator(operator,'remove_acceptance_permit') is not True:
        raise PermissionError('DEPLOYMENT_OPERATOR_NOT_AUTHORIZED')
    from .deployment_acceptance import remove_blue_acceptance_permit
    remove_blue_acceptance_permit(worker)
