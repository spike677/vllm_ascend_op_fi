"""Trusted deployment bootstrap. RPC cannot install its own trust or accept paths."""
from dataclasses import dataclass
from copy import deepcopy
from .deployment_authorization import (authority_for, configure_deployment_acceptance,
    install_deployment_acceptance_permit, remove_deployment_acceptance_permit)
from .product_config import configure_product_deployments

_ATTRIBUTE = '_demoncat_trusted_product_bootstrap'

@dataclass(frozen=True)
class Bootstrap:
    authority: object
    paths: tuple
    canary_tokens: dict


def configure_product_bootstrap(worker, *, source, authorize_operator, paths, canary_tokens):
    """Local trusted initializer after model load, on the Worker execution thread.

    Authentication belongs to the deployment. Neither this initializer nor the
    callback can be invoked or replaced through the bootstrap RPC.
    """
    if hasattr(worker, _ATTRIBUTE): raise RuntimeError('PRODUCT_BOOTSTRAP_ALREADY_BOUND')
    if not callable(authorize_operator): raise ValueError('TRUSTED_AUTHORIZATION_CALLBACK_REQUIRED')
    if not isinstance(paths, (tuple, list)) or not paths or any(type(p) is not str or not p for p in paths):
        raise ValueError('TRUSTED_DEPLOYMENT_PATHS_REQUIRED')
    if type(canary_tokens) is not dict: raise ValueError('TRUSTED_CANARY_TOKENS_REQUIRED')
    authority = authority_for(worker)
    if authority is None:
        authority = configure_deployment_acceptance(worker, source=source, authorize_operator=authorize_operator)
    elif authority.source != source or authority.authorize_operator is not authorize_operator:
        raise RuntimeError('PRODUCT_BOOTSTRAP_AUTHORITY_CONFLICT')
    setattr(worker, _ATTRIBUTE, Bootstrap(authority, tuple(paths), deepcopy(canary_tokens)))
    return dict(status='TRUSTED_BOOTSTRAP_BOUND', qualification_granted=False)


def product_bootstrap_rpc(worker, action, payload):
    state = getattr(worker, _ATTRIBUTE, None)
    if type(state) is not Bootstrap or authority_for(worker) is not state.authority:
        raise PermissionError('TRUSTED_PRODUCT_BOOTSTRAP_REQUIRED')
    fields = {'configure': {'operator'},
              'permit': {'operator', 'mechanisms', 'attempt', 'source_manifest_sha256'},
              'remove_permit': {'operator'}}
    if action not in fields or type(payload) is not dict or set(payload) != fields[action]:
        raise ValueError('PRODUCT_BOOTSTRAP_SCHEMA_INVALID')
    operator = payload['operator']
    if type(operator) is not str or not operator or len(operator) > 128:
        raise ValueError('OPERATOR_REQUIRED')
    if state.authority.authorize_operator(operator, 'product_bootstrap.' + action) is not True:
        raise PermissionError('DEPLOYMENT_OPERATOR_NOT_AUTHORIZED')
    if action == 'configure':
        return configure_product_deployments(worker, paths=list(state.paths), authority=state.authority,
            operator=operator, canary_tokens=deepcopy(state.canary_tokens))
    if action == 'permit':
        return install_deployment_acceptance_permit(worker, authority=state.authority, **payload)
    remove_deployment_acceptance_permit(worker, authority=state.authority, operator=operator)
    return dict(status='PERMIT_REMOVED', qualification_granted=False)
