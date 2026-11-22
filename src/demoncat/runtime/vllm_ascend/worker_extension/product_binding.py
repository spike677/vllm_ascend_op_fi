"""Trusted deployment RPC. Profiles are installed by bootstrap, never by users."""
from copy import deepcopy
from .serving_context import require
from .deployment_authorization import authority_for
from .continuous_binding import (install_continuous_binding, remove_continuous_binding,
                                 current_binding, observe_continuous_warmup)
from fault_injection.product_contract import PRODUCT_CANDIDATES, RequestScope

ATTRIBUTE = '_demoncat_product_profiles'


def configure_product_profiles(worker, *, profiles, authority, operator):
    require(authority_for(worker) is authority and authority.authorize_operator(operator, 'configure_product_profiles') is True,
            'TRUSTED_PRODUCT_DEPLOYMENT_AUTHORITY_REQUIRED')
    require(type(profiles) is dict and profiles and not hasattr(worker, ATTRIBUTE), 'PRODUCT_PROFILES_ALREADY_SET_OR_EMPTY')
    for name, profile in profiles.items():
        require(type(name) is str and name and set(profile) == {'mechanism', 'canary_prompts', 'mode'}
            and profile['mechanism'] in PRODUCT_CANDIDATES
            and profile['mode'] == 'PRODUCT_ACCEPTANCE_ONLY', 'PRODUCT_PROFILE_NOT_REVIEWED')
    setattr(worker, ATTRIBUTE, deepcopy(profiles))


def product_binding_rpc(worker, action, payload):
    require(type(payload) is dict, 'PRODUCT_RPC_PAYLOAD_REQUIRED')
    authority = authority_for(worker)
    require(authority is not None and authority.authorize_operator(payload.get('operator'), 'product_binding_rpc') is True,
            'PRODUCT_RPC_OPERATOR_NOT_AUTHORIZED')
    allowed = {'operator', 'profile'} | ({'group_contract', 'scope'} if action == 'attach' else
              {'request_ids'} if action == 'canary' else set())
    require(not set(payload) - allowed, 'PRODUCT_RPC_UNKNOWN_FIELD')
    profile = getattr(worker, ATTRIBUTE, {}).get(payload.get('profile'))
    require(profile is not None, 'PRODUCT_PROFILE_UNKNOWN')
    mechanism = profile['mechanism']
    if action == 'capabilities':
        return dict(status='PRODUCT_CANDIDATE', mechanism=mechanism,
                    mode='PRODUCT_ACCEPTANCE_ONLY', product_qualified=False,
                    range=dict(max_context=1024, max_tokens=4, max_num_seqs=1, single_chunk=True,
                               image_grid=[1,4,4] if mechanism=='17.spatial_padding' else None),
                    user_prompt_registry=False, named_variant=mechanism)
    b = current_binding(worker, worker.model_runner.model, mechanism)
    if action == 'attach':
        scope = RequestScope(**payload['scope'])
        result = install_continuous_binding(worker, mechanism=mechanism, prompts=profile['canary_prompts'],
                                            group_contract=payload['group_contract'])
        b = current_binding(worker, worker.model_runner.model, mechanism)
        b.product = dict(mode='PRODUCT_ACCEPTANCE_ONLY', scope=dict(mode=scope.mode, count=scope.count),
                         profile=payload['profile'])
        return dict(result, product_mode=True)
    require(b is not None and b.product is not None, 'PRODUCT_BINDING_NOT_ATTACHED')
    if action == 'canary':
        return observe_continuous_warmup(worker, mechanism, request_ids=payload.get('request_ids'))
    if action == 'reset_canary_observer':
        require(not b.leases and b.warmup_request is None, 'PRODUCT_CANARY_REQUIRES_RELEASED_OWNER')
        b.warmups.clear()
        return dict(status='CANARY_OBSERVER_RESET')
    if action == 'detach':
        remove_continuous_binding(worker, mechanism)
        return dict(status='PRODUCT_BINDING_REMOVED')
    raise ValueError('PRODUCT_RPC_ACTION_NOT_SUPPORTED')
