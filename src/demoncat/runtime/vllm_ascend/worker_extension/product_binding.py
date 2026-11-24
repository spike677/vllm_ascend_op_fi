"""Trusted deployment RPC. Profiles are installed by bootstrap, never by users."""
from copy import deepcopy
from .serving_context import require
from .deployment_authorization import authority_for
from .continuous_binding import (install_continuous_binding, remove_continuous_binding,
                                 current_binding, observe_continuous_warmup)
from fault_injection.product_contract import PRODUCT_CANDIDATES, RequestScope

from .product_native import MECHANISMS as NATIVE_TRIALS, validate_blue_native_library, validate_blue_native_worker

ATTRIBUTE = '_demoncat_product_profiles'


def configure_product_profiles(worker, *, profiles, authority, operator):
    require(authority_for(worker) is authority and authority.authorize_operator(operator, 'configure_product_profiles') is True,
            'TRUSTED_PRODUCT_DEPLOYMENT_AUTHORITY_REQUIRED')
    require(type(profiles) is dict and profiles and not hasattr(worker, ATTRIBUTE), 'PRODUCT_PROFILES_ALREADY_SET_OR_EMPTY')
    from .external_provider import external_site
    for name, profile in profiles.items():
        external=external_site(worker,profile.get('mechanism'),validate=True)
        native = profile.get('mechanism') in NATIVE_TRIALS
        fields = {'mechanism', 'canary_prompts', 'mode'} | ({'native_library', 'native_source'} if native else set())
        require(type(name) is str and name and set(profile) == fields
            and (profile['mechanism'] in PRODUCT_CANDIDATES + NATIVE_TRIALS or external is not None)
            and profile['mode'] == 'PRODUCT_ACCEPTANCE_ONLY', 'PRODUCT_PROFILE_NOT_REVIEWED')
        if native:
            validate_blue_native_worker(worker, authority.source)
            require(profile['native_source'] == authority.source, 'PRODUCT_NATIVE_SOURCE_MISMATCH')
            validate_blue_native_library(*profile['native_library'], authority.source, worker.model_runner.vllm_config.parallel_config.tensor_parallel_size)
    setattr(worker, ATTRIBUTE, deepcopy(profiles))


def product_binding_rpc(worker, action, payload):
    require(type(payload) is dict, 'PRODUCT_RPC_PAYLOAD_REQUIRED')
    authority = authority_for(worker)
    require(authority is not None and authority.authorize_operator(payload.get('operator'), 'product_binding_rpc') is True,
            'PRODUCT_RPC_OPERATOR_NOT_AUTHORIZED')
    allowed = {'operator', 'profile'} | ({'group_contract', 'scope'} if action == 'attach' else
              {'cursor'} if action == 'status' else {'owner'} if action == 'rollback' else set())
    require(not set(payload) - allowed, 'PRODUCT_RPC_UNKNOWN_FIELD')
    profile = getattr(worker, ATTRIBUTE, {}).get(payload.get('profile'))
    require(profile is not None, 'PRODUCT_PROFILE_UNKNOWN')
    mechanism = profile['mechanism']
    from .external_provider import external_site
    external=external_site(worker,mechanism,validate=True)
    if action == 'capabilities':
        return dict(status='PRODUCT_CANDIDATE', mechanism=mechanism,
                    mode='PRODUCT_ACCEPTANCE_ONLY', product_qualified=False,
                    range=dict(max_context=4096 if mechanism=='33.paged_decode' else 1024, max_tokens=4, max_num_seqs=8 if mechanism=='quant.batch_shared_scale' else 1, single_chunk=True,
                               image_grid=[1,4,4] if mechanism=='17.spatial_padding' else None),
                    user_prompt_registry=False, named_variant=mechanism,
                    external_provider=dict(source_sha256=external.source_sha256,qualification_granted=False) if external else None)
    b = current_binding(worker, worker.model_runner.model, mechanism)
    if action == 'rollback':
        # Idempotent for ranks that refused attach. Never remove another owner.
        require(type(payload.get('owner')) is str and payload['owner'], 'PRODUCT_ROLLBACK_OWNER_REQUIRED')
        if b is not None:
            require(b.group['owner'] == payload['owner'], 'PRODUCT_ROLLBACK_OWNER_MISMATCH')
            require(not b.leases, 'PRODUCT_ROLLBACK_RUNTIME_LEASES_REMAIN')
            if b.warmup_request is not None:
                b.warmup_request.close()
                b.warmup_request = None
            remove_continuous_binding(worker, mechanism)
        return dict(status='PRODUCT_BINDING_ROLLED_BACK', owner=payload['owner'], resources_zero=True)
    if action == 'attach':
        native_identity=None
        scope = RequestScope(**payload['scope'])
        if mechanism=='15.alias_detach':
            require(worker.model_runner.vllm_config.parallel_config.tensor_parallel_size == 2,
                    'ALIAS15_TP2_ACCEPTANCE_ONLY')
            # Reuse observed device/CANN identity checks; no Native library is used.
            alias_identity = validate_blue_native_worker(worker, authority.source)
        if mechanism=='33.paged_decode':
            from fault_injection.environment import _live_soc
            soc,source=_live_soc()
            require(soc is not None and soc['family']=='A2' and soc['observed_name']=='Ascend910B4',
                'PRODUCT33_REQUIRES_REVIEWED_A2_LAUNCH_A3_NOT_ADMITTED')
        if mechanism in NATIVE_TRIALS:
            native_identity=validate_blue_native_worker(worker, authority.source)
            require(profile['native_source'] == authority.source, 'PRODUCT_NATIVE_SOURCE_MISMATCH')
        result = install_continuous_binding(worker, mechanism=mechanism, prompts=profile['canary_prompts'],
                                            group_contract=payload['group_contract'], native_library=profile.get('native_library'),
                                            paged_partition=(512,256) if mechanism=='33.paged_decode' else None)
        b = current_binding(worker, worker.model_runner.model, mechanism)
        if mechanism == '15.alias_detach':b.alias_runtime_identity=alias_identity
        if mechanism in NATIVE_TRIALS:b.native_runtime_identity=native_identity
        b.product = dict(mode='PRODUCT_ACCEPTANCE_ONLY', scope=dict(mode=scope.mode),
                         profile=payload['profile'])
        return dict(result, product_mode=True,external_source_sha256=external.source_sha256 if external else None)
    require(b is not None and b.product is not None, 'PRODUCT_BINDING_NOT_ATTACHED')
    if action == 'arm_canary':
        from .product_observer import make_automatic_observer
        require(not b.leases and b.warmup_request is None and len(b.warmups)<6, 'PRODUCT_CANARY_IDLE_BUDGET')
        b.warmup_request=make_automatic_observer(b,canary=True)
        b.warmup_request.install()
        return dict(status='CANARY_OBSERVER_ARMED',rank=b.identity['rank'])
    if action == 'canary':
        require(b.warmup_request is not None and b.warmup_request.completed_canary, 'PRODUCT_CANARY_NOT_OBSERVED')
        b.warmup_request.close();b.warmup_request=None
        return dict(rank=b.identity['rank'],warmup_valid=b.warmup_valid,attempts=len(b.warmups),
                    observations=deepcopy(b.warmups),qualification_granted=False)
    if action == 'status':
        runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
        lease=getattr(runtime,'_leases',{}).get(mechanism)
        observer=getattr(getattr(lease,'provider',None),'observer',None)
        require(observer is not None,'PRODUCT_OBSERVER_NOT_ENABLED')
        cursor=payload.get('cursor',0)
        require(type(cursor) is int and 0<=cursor<=len(observer.records),'PRODUCT_STATUS_CURSOR_INVALID')
        rows=deepcopy(observer.records[cursor:cursor+1])
        return dict(rank=b.identity['rank'],requests=rows,cursor=cursor+len(rows),
                    more=cursor+len(rows)<len(observer.records),active_request_id=next(iter(observer.states),None) if hasattr(observer,'states') else observer.session.ids[0] if observer.session else None,
                    failure=observer.error)
    if action == 'reset_canary_observer':
        require(not b.leases and b.warmup_request is None, 'PRODUCT_CANARY_REQUIRES_RELEASED_OWNER')
        b.warmups.clear()
        return dict(status='CANARY_OBSERVER_RESET')
    if action == 'detach':
        remove_continuous_binding(worker, mechanism)
        return dict(status='PRODUCT_BINDING_REMOVED')
    raise ValueError('PRODUCT_RPC_ACTION_NOT_SUPPORTED')
