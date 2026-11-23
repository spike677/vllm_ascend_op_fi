"""Trusted host delivery scope. This allow-list never grants a deployment."""
RC_ID = 'demoncatpfi-rc3-preview-20261123-r6'
TEST_CANDIDATES = frozenset(('28.fia.denominator_replay', 'quant.batch_shared_scale','39.output_proxy','18.api_half','22.wrong_row','39.matmul_range','17.spatial_padding','41.tail_column','26.fia_destination','20.buffer_substitute'))


def prepare_binding(deployment_callback):
    if not callable(deployment_callback):
        raise TypeError('RC3_TRUSTED_DEPLOYMENT_CALLBACK_REQUIRED')

    def bind(identity, config, group):
        if config.implementation_version != '1' or config.mechanism not in TEST_CANDIDATES:
            raise RuntimeError('RC3_PREVIEW_OUTSIDE_DELIVERY_SCOPE:' + config.mechanism)
        if group.get('injection_policy') != 'ALL_RANK_EVERY_ELIGIBLE_CALL':
            raise RuntimeError('RC3_CONTINUOUS_GROUP_REQUIRED')
        return deployment_callback(identity, config, group)
    return bind


def next_request_phase(reply, plan, completed):
    """Host reads the shared public plan; never guesses the next request."""
    if reply.get('status')!='QUERY_OK':
        raise RuntimeError('RC3_QUERY_FAILED_RECOVER_DO_NOT_REPLAY')
    state=reply['response']['provider_state']
    phases=[row['phase'] for row in plan['phases']]
    if state.get('request_plan')!=plan or completed!=phases[:len(completed)]:
        raise RuntimeError('RC3_REQUEST_PLAN_CHANGED')
    phase=state.get('next_phase')
    if phase=='complete':
        if state.get('complete') is not True or len(completed)!=len(phases):
            raise RuntimeError('RC3_INCOMPLETE_RECOVERY')
    elif len(completed)>=min(len(phases),plan['max_iterations']) or phase!=phases[len(completed)]:
        raise RuntimeError('RC3_UNEXPECTED_PHASE_DO_NOT_REPLAY')
    return phase
