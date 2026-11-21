"""Trusted host delivery scope. This allow-list never grants a deployment."""
RC_ID = 'demoncatpfi-rc3-preview-20261121'
TEST_CANDIDATES = frozenset(('28.fia.denominator_replay', 'quant.batch_shared_scale'))


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
