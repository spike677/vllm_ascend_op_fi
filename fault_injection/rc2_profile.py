"""RC2 test handoff policy; never grants a live deployment or target admission."""
RC_ID = 'demoncatpfi-yellow-rc2-20261119'
TEST_CANDIDATES = frozenset(('39.output_proxy', 'quant.batch_shared_scale', '18.api_half', '8.restore_offset', '17.spatial_padding', '22.wrong_row'))


def test_candidate(mechanism):
    return mechanism in TEST_CANDIDATES


def prepare_binding(deployment_callback):
    """Wrap HostAdapter's existing trusted inert-binding callback.

The same allow-list is displayed by delivery_scope. Historical factories remain
available to their explicit legacy contracts, outside this RC2 product profile.
This callback never chooses layers or creates a Worker, owner, or request.
"""
    if not callable(deployment_callback):
        raise TypeError('RC2_TRUSTED_DEPLOYMENT_CALLBACK_REQUIRED')

    def bind(identity, config):
        if config.implementation_version != '1' or not test_candidate(config.mechanism):
            raise RuntimeError('RC2_NOT_IN_READY_TEST_SCOPE:' + config.mechanism)
        return deployment_callback(identity, config)

    return bind
