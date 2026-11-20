"""RC2.2 test profile; eligibility still requires exact live deployment admission."""
from .rc2_profile import TEST_CANDIDATES as RC2_CANDIDATES
RC_ID = 'demoncatpfi-yellow-rc2.2-20261120'
TEST_CANDIDATES = RC2_CANDIDATES | {'41.tail_column'}

def prepare_binding(deployment_callback):
    if not callable(deployment_callback):
        raise TypeError('RC22_TRUSTED_DEPLOYMENT_CALLBACK_REQUIRED')
    def bind(identity, config):
        if config.implementation_version != '1' or config.mechanism not in TEST_CANDIDATES:
            raise RuntimeError('RC22_NOT_IN_READY_TEST_SCOPE:' + config.mechanism)
        return deployment_callback(identity, config)
    return bind
