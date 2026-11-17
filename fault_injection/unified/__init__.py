"""DemonCATPFI version-one control protocol, independent of native qualification."""
from .config import ConfigError, example_config, migrate_legacy, validate_config


def configure(mechanism, config):
    from .session import UnifiedSession
    return UnifiedSession(validate_config(config, mechanism))
