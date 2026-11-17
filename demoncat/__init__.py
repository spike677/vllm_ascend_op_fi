"""Public DemonCATPFI API; coexist with the separately installed Worker package."""
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from fault_injection.unified import configure, example_config, migrate_legacy, validate_config, ConfigError
from fault_injection.unified.adapters import registry
from fault_injection.unified.protocol import Request, Result, AdapterProtocol

__all__=['configure','example_config','migrate_legacy','validate_config','ConfigError',
         'registry','Request','Result','AdapterProtocol']
