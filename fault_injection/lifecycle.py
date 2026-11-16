"""Owner-scoped lifecycle for the remaining model adapters.

This is a local model/backend contract, not the unavailable inner framework's
API. It deliberately does not wrap the separately accepted Fault39 backend.
"""
from dataclasses import asdict, dataclass
import copy
from pathlib import Path
import threading
import uuid

from .catalog import resolve


@dataclass(frozen=True)
class FaultConfig:
    mechanism: str
    arm: str = 'fault'
    unsupported: str = 'error'
    evidence_dir: str | None = None
    target: str = 'auto'

    def __post_init__(self):
        if type(self.mechanism) is not str:
            raise TypeError('mechanism must be an explicit catalog string')
        capability = resolve(self.mechanism)
        if self.mechanism.startswith('39.') and self.arm != 'observe':
            raise ValueError('Fault39 uses its separately validated demoncat_fault39 backend')
        if type(self.arm) is not str or self.arm not in ('baseline', 'fault', 'observe'):
            raise ValueError('arm must be baseline, fault or observe')
        if self.arm == 'observe' and capability.integration != 'native_provider_bridge':
            raise ValueError('Read-only operator observation requires a native provider bridge')
        if type(self.unsupported) is not str or self.unsupported not in ('error', 'passthrough'):
            raise ValueError('unsupported must be error or passthrough')
        if type(self.target) is not str or self.target != 'auto':
            raise ValueError('Only target=auto is supported; no layer or address selectors')
        if self.evidence_dir is not None:
            if not isinstance(self.evidence_dir, (str, Path)) or not str(self.evidence_dir).strip():
                raise ValueError('evidence_dir must be a nonempty path or None')
            object.__setattr__(self, 'evidence_dir', str(self.evidence_dir))

    @classmethod
    def from_dict(cls, values):
        if type(values) is not dict:
            raise TypeError('FaultConfig mapping must be a dict')
        # The dataclass constructor rejects every unconsumed field, including
        # ranks/cards, callsite/op and layer names. They are not state labels.
        return cls(**values)


class ModelFaultBackend:
    """Register disabled, enable a new scoped adapter, and truly restore on OFF.

    All mutating methods belong to the registration thread. Concurrent or
    reentrant execution is rejected by adapters; this API does not claim an
    asynchronous native drain or multi-fault composition implementation.
    """
    def __init__(self, model, runtime=None):
        if not callable(getattr(model, 'named_modules', None)):
            raise TypeError('An actual model with named_modules() is required')
        self.model, self.runtime = model, runtime
        self.owner = uuid.uuid4().hex
        self.generation = 0
        self.state = 'NEW'
        self.config = None
        self._thread = None
        self._session = None
        self._last_session = None
        self._history = []
        self._error = None
        self._required_binding = None

    def _check_owner(self):
        if self._thread is not None and self._thread != threading.get_ident():
            raise RuntimeError('Backend control must run on its registration thread')

    def prepare(self):
        self._check_owner()
        if self.config is not None or self._session is not None:
            raise RuntimeError('prepare requires an unregistered backend')
        # Read-only discovery is independently callable through inspect_model;
        # prepare verifies the model contract without installing any hooks.
        list(self.model.named_modules())
        self.state = 'PREPARED'
        return self.query()

    def register(self, config):
        self._check_owner()
        if self.config is not None or self._session is not None:
            raise RuntimeError('Backend already registered or awaiting recovery')
        if isinstance(config, dict):
            config = FaultConfig.from_dict(config)
        if not isinstance(config, FaultConfig):
            raise TypeError('register requires FaultConfig or its exact field mapping')
        from .providers import qualified_binding

        self._thread = threading.get_ident()
        self.config = config
        if self.runtime is None and config.arm != 'observe':
            self._required_binding = qualified_binding(config.mechanism)
        self.state = 'REGISTERED_DISABLED'
        self._error = None
        return self.query()

    def enable(self):
        self._check_owner()
        if self.config is None:
            raise RuntimeError('Register a configuration before enabling')
        if self._session is not None:
            raise RuntimeError('An active or recovery-pending session already exists')
        from . import attach
        self.generation += 1
        self._error = None
        try:
            self._session = attach(self.model, self.config.mechanism,
                arm=self.config.arm, unsupported=self.config.unsupported,
                evidence_dir=self.config.evidence_dir, runtime=self.runtime,
                _required_binding=self._required_binding)
            self._session.__enter__()
        except BaseException as exc:
            self._error = repr(exc)
            if self._session is not None and self._session.cleanup_completed is not True:
                # A session refused before acquiring a lease has no edits.
                # Any acquired-but-unrestored lease must remain reachable.
                if self._session._locked:
                    self.state = 'RECOVERY_FAILED'
                    raise
            self._archive_session()
            self.state = 'ENABLE_FAILED'
            raise
        self.state = 'ENABLED'
        return self.query()

    def inject(self, mechanism):
        """Start a qualified injection using the actual model already bound to this worker."""
        from .providers import qualified_binding
        from .core import UnsupportedCall

        self._check_owner()
        binding = qualified_binding(mechanism)
        if self.runtime is not None or binding is None:
            raise UnsupportedCall('A qualified provider and replay for the loaded model are required')
        self.register(FaultConfig(mechanism))
        self._required_binding = binding
        return self.enable()

    def _archive_session(self):
        if self._session is not None:
            self._last_session = self._session.report()
            self._history.append(dict(generation=self.generation, report=self._last_session))
            self._session = None

    def disable(self):
        self._check_owner()
        if self._session is None:
            if self.config is not None:
                self.state = 'DISABLED_CONFIRMED'
            return self.query()
        try:
            self._session.close()
        except BaseException as exc:
            self._error = repr(exc)
            self.state = ('DISABLED_CONFIRMED' if self._session.cleanup_completed is True
                          else 'RECOVERY_FAILED')
            # Even when only report persistence failed, keep the session so an
            # owner retry can finish evidence rather than discard that failure.
            raise
        if self._session.cleanup_completed is not True:
            self.state = 'RECOVERY_FAILED'
            raise RuntimeError('Adapter did not confirm owned cleanup')
        self._archive_session()
        self.state = 'DISABLED_CONFIRMED'
        return self.query()

    def clean(self):
        self._check_owner()
        self.disable()
        self.config = None
        self._required_binding = None
        self.state = 'UNREGISTERED'
        return self.query()

    def unregister(self):
        return self.clean()

    def assert_coverage(self):
        self._check_owner()
        if self.state != 'ENABLED' or self._session is None:
            raise RuntimeError('Coverage requires the current enabled generation')
        return self._session.assert_coverage()

    def query(self):
        report = self._session.report() if self._session is not None else self._last_session
        cleanup = self._session is None or self._session.cleanup_completed is True
        return dict(schema='demoncat.model_fault.backend.v1', owner=self.owner,
            generation=self.generation, state=self.state, registered=self.config is not None,
            active=self._session is not None and self._session._active,
            cleanup_confirmed=cleanup, recovery_pending=not cleanup and self.state == 'RECOVERY_FAILED',
            config=asdict(self.config) if self.config is not None else None,
            session=copy.deepcopy(report), completed_generations=len(self._history), last_error=self._error,
            scope='one model instance, one synchronous owner thread; no rank/card scope inferred',
            native_acceptance_inherited=False, inner_framework_integration='PENDING')
