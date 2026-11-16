"""Deployment-owned native runtime registration, distinct from model selection.

Factories are trusted Python integrations, not evidence of native correctness.
They receive the model and must perform ABI/loaded-build checks themselves.
There is no implicit import, address setting or vendor loading here.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import threading

from .catalog import resolve
from .core import UnsupportedCall
from .coverage import CoverageCapability

_LOCK = threading.Lock()
_FACTORIES = {}

VALIDATION_LEVELS = (
    'DECLARED', 'IMPLEMENTED', 'CPU_VERIFIED', 'NATIVE_VERIFIED', 'MODEL_VERIFIED', 'GENERALIZED',
)
_QUALIFIED_OPERATORS = {'20.buffer_substitute': {'aclnnQuantMatmulWeightNz', 'aclnnMatmulWeightNz'}}


@dataclass(frozen=True)
class ProviderProfile:
    """One deployment-reviewed scope on an existing runtime binding."""
    implementation_version: str
    validation_level: str
    evidence_sha256: str
    environment_fingerprint: str
    required_libraries: tuple[str, ...]
    build_profile_sha256: str
    operator: str
    boundary: str
    native_signature: str
    layout_sha256: str
    soc: str
    dtype: str
    format: str
    ndim: int
    execution_mode: str
    tp: int
    rank: int
    logical_device: int
    model_class: str | None = None
    model_type: str | None = None

    def __post_init__(self):
        if self.validation_level not in VALIDATION_LEVELS:
            raise ValueError('Unknown validation level')
        for field in ('evidence_sha256', 'environment_fingerprint', 'build_profile_sha256', 'layout_sha256'):
            value = getattr(self, field)
            if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError('Expected SHA-256: ' + field)
        for field in ('implementation_version', 'operator', 'boundary', 'native_signature',
                      'soc', 'dtype', 'format', 'execution_mode'):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Missing profile identity: ' + field)
        if not isinstance(self.required_libraries, tuple) or not self.required_libraries or any(
            not isinstance(name, str) or not name or Path(name).name != name
            for name in self.required_libraries
        ) or len(set(self.required_libraries)) != len(self.required_libraries):
            raise ValueError('Expected distinct required library basenames')
        if (any(type(value) is not int for value in (self.ndim, self.tp, self.rank, self.logical_device))
                or self.ndim <= 0 or self.tp <= 0 or not 0 <= self.rank < self.tp or self.logical_device < 0):
            raise ValueError('Invalid rank, device or tensor dimension scope')
        for value in (self.model_class, self.model_type):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError('Invalid model scope')
        if self.validation_level in ('MODEL_VERIFIED', 'GENERALIZED') and not self.model_class:
            raise ValueError('Model acceptance requires a concrete model implementation in this scope')


@dataclass(frozen=True)
class _Binding:
    factory: object
    observer_factory: object
    profiles: tuple[ProviderProfile, ...]
    replay: object
    first_live_validation: bool


def register_runtime(fault, factory, *, observer_factory=None, profiles=(), replay=None,
                     first_live_validation=False):
    capability = resolve(fault)
    if capability.integration not in ('native_provider_bridge', 'automatic_quantization_requires_runtime'):
        raise ValueError('This mechanism does not accept deployment runtime factories: '+fault)
    if not callable(factory): raise TypeError('Runtime factory must be callable')
    profiles = tuple(profiles)
    if observer_factory is not None or profiles:
        if capability.integration != 'native_provider_bridge' or not callable(observer_factory):
            raise ValueError('Native adaptation profiles require a deployment-owned read-only observer factory')
        from .native import OPERATORS
        if any(not isinstance(profile, ProviderProfile) or profile.operator not in OPERATORS[fault]
               for profile in profiles):
            raise ValueError('Profile operator does not belong to this mechanism')
    if type(first_live_validation) is not bool or (first_live_validation and replay is None):
        raise ValueError('First live validation requires an explicit qualified worker replay')
    if replay is not None:
        operators = {profile.operator for profile in profiles}
        if (not callable(replay) or len(operators) != 1
                or not operators <= _QUALIFIED_OPERATORS.get(fault, set())):
            raise ValueError('Qualification requires a reviewed Fault20 WeightNz/QMM profile and worker replay')
    with _LOCK:
        if fault in _FACTORIES: raise ValueError('Runtime already registered: '+fault)
        _FACTORIES[fault] = _Binding(factory, observer_factory, profiles, replay, first_live_validation)


def unregister_runtime(fault, factory):
    """Remove this deployment factory, not an already running fault session.

    Controllers must disable/clean their ModelFaultBackend separately. Existing
    sessions keep their concrete provider and remain responsible for its cleanup.
    """
    with _LOCK:
        binding = _FACTORIES.get(fault)
        if binding is None or binding.factory is not factory:
            raise ValueError('Runtime registration ownership mismatch')
        del _FACTORIES[fault]


def resolve_runtime(fault, model, supplied=None):
    if supplied is not None: return supplied
    with _LOCK:
        binding = _FACTORIES.get(fault)
    if binding and binding.profiles:
        raise UnsupportedCall('Profiled runtime requires a live, lossless-verified deployment binding. '
                              'Use arm=observe to resolve a candidate.')
    return binding.factory(model) if binding else None


def resolve_observer(fault, model, supplied=None):
    if supplied is not None:
        # A custom observer cannot inherit another deployment's validation.
        return supplied, ()
    with _LOCK:
        binding = _FACTORIES.get(fault)
    if binding is None or binding.observer_factory is None:
        return None, ()
    observer = binding.observer_factory(model)
    if observer is None:
        raise UnsupportedCall('Registered observer factory did not return a read-only adapter')
    return observer, binding.profiles


def registered_profiles(fault):
    with _LOCK:
        binding = _FACTORIES.get(fault)
    return binding.profiles if binding else ()


def qualified_binding(fault):
    with _LOCK:
        binding = _FACTORIES.get(fault)
        return binding if binding and binding.replay is not None else None


def coverage_capability(fault):
    """Expose the current bound provider's policy without granting admission."""
    resolve(fault)
    if qualified_binding(fault) is None:
        return CoverageCapability()
    return CoverageCapability(single_target=True, max_concurrency=1, recovery_scope='single_target',
                              stop_policy='one_attempt')


def registered_runtimes():
    with _LOCK: return sorted(_FACTORIES)
