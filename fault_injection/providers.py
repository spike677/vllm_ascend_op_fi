"""Deployment-owned native runtime registration, distinct from model selection.

Factories are trusted Python integrations, not evidence of native correctness.
They receive the model and must perform ABI/loaded-build checks themselves.
There is no implicit import, address setting or vendor loading here.
"""
import threading

from .catalog import resolve

_LOCK = threading.Lock()
_FACTORIES = {}


def register_runtime(fault, factory):
    capability = resolve(fault)
    if capability.integration not in ('native_provider_bridge', 'automatic_quantization_requires_runtime'):
        raise ValueError('This mechanism does not accept deployment runtime factories: '+fault)
    if not callable(factory): raise TypeError('Runtime factory must be callable')
    with _LOCK:
        if fault in _FACTORIES: raise ValueError('Runtime already registered: '+fault)
        _FACTORIES[fault] = factory


def unregister_runtime(fault, factory):
    """Remove this deployment factory, not an already running fault session.

    Controllers must disable/clean their ModelFaultBackend separately. Existing
    sessions keep their concrete provider and remain responsible for its cleanup.
    """
    with _LOCK:
        if _FACTORIES.get(fault) is not factory: raise ValueError('Runtime registration ownership mismatch')
        del _FACTORIES[fault]


def resolve_runtime(fault, model, supplied=None):
    if supplied is not None: return supplied
    with _LOCK:
        factory = _FACTORIES.get(fault)
    return factory(model) if factory else None


def registered_runtimes():
    with _LOCK: return sorted(_FACTORIES)
