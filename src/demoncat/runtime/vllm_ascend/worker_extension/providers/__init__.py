"""Compatibility entry points for the common Worker provider registry."""


def provider_capabilities():
    from ..registry import rpc_provider_capabilities
    return rpc_provider_capabilities()


def resolve_provider(mechanism, implementation_version, worker, model, identity):
    from ..registry import create_worker_provider
    return create_worker_provider(mechanism, implementation_version, worker, model, identity)
