"""Capability inventory; a catalog entry is not an executable adapter."""
from dataclasses import asdict, dataclass

MODEL_SESSION_ROUTES = frozenset({
    'automatic_attention', 'automatic_operator', 'automatic_module',
    'automatic_quantization_requires_runtime', 'native_provider_bridge',
})
DEPLOYMENT_RUNTIME_ROUTES = frozenset({
    'automatic_quantization_requires_runtime', 'native_provider_bridge',
})


@dataclass(frozen=True)
class Evidence:
    scope: str
    path: str
    verified: tuple[str, ...]


@dataclass(frozen=True)
class Capability:
    name: str
    boundary: str
    integration: str
    validation: str
    limitation: str
    source: str | None
    worker: str | None
    provider: str | None
    gaps: tuple[str, ...]
    evidence: Evidence | None


def _registry_capabilities():
    from fault_registry import definitions
    for row in definitions():
        values = row['legacy_catalog']
        values['gaps'] = tuple(values['gaps'])
        if values['evidence'] is not None:
            values['evidence']['verified'] = tuple(values['evidence']['verified'])
            values['evidence'] = Evidence(**values['evidence'])
        yield Capability(**values)


CAPABILITIES = tuple(_registry_capabilities())


def list_faults():
    return [dict(name=item.name, boundary=item.boundary, integration=item.integration,
                 validation=item.validation, limitation=item.limitation, audit=_audit(item))
            for item in CAPABILITIES]


def _audit(item):
    from fault_registry import resolve as registry_resolve
    definition = registry_resolve(item.name)
    if item.name.startswith('39.'):
        host_binding = 'fault39.native.matmul' if item.name == '39.matmul_range' else None
    else:
        host_binding = 'model_fault.' + item.name
    return dict(
        unified=dict(canonical_id=definition['canonical_id'], contract_version='1',
                     adapter_status=definition['adapter_status'], reason=definition['adapter_reason']),
        implementation=dict(source=item.source, host_binding=host_binding,
                            worker_entry=item.worker, native_provider=item.provider,
                            model_session_factory_present=item.integration in MODEL_SESSION_ROUTES
                            and not item.name.startswith('39.')),
        evidence=asdict(item.evidence) if item.evidence is not None else None,
        gaps=item.gaps + ('HOST_INTEGRATION_PENDING',),
        production_status='PENDING',
    )


def audit_mechanism(name):
    return _audit(resolve(name))


def resolve(name):
    for item in CAPABILITIES:
        if item.name == name:
            return item
    choices = [x.name for x in CAPABILITIES if x.name.split('.')[0] == str(name)]
    raise ValueError('Choose an explicit mechanism variant: ' + ', '.join(choices or [x.name for x in CAPABILITIES]))
