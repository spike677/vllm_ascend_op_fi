"""Trusted Python site registration. No import strings or user code loader.

External sites share ProductProvider, AutomaticRequests, RegionLeases and the
existing WorkerFaultRuntime. Registration is inert and grants no qualification.
Version one is serial text on the already source-validated runner/model routes.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import re

from .config import FaultConfig
from .serving_context import require

ATTRIBUTE = '_demoncat_external_product_sites'


@dataclass(frozen=True)
class ExternalFaultConfig(FaultConfig):
    """Constructed by trusted host code, not FaultConfig.from_dict or CLI JSON."""
    def __post_init__(self):
        if (self.fault_id != 'external' or type(self.variant) is not str
                or re.fullmatch(r'[a-z][a-z0-9_]{0,63}', self.variant) is None
                or self.implementation_version != '1'):
            raise ValueError('EXTERNAL_EXPLICIT_NAMED_VARIANT_V1_REQUIRED')


@dataclass(frozen=True)
class ExternalSite:
    mechanism: str
    factory: object
    source_sha256: str
    expected_deployment: dict
    stage: str = 'model'
    region_kind: str = 'transient'


def factory_sha256(factory):
    require(inspect.isfunction(factory) or inspect.isclass(factory), 'EXTERNAL_LOCAL_PYTHON_FACTORY_REQUIRED')
    source = inspect.getsourcefile(factory)
    require(source is not None, 'EXTERNAL_SOURCE_FILE_REQUIRED')
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def register_external_site(worker, spec, *, authority, operator):
    """Only trusted deployment bootstrap calls this with an actual code object."""
    from .deployment_authorization import authority_for
    require(authority is authority_for(worker) and authority is not None
            and authority.authorize_operator(operator, 'register_external_site') is True,
            'EXTERNAL_TRUSTED_DEPLOYMENT_AUTHORITY_REQUIRED')
    require(type(spec) is ExternalSite, 'EXTERNAL_TYPED_SPEC_REQUIRED')
    require(spec.stage in ('model', 'sampler', 'vision'), 'EXTERNAL_STAGE_UNSUPPORTED')
    require(spec.region_kind == 'transient', 'EXTERNAL_V1_PERSISTENT_WEIGHT_OR_KV_RESTORE_UNPROVEN')
    require(spec.mechanism.startswith('external.'), 'EXTERNAL_CANNOT_SHADOW_BUILTIN')
    ExternalFaultConfig('external', spec.mechanism.removeprefix('external.'))
    require(type(spec.expected_deployment) is dict and bool(spec.expected_deployment), 'EXTERNAL_DEPLOYMENT_IDENTITY_REQUIRED')
    require(factory_sha256(spec.factory) == spec.source_sha256, 'EXTERNAL_SOURCE_SHA_MISMATCH')
    from .deployment_acceptance import observe_deployment
    actual = observe_deployment(worker, worker.model_runner.model)
    # Full identity, not a subset selected to omit inconvenient differences.
    require(actual == spec.expected_deployment, 'EXTERNAL_DEPLOYMENT_IDENTITY_MISMATCH')
    require(not getattr(worker, '_demoncat_continuous_bindings', {}), 'EXTERNAL_REGISTER_REQUIRES_IDLE_WORKER')
    sites = getattr(worker, ATTRIBUTE, {})
    require(type(sites) is dict and spec.mechanism not in sites, 'EXTERNAL_SITE_ALREADY_REGISTERED')
    sites[spec.mechanism] = (ExternalSite(spec.mechanism, spec.factory, spec.source_sha256,
                                        deepcopy(spec.expected_deployment), spec.stage, spec.region_kind), worker.model_runner.model)
    setattr(worker, ATTRIBUTE, sites)
    return dict(status='EXTERNAL_SITE_REGISTERED_NOT_QUALIFIED', mechanism=spec.mechanism,
                source_sha256=spec.source_sha256, qualification_granted=False)


def external_site(worker, mechanism, *, validate=False):
    pair = getattr(worker, ATTRIBUTE, {}).get(mechanism)
    if pair is None:
        return None
    spec, model = pair
    if validate:
        from .deployment_acceptance import observe_deployment
        require(worker.model_runner.model is model, 'EXTERNAL_MODEL_IDENTITY_CHANGED')
        require(factory_sha256(spec.factory) == spec.source_sha256, 'EXTERNAL_SOURCE_SHA_MISMATCH')
        require(observe_deployment(worker, model) == spec.expected_deployment, 'EXTERNAL_DEPLOYMENT_IDENTITY_MISMATCH')
    return spec


def external_capabilities(worker):
    return [dict(mechanism=name, implementation_version='1', attachment='trusted_external_site',
                 execution='external_product_binding_and_current_permit_required',
                 source_sha256=spec.source_sha256, stage=spec.stage, region_kind=spec.region_kind, qualification_granted=False)
            for name, (spec, _) in sorted(getattr(worker, ATTRIBUTE, {}).items())]


def create_external_provider(worker, model, identity, mechanism, version):
    from .providers.product import ProductProvider
    from .deployment_acceptance import deployment_acceptance_report
    spec = external_site(worker, mechanism, validate=True)
    require(spec is not None and version == '1', 'EXTERNAL_SITE_NOT_REGISTERED')
    report = deployment_acceptance_report(mechanism, worker, model)
    # No static public record can implicitly authorize external code.
    require(report['can_prepare'] and report['acceptance_permit'] is not None,
            'EXTERNAL_CURRENT_ACCEPTANCE_PERMIT_REQUIRED')
    provider = ProductProvider(worker, model, identity, mechanism)
    provider._deployment_admission = (worker, model, report)
    return provider


def make_external_site(binding, *, fault, region_leases):
    spec = external_site(binding.worker, binding.mechanism, validate=True)
    require(spec is not None, 'EXTERNAL_SITE_NOT_REGISTERED')
    site = spec.factory(binding.model, binding.worker.model_runner, fault=fault, region_leases=region_leases)
    require(all(callable(getattr(site, k, None)) for k in ('install', 'close', 'finish_forward'))
            and all(hasattr(site, k) for k in ('events', 'sites', 'forward_calls', 'pending', 'uncovered')),
            'EXTERNAL_SITE_PROTOCOL_INCOMPLETE')
    require(getattr(site, 'region_leases', None) is region_leases, 'EXTERNAL_SHARED_LEASE_MANAGER_REQUIRED')
    if spec.stage == 'sampler':
        require(all(callable(getattr(site,k,None)) for k in ('begin_sample','finish_sample')),
                'EXTERNAL_SAMPLE_PROTOCOL_INCOMPLETE')
    return site
