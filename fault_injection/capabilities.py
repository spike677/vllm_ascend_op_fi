"""Machine-readable integration scope, separate from execution acceptance.

This registry does not arm hooks, load vendor libraries or initialize a device.
All 17 descriptions derive from fault_registry. Fault39 keeps its independent
execution package and the legacy host binding contract remains unchanged.
"""
from copy import deepcopy
from dataclasses import asdict

from .catalog import audit_mechanism, list_faults, resolve
from .coverage import CoverageCapability

INTEGRATION_STATUS = 'INNER_FRAMEWORK_INTEGRATION_PENDING'

from fault_registry import definitions

_DETAILS = {row['legacy_id']: tuple(row['legacy_details']) for row in definitions()}


def describe_mechanism(name):
    """Return a fresh JSON-safe definition without upgrading historical proof."""
    capability = resolve(name)
    status, mechanism, restriction, observable = _DETAILS[name]
    return dict(schema='demoncat.mechanism.capability.v1', name=name,
        mechanism=mechanism, boundary=capability.boundary,
        integration_status=INTEGRATION_STATUS, implementation_status=status,
        target='auto', user_layer_names_required=False,
        dispatch_adapter_present=status in ('automatic_adapter', 'deployment_provider_required'),
        runtime_factory_required=capability.integration in (
            'native_provider_bridge', 'automatic_quantization_requires_runtime'),
        native_launch_verified=False, current_model_npu_accepted=False,
        validation=dict(historical_scope=capability.validation,
                        new_routing_scope='CPU tests only; no inherited NPU or model-quality acceptance'),
        restrictions=restriction, expected_observability=observable,
        audit=audit_mechanism(name), coverage=asdict(CoverageCapability()),
        evidence_contract=dict(task='statistics', level='mix', summary_mode='statistics',
            public_groups=['baseline', 'fault'], raw_files=['dump.json', 'stack.json', 'construct.json'],
            required_identity=['config', 'collector_identity'], rtol=1e-5, atol=1e-8,
            comparison='abs(F-B) > atol + rtol*abs(B)', preserve_null=True,
            preserve_actual_steps=True, use_injection_logs_to_choose_first_difference=False,
            numeric_first_difference_proves_unique_mechanism=False))


def list_mechanisms():
    """All seventeen definitions; independent packages remain separate executors."""
    return [describe_mechanism(row['name']) for row in list_faults()
            if row['name'] in _DETAILS]


def inspect_support(model, name):
    """Read actual model structure; never call forward or a provider factory.

    Candidate discovery is deliberately not a claim that a call, launch, or
    request was observed. Positive results still require adapter runtime gates.
    """
    if name.startswith('39.'):
        return dict(schema='demoncat.mechanism.support.v1', capability=describe_mechanism(name),
                    status='INDEPENDENT_BACKEND_REQUIRED', candidates=[],
                    provider_loaded_or_called=False, hooks_installed=False,
                    current_model_npu_accepted=False)
    from . import inspect_model
    spec = describe_mechanism(name)
    inventory = inspect_model(model)
    capability = resolve(name)
    candidates = []
    kind = capability.integration
    if kind == 'automatic_module':
        candidates = inventory['module_candidates'][name]
    elif kind == 'native_provider_bridge':
        from .native import TARGET_KINDS
        candidates = [c for c in inventory['native']['candidates']
                      if set(c['kinds']) & TARGET_KINDS[name]]
    elif kind == 'automatic_attention':
        # Attention discovery exposes its own confirmed/rejected structure
        # records; it does not guarantee the pinned vendor can handle Q/K/V.
        candidates = inventory.get('candidates', [])
    elif kind == 'automatic_quantization_requires_runtime':
        candidates = inventory['quantization'].get('candidates', [])
    factory = name in inventory['registered_runtime_factories']
    if spec['implementation_status'] in ('not_implemented', 'legacy_evidence_only', 'strategy_bridge_only'):
        status = spec['implementation_status'].upper()
    elif spec['runtime_factory_required'] and not factory:
        status = 'DEPLOYMENT_PROVIDER_REQUIRED'
    elif kind == 'automatic_operator':
        status = 'RUNTIME_OBSERVATION_REQUIRED'
    else:
        status = 'STATIC_CANDIDATE_FOUND' if candidates else 'NO_SUPPORTED_CANDIDATE'
    result = dict(schema='demoncat.mechanism.support.v1', capability=spec, status=status,
        candidates=deepcopy(candidates), runtime_factory_registered=factory,
        provider_loaded_or_called=False, hooks_installed=False, native_calls_observed=0,
        runtime_call_validation_required=True, current_model_npu_accepted=False)
    if kind == 'native_provider_bridge':
        from .providers import coverage_capability, registered_profiles

        result['capability']['coverage'] = asdict(coverage_capability(name))
        result['adaptation'] = dict(
            profiles=[asdict(profile) for profile in registered_profiles(name)],
            status='RUNTIME_OBSERVATION_REQUIRED',
        )
    return result
