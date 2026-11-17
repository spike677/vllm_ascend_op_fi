"""Unified model discovery and bounded fault execution.

No layer paths, model-family imports or fixed layer counts in the public API.
Unsupported legacy routes are listed, never silently treated as armed faults.
"""
from .catalog import list_faults, resolve
from .core import AttentionCall, Decision, UnsupportedCall
from .providers import (ProviderProfile, register_runtime, unregister_runtime,
                        registered_runtimes, qualified_binding, resolve_observer, resolve_runtime)
from .environment import inspect_environment, environment_fingerprint
from .capabilities import describe_mechanism, list_mechanisms, inspect_support
from .coverage import (CallOutcome, Candidate, CandidateSet, CoverageCapability, CoverageReport,
                       Eligibility, EligibilityResult, GroupRecoveryReport, IndependentObservation,
                       ObservedCall, Outcome, Phase, QualifiedTargetSet, RecoveryReport,
                       TargetIdentity, TargetLease, TargetLosslessProof, coverage_phase,
                       BoundaryTargetIdentity, PythonHookBoundary, TensorAliasBoundary,
                       OperatorApiBoundary, KernelLaunchBoundary, CollectiveBoundary)
from .lifecycle import FaultConfig, ModelFaultBackend


def inspect_model(model, *, runner=None):
    from .attention import inspect_model as inspect_attention
    result = inspect_attention(model)
    from .model_discovery import inspect_structure
    result['structure'] = inspect_structure(model, attention_paths=[
        row['path'] for row in result['candidates']
    ], runner=runner)
    # Structure candidates only: a Linear module is not proof of a native QMM
    # runtime slot or of an allocator relationship.
    import torch
    result['linear_candidates'] = [dict(path=name, type=type(m).__name__,
        input_features=m.in_features, output_features=m.out_features,
        native_binding_verified=False) for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    result['capabilities'] = list_faults()
    from .module_faults import discover_modules
    from .native import inspect_native
    from .quantization import inspect_quantization
    result['module_candidates'] = {fault: [dict(path=p, type=type(m).__qualname__,
        status='candidate_not_observed') for p,m in discover_modules(model, fault)]
        for fault in ('17.spatial_padding','18.api_half')}
    result['quantization'] = inspect_quantization(model)
    result['native'] = inspect_native(model)
    result['alias'] = dict(status='runtime_observation_required',
        boundary='public torch.as_strided inside this model forward',
        note='Actual format/storage/view/copy_/consumer cannot be inferred from static layer names')
    result['registered_runtime_factories'] = registered_runtimes()
    return result


def attach(model, fault, *, arm='fault', unsupported='error', evidence_dir=None, runtime=None,
           _required_binding=None):
    """Return a scoped session; entering it automatically binds actual modules."""
    capability = resolve(fault)
    if not callable(getattr(model, 'named_modules', None)):
        raise UnsupportedCall('Expected an actual model with named_modules(); no fault was installed')
    if arm not in ('baseline', 'fault', 'observe') or unsupported not in ('error', 'passthrough'):
        raise ValueError('Invalid arm or unsupported policy')
    if arm == 'observe' and capability.integration != 'native_provider_bridge':
        raise ValueError('Read-only operator observation is available on native provider bridges')
    common = dict(arm=arm, unsupported=unsupported, evidence_dir=evidence_dir)
    if capability.integration == 'automatic_attention':
        if runtime is not None:
            raise ValueError('FIA uses its pinned vendor launcher, not a runtime provider override')
        from .attention import AttentionSession
        from .fia import FiaBackend
        adapter = AttentionSession(model, FiaBackend(arm=arm, evidence_dir=evidence_dir, mechanism=fault),
                                   unsupported=unsupported)
    elif capability.integration == 'automatic_operator':
        if runtime is not None:
            raise ValueError('Alias detach uses the real NPU format provider; no runtime override')
        from .alias import AliasSession
        adapter = AliasSession(model, **common)
    elif capability.integration == 'automatic_module':
        if runtime is not None:
            raise ValueError('Structural module adapters do not accept native runtime overrides')
        from .module_faults import ModuleFaultSession
        adapter = ModuleFaultSession(model, fault, **common)
    elif capability.integration == 'automatic_quantization_requires_runtime':
        from .quantization import QuantizationSession
        adapter = QuantizationSession(model, runtime=resolve_runtime(fault,model,runtime), **common)
    elif capability.integration == 'native_provider_bridge':
        from .native import NativeSession
        if arm == 'observe':
            observer, profiles = resolve_observer(fault, model, runtime)
            adapter = NativeSession(model, fault, runtime=observer, profiles=profiles, **common)
        else:
            binding = qualified_binding(fault) if runtime is None else None
            if _required_binding is not None and binding is not _required_binding:
                raise UnsupportedCall('Qualified worker binding changed before enable')
            if binding is None:
                adapter = NativeSession(model, fault, runtime=resolve_runtime(fault, model, runtime), **common)
            else:
                from .qualification import QualifiedNativeSession
                adapter = QualifiedNativeSession(model, fault, binding, **common)
    else:
        raise UnsupportedCall(f'{fault}: {capability.integration}; {capability.limitation}. '
                              'Use inspect_model(model) for discovery; no fault was installed.')
    from .session import ModelFaultSession
    return ModelFaultSession(adapter, capability, arm, evidence_dir)


__all__ = ['attach', 'inspect_model', 'list_faults', 'register_runtime', 'unregister_runtime',
           'registered_runtimes', 'AttentionCall', 'Decision', 'UnsupportedCall',
           'describe_mechanism', 'list_mechanisms', 'inspect_support', 'FaultConfig', 'ModelFaultBackend',
           'inspect_environment', 'environment_fingerprint', 'ProviderProfile',
           'CandidateSet', 'Candidate', 'TargetIdentity', 'EligibilityResult', 'Eligibility',
           'CoverageCapability', 'CoverageReport', 'RecoveryReport', 'TargetLease',
           'ObservedCall', 'IndependentObservation', 'CallOutcome', 'Outcome', 'Phase', 'coverage_phase',
           'QualifiedTargetSet', 'TargetLosslessProof', 'GroupRecoveryReport',
           'BoundaryTargetIdentity', 'PythonHookBoundary', 'TensorAliasBoundary',
           'OperatorApiBoundary', 'KernelLaunchBoundary', 'CollectiveBoundary']
