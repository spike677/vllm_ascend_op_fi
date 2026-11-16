"""Machine-readable integration scope, separate from execution acceptance.

This registry does not arm hooks, load vendor libraries or initialize a device.
Fault39 has its own delivered native package and is deliberately excluded from
this remaining-mechanisms contract. Legacy attach names remain compatible.
"""
from copy import deepcopy
from dataclasses import asdict

from .catalog import audit_mechanism, list_faults, resolve
from .coverage import CoverageCapability

INTEGRATION_STATUS = 'INNER_FRAMEWORK_INTEGRATION_PENDING'

_DETAILS = {
    '8.restore_offset': ('legacy_evidence_only', 'paired_moe_restore_mapping',
        'Missing reusable worker; actual routing indices, top-k weights and restore consumer required.',
        'Restore output or its consumer; no automatic msprobe alignment is supplied.'),
    '15.alias_detach': ('automatic_adapter', 'private_format_view_alias_detach',
        'NPU NZ29/BF16 16x16 base and public as_strided 8x8/stride16,1/offset17; native internal calls unsupported.',
        'View initial values and copy_ output may agree; original base after write is the first value difference.'),
    '17.spatial_padding': ('automatic_adapter', 'conv3d_padding_argument_proxy',
        'Standard Conv3d; shape-preserving 16x16 patch geometry only; reconstructed CPU adapter, new NPU acceptance required.',
        'Conv3d padding input changes first; tensor output changes afterward. Not equal-input kernel corruption.'),
    '18.api_half': ('automatic_adapter', 'rmsnorm_module_return_gain_proxy',
        'Standard nn.RMSNorm only; gain0.5 at module return, not historical native wrapper or Reduce internals.',
        'Observation order depends on module hooks; native RMSNorm may appear healthy before a downstream input differs.'),
    '20.buffer_substitute': ('deployment_provider_required', 'native_input_buffer_substitute',
        'Matched ABI, semantic slot, live storage and original transform; a Linear candidate proves no slot.',
        'Python may record original caller input; actual native operator output and downstream values require separate checking.'),
    '22.wrong_row': ('legacy_evidence_only', 'index_select_return_row_proxy',
        'Missing reusable worker; real RoPE cache/index/consumer mapping required; not address overflow.',
        'IndexSelect post-return proxy or downstream consumer, depending on collection boundary.'),
    '25.reduce_scatter': ('not_implemented', 'collective_reduction',
        'No accepted dedicated implementation; actual collective, process group and rank semantics are required.',
        'Unspecified until an actual mechanism and observable boundary are implemented.'),
    '26.residual_stomp': ('deployment_provider_required', 'native_output_underflow_proxy',
        'Actual allocator adjacency, live residual and exact writer/consumer lifecycle are mandatory.',
        'Residual consumer may differ while the restored normal output agrees; statistics alone do not prove memory aliasing.'),
    '26.fia_destination': ('deployment_provider_required', 'fia_live_residual_destination_proxy',
        'Actual output destination and live residual consumer proof; not natural OOB; historical consumer coverage incomplete.',
        'Residual read/consumer, after real writer destination diversion; original output can be restored.'),
    'quant.batch_shared_scale': ('deployment_provider_required', 'dynamic_quant_reduction_scope_error',
        'Verified W8A8 native backend plus TP2 pure-decode scheduler and healthy-input identity; no shape[0] request guess.',
        'Quantized output/scale can change; distinguish backend scope error from anomalous prompt-induced batching.'),
    '28.fa.legal_tiling': ('strategy_bridge_only', 'legal_fa_execution_grouping',
        'Existing certified FA S1S2 StrategySession; model-to-native provider not implemented in attach.',
        'FA output may differ or be exactly equal; legal strategy changes need not violate any precision standard.'),
    '28.fia.legal_tiling': ('automatic_adapter', 'legal_fia_coupled_grouping',
        'Pinned FP16 B1/B8 Hq16/Hkv8 S2048 D128 vendor; M/S2 512->128; no inherited new-router NPU acceptance.',
        'FIA output may differ below tolerance; prior legal-strategy search did not reach the0.1% statistical target.'),
    '28.fia.denominator_replay': ('automatic_adapter', 'fia_softmax_denominator_replay',
        'Pinned FP16 B1/B8 Hq16/Hkv8 S2048 D128 vendor; distinct arithmetic error, not historical legal-tiling fault.',
        'FIA output then downstream logits; historical0.7968% Norm applies only to the frozen Qwen experiment.'),
    '33.paged_decode': ('deployment_provider_required', 'paged_attention_reduction_grouping',
        'Native IFA tiling, actual page table, KV lifetime and request mapping required; dense prefill is not equivalent.',
        'Paged attention output then downstream; a FIA API label alone does not identify fault28 versus fault33.'),
    '41.tail_column': ('not_implemented', 'cached_tail_state_corruption',
        'Historical dispatcher interception was blocked; live cache writer/storage/consumer must be proven first.',
        'Unspecified until actual dispatcher and cache lifetime are established.'),
}


def describe_mechanism(name):
    """Return a fresh JSON-safe definition without upgrading historical proof."""
    capability = resolve(name)
    if name not in _DETAILS:
        raise ValueError('Fault39 is provided separately by demoncat_fault39; not in the remaining-mechanisms contract')
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
    """All fifteen contract entries, including explicit unavailable routes."""
    return [describe_mechanism(row['name']) for row in list_faults()
            if row['name'] in _DETAILS]


def inspect_support(model, name):
    """Read actual model structure; never call forward or a provider factory.

    Candidate discovery is deliberately not a claim that a call, launch, or
    request was observed. Positive results still require adapter runtime gates.
    """
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
