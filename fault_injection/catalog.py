"""Capability inventory; a catalog entry is not an executable adapter."""
from dataclasses import asdict, dataclass
from .delivery_scope import describe_delivery

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


CAPABILITIES = (
    Capability('18.variance_over4', 'whole RMSNorm output fixed gain proxy', 'continuous_product_only',
        'CPU development; no inherited NPU qualification',
        'RMS mean-square k=4 gain=0.5 proxy; not centered variance or exact epsilon-dependent kernel replay',
        'fault_injection/sites/norm18_variance.py', None, None,
        ('NATIVE_VALIDATION_PENDING',), None),
    Capability('18.variance_under4', 'whole RMSNorm output fixed gain proxy', 'continuous_product_only',
        'CPU development; no inherited NPU qualification',
        'RMS mean-square k=0.25 gain=2 proxy; not centered variance or exact epsilon-dependent kernel replay',
        'fault_injection/sites/norm18_variance.py', None, None,
        ('NATIVE_VALIDATION_PENDING',), None),
    Capability('8.restore_offset', 'MoE restore indices/top-k mapping', 'evidence_only', 'historical fixed experiment', 'No complete reusable local source',
        None, None,
        None,
        ("MODEL_ADAPTER_MISSING", "NATIVE_PROVIDER_MISSING", "REQUEST_MAPPING_UNPROVEN"),
        Evidence("historical Qwen3-VL TP2",
                 "artifacts/fault8/systemic_restore_offset_20260916/INJECTION_LOGIC.md",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('15.alias_detach', 'as_strided view construction', 'automatic_operator', 'old NPU mechanism; new routing CPU simulation only', 'Public torch.as_strided only; exact NZ29/BF16 geometry, actual copy_ and original-base consumer required; old name 27',
        "fault_injection/alias.py", None,
        None,
        ("STORAGE_PROOF_MISSING", "CONSUMER_PROOF_MISSING", "NATIVE_VALIDATION_PENDING"),
        Evidence("current Python route",
                 "tests/test_alias_auto.py",
                 ("CPU_VERIFIED",))),
    Capability('17.spatial_padding', 'Functional.conv3d padding argument', 'automatic_module', 'new native CPU Conv3d tests; historical fixed NPU evidence remains separate', 'New reconstruction from documented proxy; only standard Conv3d with proven shape-preserving 16x16 patch geometry',
        "fault_injection/module_faults.py", None,
        None,
        ("LAYER_ROLE_UNRESOLVED", "NATIVE_VALIDATION_PENDING"),
        Evidence("current CPU route",
                 "tests/test_module_faults.py",
                 ("CPU_VERIFIED",))),
    Capability('18.api_half', 'RMSNorm module return gain proxy', 'automatic_module', 'new CPU module proxy tests; historical native wrapper evidence remains separate', 'Standard nn.RMSNorm only; new module-return boundary, not a native Reduce error or the old wrapper',
        "fault_injection/module_faults.py", None,
        None,
        ("OPERATOR_ADAPTER_MISSING", "NATIVE_VALIDATION_PENDING"),
        Evidence("standard RMSNorm return, CPU",
                 "tests/test_module_faults.py",
                 ("CPU_VERIFIED",))),
    Capability('20.buffer_substitute', 'CANN executor semantic slot', 'native_provider_bridge', '124 NPU: Qwen3-0.6B and SmolLM2-135M/Llama B/F/OFF; three statistics/mix steps', 'CANN8.5.1 v17 MatmulWeightNz, FP16/BF16 ND input <=65536 bytes; newly tested BF16 with NZ29 weights; no new INT2/INT4 acceptance',
        "fault_injection/native.py", None,
        "fault_injection/deployments/buffer20.py",
        ("TARGET_IDENTITY_UNPROVEN", "STORAGE_PROOF_MISSING", "NATIVE_VALIDATION_PENDING"),
        Evidence("9171f3b v17 WeightNz, build20-a1 only",
                 "experiments/provider_adaptation_20261116/validation_summary.json",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('22.wrong_row', 'IndexSelect returned row proxy', 'evidence_only', 'historical proxy', 'No automatic model integration',
        None, None,
        None,
        ("MODEL_ADAPTER_MISSING", "REQUEST_MAPPING_UNPROVEN", "CONSUMER_PROOF_MISSING"),
        Evidence("historical fixed RoPE",
                 "artifacts/fault22/wrong_row_output_proxy_steps012_20260929/INJECTION_LOGIC.md",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('25.reduce_scatter', 'collective reduction', 'not_implemented', 'not accepted', 'Do not substitute a different fault',
        None, None,
        None,
        ("OPERATOR_ADAPTER_MISSING", "NATIVE_PROVIDER_MISSING", "RECOVERY_UNPROVEN"),
        None),
    Capability('26.residual_stomp', 'native output allocation', 'native_provider_bridge', '124 NPU: real 32B W8A8 TP2 B/F/OFF, 45 eligible native writes per rank after forward-lifetime fix; two captured steps', 'CANN8.5.1/custom AddRmsNormBias; actual allocator adjacency required; rank0 statistical first difference downstream, rank1 NaN blocks independent earliest localization',
        "fault_injection/native.py", "fault_injection/deployments/residual26_worker.py",
        "fault_injection/deployments/residual26.py",
        ("CONSUMER_PROOF_MISSING", "INDEPENDENT_FIRST_DIFF_UNPROVEN"),
        Evidence("provider26-final-a2 QMM eligible TP2 calls only",
                 "experiments/provider_adaptation_20261116/validation_summary.json",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('26.fia_destination', 'FIA output destination proxy', 'native_provider_bridge', '124 NPU: real 32B W8A8 TP2 inference B/F/OFF, 126 completed writes per rank; generated tokens/text changed in 4/8 requests', 'CANN8.5.1 FIA V3 and custom AddRmsNormBias; current-forward producer only; raw prefill first difference in consumer residual input, decode tokens already differ',
        "fault_injection/native.py", "fault_injection/deployments/residual26_worker.py",
        "fault_injection/deployments/residual26.py",
        ("CONSUMER_PROOF_MISSING", "RECOVERY_UNPROVEN"),
        Evidence("provider26-final-a2 FIA eligible TP2 calls only",
                 "experiments/provider_adaptation_20261116/validation_summary.json",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('quant.batch_shared_scale', 'native quantization reduction', 'automatic_quantization_requires_runtime', '124 NPU: real 32B W8A8 TP2 B/F/OFF; DynamicQuant first difference, three steps on both ranks', 'CANN9.1 pinned runner and TLS native provider; first observed eligible module, actual scheduler/healthy-input mapping; not all-layer coverage or input-induced batching',
        "fault_injection/quantization.py", "fault_injection/deployments/batch_scale.py",
        "fault_injection/deployments/batch_scale.py",
        ("REQUEST_MAPPING_UNPROVEN", "TARGET_IDENTITY_UNPROVEN"),
        Evidence("quant-model-a2 first eligible module, B8 pure decode",
                 "experiments/provider_adaptation_20261116/validation_summary.json",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('28.fa.legal_tiling', 'FA S1S2 host policy', 'strategy_bridge', 'pinned FA operator/model matrix', 'Use existing certified StrategySession; not FIA',
        "fault_injection/strategy.py", None,
        None,
        ("OPERATOR_ADAPTER_MISSING", "REQUEST_MAPPING_UNPROVEN"),
        Evidence("separate CPU strategy bridge",
                 "tests/test_strategy_bridge.py",
                 ("CPU_VERIFIED",))),
    Capability('28.fia.legal_tiling', 'FIA host M/S2 selection', 'automatic_attention', 'old pinned Qwen3 experiments; new router CPU only', 'Controlled coupled M/S2 512->128 healthy templates; no replay key, no new NPU acceptance, not historical default policy',
        "fault_injection/fia.py", None,
        None,
        ("NATIVE_VALIDATION_PENDING", "INDEPENDENT_FIRST_DIFF_UNPROVEN"),
        Evidence("current shared Python attention route",
                 "tests/test_fia_masks.py",
                 ("CPU_VERIFIED",))),
    Capability('28.fia.denominator_replay', 'FIA native online softmax', 'automatic_attention', 'old Qwen3 NPU; new router CPU only', 'New arithmetic error, exact pinned kernel geometry; not historical fault28',
        "fault_injection/fia.py", None,
        None,
        ("TARGET_IDENTITY_UNPROVEN", "NATIVE_VALIDATION_PENDING"),
        Evidence("current shared Python attention route",
                 "tests/test_fia_masks.py",
                 ("CPU_VERIFIED",))),
    Capability('33.paged_decode', 'paged IFA SInner selection', 'native_provider_bridge', '124 NPU: Qwen3 28 layers/B8, 84 actual switches, B/F/OFF and native launch audit', 'CANN9.1 pinned vendor, Q1/KV16384/H16/KV8/D128 FP16/page128; raw statistics first differs at downstream mean, not uniquely at IFA; no free-running generation acceptance',
        "fault_injection/native.py", None,
        "fault_injection/deployments/paged33.py",
        ("REQUEST_MAPPING_UNPROVEN", "RECOVERY_UNPROVEN", "INDEPENDENT_FIRST_DIFF_UNPROVEN"),
        Evidence("provider33-model-a2, fixed teacher decode",
                 "experiments/provider_adaptation_20261116/validation_summary.json",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('39.matmul_range', 'native matmul arithmetic', 'native_provider_bridge', 'old pinned finite-input experiment; new bridge CPU simulation only', 'Deployment provider pending; intermediate accumulator type/location not proven',
        "fault39/src/demoncat_fault39/backend.py", "fault39/src/demoncat_fault39/vllm_worker.py",
        "fault39/src/demoncat_fault39/provider.py",
        ("TARGET_IDENTITY_UNPROVEN", "REQUEST_MAPPING_UNPROVEN", "NATIVE_VALIDATION_PENDING"),
        Evidence("independent ABI v1 matched profiles and model cases",
                 "experiments/provider_adaptation_20261116/validation_summary.json",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('39.output_proxy', 'native matmul output', 'native_provider_bridge', 'old pinned outcome proxy; new bridge CPU simulation only', 'Deployment provider pending; explicit output perturbation, not arithmetic root cause',
        None, None,
        None,
        ("HOST_CONTRACT_MISSING", "NATIVE_PROVIDER_MISSING", "STORAGE_PROOF_MISSING", "RECOVERY_UNPROVEN"),
        Evidence("legacy v19 single-hit projection output proxy",
                 "fault20_fast_validation/reports/fault39_float_matmul_output_nonfinite_proxy_validation.md",
                 ("NATIVE_VERIFIED", "MODEL_VERIFIED", "RECOVERY_VERIFIED"))),
    Capability('41.tail_column', 'cached tail/state', 'not_implemented', 'design/blocked', 'Real dispatcher integration not completed',
        None, None,
        None,
        ("OPERATOR_ADAPTER_MISSING", "STORAGE_PROOF_MISSING", "CONSUMER_PROOF_MISSING", "NATIVE_PROVIDER_MISSING", "RECOVERY_UNPROVEN"),
        None),
)


def list_faults():
    return [dict(name=item.name, boundary=item.boundary, integration=item.integration,
                 validation=item.validation, limitation=item.limitation, audit=_audit(item),
                 current_delivery=describe_delivery(item.name))
            for item in CAPABILITIES]


def _audit(item):
    if item.integration == 'continuous_product_only':
        host_binding = None
    elif item.name.startswith('39.'):
        host_binding = 'fault39.native.matmul' if item.name == '39.matmul_range' else None
    else:
        host_binding = 'model_fault.' + item.name
    return dict(
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
