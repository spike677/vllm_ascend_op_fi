"""Capability inventory; a catalog entry is not an executable adapter."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Capability:
    name: str
    boundary: str
    integration: str
    validation: str
    limitation: str


CAPABILITIES = (
    Capability('8.restore_offset', 'MoE restore indices/top-k mapping', 'evidence_only', 'historical fixed experiment', 'No complete reusable local source'),
    Capability('15.alias_detach', 'as_strided view construction', 'automatic_operator', 'old NPU mechanism; new routing CPU simulation only', 'Public torch.as_strided only; exact NZ29/BF16 geometry, actual copy_ and original-base consumer required; old name 27'),
    Capability('17.spatial_padding', 'Functional.conv3d padding argument', 'automatic_module', 'new native CPU Conv3d tests; historical fixed NPU evidence remains separate', 'New reconstruction from documented proxy; only standard Conv3d with proven shape-preserving 16x16 patch geometry'),
    Capability('18.api_half', 'RMSNorm module return gain proxy', 'automatic_module', 'new CPU module proxy tests; historical native wrapper evidence remains separate', 'Standard nn.RMSNorm only; new module-return boundary, not a native Reduce error or the old wrapper'),
    Capability('20.buffer_substitute', 'CANN executor semantic slot', 'native_provider_bridge', '124 NPU: Qwen3-0.6B and SmolLM2-135M/Llama B/F/OFF; three statistics/mix steps', 'CANN8.5.1 v17 MatmulWeightNz, FP16/BF16 ND input <=65536 bytes; newly tested BF16 with NZ29 weights; no new INT2/INT4 acceptance'),
    Capability('22.wrong_row', 'IndexSelect returned row proxy', 'evidence_only', 'historical proxy', 'No automatic model integration'),
    Capability('25.reduce_scatter', 'collective reduction', 'not_implemented', 'not accepted', 'Do not substitute a different fault'),
    Capability('26.residual_stomp', 'native output allocation', 'native_provider_bridge', '124 NPU: real 32B W8A8 TP2 B/F/OFF, 45 eligible native writes per rank after forward-lifetime fix; two captured steps', 'CANN8.5.1/custom AddRmsNormBias; actual allocator adjacency required; rank0 statistical first difference downstream, rank1 NaN blocks independent earliest localization'),
    Capability('26.fia_destination', 'FIA output destination proxy', 'native_provider_bridge', '124 NPU: real 32B W8A8 TP2 inference B/F/OFF, 126 completed writes per rank; generated tokens/text changed in 4/8 requests', 'CANN8.5.1 FIA V3 and custom AddRmsNormBias; current-forward producer only; raw prefill first difference in consumer residual input, decode tokens already differ'),
    Capability('quant.batch_shared_scale', 'native quantization reduction', 'automatic_quantization_requires_runtime', '124 NPU: real 32B W8A8 TP2 B/F/OFF; DynamicQuant first difference, three steps on both ranks', 'CANN9.1 pinned runner and TLS native provider; first observed eligible module, actual scheduler/healthy-input mapping; not all-layer coverage or input-induced batching'),
    Capability('28.fa.legal_tiling', 'FA S1S2 host policy', 'strategy_bridge', 'pinned FA operator/model matrix', 'Use existing certified StrategySession; not FIA'),
    Capability('28.fia.legal_tiling', 'FIA host M/S2 selection', 'automatic_attention', 'old pinned Qwen3 experiments; new router CPU only', 'Controlled coupled M/S2 512->128 healthy templates; no replay key, no new NPU acceptance, not historical default policy'),
    Capability('28.fia.denominator_replay', 'FIA native online softmax', 'automatic_attention', 'old Qwen3 NPU; new router CPU only', 'New arithmetic error, exact pinned kernel geometry; not historical fault28'),
    Capability('33.paged_decode', 'paged IFA SInner selection', 'native_provider_bridge', '124 NPU: Qwen3 28 layers/B8, 84 actual switches, B/F/OFF and native launch audit', 'CANN9.1 pinned vendor, Q1/KV16384/H16/KV8/D128 FP16/page128; raw statistics first differs at downstream mean, not uniquely at IFA; no free-running generation acceptance'),
    Capability('39.matmul_range', 'native matmul arithmetic', 'native_provider_bridge', 'old pinned finite-input experiment; new bridge CPU simulation only', 'Deployment provider pending; intermediate accumulator type/location not proven'),
    Capability('39.output_proxy', 'native matmul output', 'native_provider_bridge', 'old pinned outcome proxy; new bridge CPU simulation only', 'Deployment provider pending; explicit output perturbation, not arithmetic root cause'),
    Capability('41.tail_column', 'cached tail/state', 'not_implemented', 'design/blocked', 'Real dispatcher integration not completed'),
)


def list_faults():
    return [asdict(item) for item in CAPABILITIES]


def resolve(name):
    for item in CAPABILITIES:
        if item.name == name:
            return item
    choices = [x.name for x in CAPABILITIES if x.name.split('.')[0] == str(name)]
    raise ValueError('Choose an explicit mechanism variant: ' + ', '.join(choices or [x.name for x in CAPABILITIES]))
