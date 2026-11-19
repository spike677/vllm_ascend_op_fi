"""Read-only historical references; never an admission or a mode switch."""
from copy import deepcopy

_ENTRIES = {
 '28.fa.legal_tiling':dict(path_type='HF_SDPA_TO_NATIVE_FA',state='HISTORICAL_ONLY',
     commit='96aa4d874fa296a7f2a489ac931f75250950fd15',
     evidence='artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029/manifest.json',
     manifest_sha256='db24066e8265bf44b457030e935b2c9dc8a8787dee92312601a0bb52159df7c4',
     geometry='Qwen3-0.6B; B1/B8; S2048; FP16; pinned FA branch'),
 '28.fia.legal_tiling':dict(path_type='EXPERIMENT_MODEL_ADAPTER_FIA',state='HISTORICAL_ONLY',
     commit='13217d6c6b97e77e6857d92615ce5b28c433160b',evidence='reports/acceptance.json',
     geometry='Qwen3-0.6B; B8/S2048/BNSD/Hq16/Hkv8/D128; FP16; CANN9.1.0; A2'),
 '28.fia.denominator_replay':dict(path_type='EXPERIMENT_MODEL_ADAPTER_FIA',state='TND_FAULT_IMPLEMENTATION_INCOMPLETE',
     commit='f879af9bd19a4c1ca099c81c515e4c33cbee20c7',evidence='reports/formal/review.json',
     geometry='Qwen3-0.6B; B8/S2048/BNSD/Hq16/Hkv8/D128; M128/S2=128; FP16; CANN9.1.0; A2'),
}

def research_entry(mechanism):
    row=_ENTRIES.get(mechanism)
    if row is None:return None
    return deepcopy(dict(row,delivery_capability=False,qualification_granted=False,
        historical_worker_proof=False,explicit_research_call_required=True,
        public_configuration_switch=False,
        note='Research retained, not a delivery capability. Historical model adapter is not a real vLLM Worker acceptance.'))

def refuse_public_historical(mechanism):
    row=_ENTRIES.get(mechanism)
    if row is not None:
        from .core import UnsupportedCall
        raise UnsupportedCall(row['state']+':EXPLICIT_RESEARCH_ENTRY_REQUIRED')
