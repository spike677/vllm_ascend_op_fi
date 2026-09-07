# Evaluation

Evaluation compares baseline, fault, and recovery runs using common metrics:
response/token divergence, non-finite propagation, hit/effect repeatability,
request health, and exact recovery.

Evaluation code may provide development runners and report generators. It does
not own the production injection CLI or controller lifecycle.

V0 requires a passed lossless `ValidationResult`, then enforces
`baseline -> fault -> disarm -> recovery`. It records mechanism and model
effects in the shared `EvaluationResult`. Missing token IDs and quality or
performance metrics remain `null`; token pieces may be used as an explicitly
labelled fallback for deterministic divergence analysis.

The live Qwen V1 adapter and phase suite are documented in
`docs/evaluation/qwen_2card_fault_effect_v1.md`. They reuse this contract and
runner boundary; native injection logic is not duplicated in Python.
