# DynamicQuant batch-shared scale

Generated from registry.yaml. Historical evidence is bounded to the recorded scope.

**Fault ID**: FQUANT

**Mechanism ID**: FQUANT.BATCH_SHARED_SCALE

**Compatibility ID**: quant.batch_shared_scale

**Fault Name**: DynamicQuant batch-shared scale

**Fault Family**: QUANTIZATION_ERROR

**Root Cause**: Incorrect quantization reduction scope couples independent request rows through a shared scale.

**Simulated Semantics**: native quantization reduction

**Injection Primitive**: QUANTIZATION_REDUCTION_SCOPE

**Injection Layer**: native

**Operator**: DynamicQuant

**Model Path**: Observed runtime path required; no model-name inference.

**Trigger Condition**: After actual eligibility and target filtering; see common API contract.

**Parameters**: {"quant_scope": "batch", "scale_policy": "shared", "projection_scope": "observed"}

**Parameter Schema**: schemas/mechanisms/fquant.batch_shared_scale.schema.json

**Observable Effect**: Quantized output/scale can change; distinguish backend scope error from anomalous prompt-induced batching.

**Recovery**: Stop admission, drain owned calls, restore exact owned state, verify healthy replay.

**Evidence**: experiments/c2c3_closure_20261117/C2_C3_MATRIX.md|docs/overnight_fault_cards/quant.batch_shared_scale.md

**Limitations**: CANN9.1 pinned runner and TLS native provider; first observed eligible module, actual scheduler/healthy-input mapping; not all-layer coverage or input-induced batching

**Unified Execution**: NOT_IMPLEMENTED: Legacy implementation retained; unified parameter/trigger/lease bridge not yet implemented. No injection through this API.
