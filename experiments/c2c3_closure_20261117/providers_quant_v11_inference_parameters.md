# Source74 inference parameter row provenance

The source74 controlled census completed two healthy B8 decodes with 270
actual Native parents and projection calls, zero fault calls, and six
row-proven projection calls. Every request still retains all 135 candidates.
The new before/after AddRms diagnostics identify the first remaining failure:
both actual row inputs have current, equal request-row proofs, but gamma is
the actual `model.layers.0.post_attention_layernorm.weight` Parameter with
`requires_grad=True` and no broadcast registration. The only failed gate is
`gamma_broadcast_proven`. Neither gamma nor its metadata/write epoch changes
during that call. This is evidence about this recorded call; it does not
qualify the remaining projections.

`register_inference_broadcast_parameter` adds a separate observation route
for such inference weights. The previous frozen-parameter entry point still
rejects `requires_grad=True`. The new route requires an active observer on
its original thread, disabled gradient recording, a real Parameter in its
owner's `_parameters`, and the exact owner reached through the held model's
actual `_modules` path. Every module on that path must be in eval mode.
Registration happens before any observed operation in the current forward.
The implementation never calls `eval()`, changes `requires_grad`, or writes
parameter data.

Each use rechecks the live model path and parameter object, current thread
and forward, eval flags, gradient and inference-mode state, the original
autograd flag, complete tensor/storage/version metadata, and observed-write
epoch. An inference Tensor without a version counter still loses permission
after an observed write through any storage alias. Changing owner, storage,
context or epoch cannot be repaired by registering again after calls begin.
The existing row-free broadcast geometry check remains mandatory. Actual
request rows are still derived from producer objects and scheduler evidence;
parameter shape alone never supplies request identities or order.

The AddRms before/after diagnostics retain the registration and fresh context
checks, including concrete refusal names. Pointwise broadcast and embedding
producer evidence also retain the context when they use this new route.
The existing source-bound AddRms rule still needs both current input row
proofs, equal row order, its exact operator/schema and valid actual outputs.
Unknown Attention/MoE edges remain unproven.

Validation uses CPU contracts with real Torch TLS, dispatcher and Parameter
objects. Positive cases cover `inference_mode` and `no_grad`, unchanged
requires-grad flags, source-bound AddRms propagation and the existing
Runtime/provider seven-phase C1 test with an explicitly fake Native backend.
Negative cases cover training, gradient recording, changed inference mode,
foreign thread, wrong/replaced owner path, parameter/storage replacement,
versioned and versionless alias writes, late re-registration, request-axis parameters
and stale forward registrations. These tests are not NPU qualification.

Local Torch 2.10 CPU validation passed 168 tests across the row, source,
metadata, inventory and C1 suites. After adding the same-Parameter storage
replacement negative case, the final inference/AddRms/embedding subset passed
43 tests (19 new inference-parameter contracts). Production Torch 2.9 and
natural-request validation remain for the new frozen package.

Root should freeze a new Python source/package and run the existing
controlled two-healthy, zero-fault
`providers_quant_inventory_probe.py --group-abi` case. Use the unchanged
source64 Native build
`b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`;
this Python-only change requires no SDK rebuild. This entry point is an
observation probe, not public Core C2/C3 acceptance or the seven-phase C1
probe. Record all 135 candidates, every actual Native/public call, the first
remaining refusal, healthy repeat equality and zero owned observer resources.
No new C1/C2/C3 qualification is granted by this increment.
