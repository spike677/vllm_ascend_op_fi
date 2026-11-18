# Independent offline review of the source46 RoPE result

`review_natural_rope_result.py` reads one complete `result.json` and prints one
compact JSON report. It uses only the Python standard library. It does not import
Torch, vLLM, a provider, or a device runtime; start a Worker; open a container;
execute a model; modify the result; or consume raw statistics.

The reviewer is scoped to the frozen `11607cda` probe/provider code and the
`LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW` variant. The expected raw
file hash is mandatory. For the root's original source46 result:

```sh
python3 -B review_natural_rope_result.py "$RESULT" \
  --expected-sha256 2e92fcb0e841ecb92be8c790160f9f11607fe48eb70000b006a562ac0d7ee497 \
  --expected-targets 28
```

The actual denominator comes from every natural reader and its independent
operator counts, structural sites, outer calls, and Triton launches. The count
argument is an additional assertion, not the source of the denominator. The
reviewer checks:

- Four fresh healthy phases with identical complete input, hidden and vocabulary
  output identities, plus the separately recorded scheduler-to-driver mapping.
- Captured input byte digests against scheduler token/position vectors; unique
  real request IDs and invocations, matching Worker/model/rank/device identities.
- Each source-pinned outer reader, actual Triton JIT entry and launcher, exact
  positions/cache objects, Q/K storage spans, scalar geometry, and returned
  consumers of both actual Q/K views.
- Every F changes the final legal position to the first legal position in a
  different allocation and changes the complete Q/K result. Original positions,
  cache identity/bytes, and all non-target input/output rows remain unchanged.
  Propagated changes to later target rows are allowed and remain visible.
- Every C2 target is hit once and every C3 target twice, with exact original-call
  counts, owner/generation, target identities and request tickets. It recomputes
  discovery, observations, execution and effect hashes from the full records.
- Each OFF reproduces full input, hidden and logits hashes; every target and
  Worker has matching drain/release/health receipts and zero recorded resources;
  final host cleanup and the post-cleanup healthy sampled token agree.

`contract_satisfied`, `complete` and headline hit counts never supply the verdict.
Counts are reconciled with individual evidence records. Duplicate JSON keys,
non-finite JSON constants, missing fields, mismatched raw SHA, unknown sources,
incomplete observations and forged join flags with inconsistent metadata fail
closed. Only the report is printed; large source bodies and tensor records are
not copied into the review output.

The source46 probe does **not** export the original live C2Proof object's
issuance/use receipt. Its pinned code obtains the proof from the owning C2
session and the frozen runtime checks exact issued-object identity before C3.
If all C3 execution checks pass, the report records that source-bound inference
as `CONSISTENT_WITH_ORIGINAL_LIVE_C2_GATE_IN_FROZEN_11607CDA`. It does not claim an
independent observation of the original Python capability object.

Accordingly, the old artifact can pass `c2_execution_evidence_valid` and
`c2_review_passed`, and independently pass `c3_execution_evidence_valid`, while
`c3_review_passed` remains false with
`C3_PROOF_ISSUANCE_USE_RECEIPT_MISSING`. This missing observation does not negate
the recorded C3 execution or create another mechanism gate. A later runtime
receipt requires a new run and a reviewed receipt schema; adding a fabricated
JSON proof to the old result is not accepted. No serialized report recreates a
live capability. The reviewer itself never grants public qualification.

Exit code 2 means rejected or incomplete evidence, including the known old C3
receipt gap; it does not imply that the natural C3 execution failed. No raw
device ABI payload, integer-overflow reproduction, or statistics acceptance is
claimed. Digest comparisons rely on the recorded full-tensor observations and
the externally pinned artifact origin; the reviewer does not possess those
device tensors.

Fifteen new CPU tests generate their baseline through the existing real CPU
RoPE forward/controller fixtures and then corrupt individual evidence edges.
The new tests and seven serving fixture tests pass together (22 tests). The
private test-only CPU mode never grants Worker qualification, and the CLI has no
simulation override.
