# Native26 single-site C1 effects exploration

`providers26_group_probe.py --stage c1_single_effects` with an explicit
`--mechanism 26.residual_stomp` (QMM) or `--mechanism 26.fia_destination` (typed FIA)
uses the existing serving adapter, `Writer26Experiment`, Native v2 leases,
completion receipts, and stop/drain/restore/ack lifecycle. It adds no operator,
Worker creation, address-based recovery, or fault retry. This is a new source
snapshot requiring actual device execution and independent review; CPU contracts
do not establish Native qualification. FIA requires the current typed v3 bridge
and exact fresh build profile; an untyped or older bundle is refused before
installation.

The standard host driver enqueues one fixed fresh prefill at a time. RPC `advance`
only arms the next already-enqueued request; model execution remains in the
standard inference request. The eight observed requests are:

1. Complete healthy inventory.
2. A second complete healthy inventory with matching whole-request output.
3. Selected site's current reference execution.
4. Its destination-lossless execution.
5. Its OFF execution and full-output equality; then Native seals exactly that
   qualified lease.
6. One fault execution at the same actual site.
7. Its OFF execution and complete equality to the healthy reference.
8. Same-Worker full-vocabulary health after Native removal.

The existing probe also runs ordinary standard-generation controls before binding
and after removing the session. Those two controls do not replace any of the
eight instrumented requests or execute a model from inside an RPC.

The two healthy inventories retain **every actual Native writer** (including
unresolved and unsupported rows). There is no fixed 135 assertion: if the actual
model exposes 135 calls, all 135 remain in every raw request row and in the frozen
inventory. Likewise, a 30-call FIA inventory with 29 resolved candidates keeps
the first call lacking an actual preceding producer as an ineligible row in
every request. Neither count is hardcoded as an admission rule.
All registered module aliases are retained separately; aliases never
create extra target leases. Among the twice-healthy, uniquely resolved candidates,
the minimum actual target key selects exactly one site within that Worker
lifetime. The key binds the actual module object, forward implementation, device,
format and geometry, plus QMM's actual cache/registration. FIA also checks its
current typed GWS/execute records and actual public output. No target/layer/address/rank input
is accepted. No failed selection is replaced or silently skipped.

`max_targets` keeps its existing meaning for the legacy lossless/effects and group
paths: a complete resolved set exceeding that budget is rejected, never sliced.
The explicit new mode has an independent fixed one-site plan. Setting
`--max-targets 1` on the old `effects` stage still rejects a multi-site set. It is
not a selector. The new mode still checks the independent Native trace capacity
for the six remaining complete requests after the two healthy requests. Typed
FIA also budgets every stock record in those complete requests plus five owned
before-execute records for Reference/Lossless/OFF/F/OFF. Both observer capacities
remain 32768. The source59 all-candidate effects capacity refusal remains its
original evidence; the single-site plan does not turn it into coverage acceptance.

Before FIA selection can be qualified, the existing bridge retains two complete
unmodified public outputs for **every** observed call, including ineligible rows.
For 30 actual calls these are 60 full-byte samples. The selected site's new owned
Reference must match both complete stock samples before Lossless/OFF or F; no old
Reference is reused. Native format 47 remains NCL, while public format 2 remains
ND. The current typed descriptor preserves its exact storage rank and dimensions,
including a rank-one flat storage only when its complete contiguous byte span
equals the rank-three view. The existing bounded geometry, actual public return,
consumer-time snapshot, owner/token and restoration checks are unchanged.

Current module aliases, selected Native formals, actual producer/consumer and
writer Tensor/storage owners, and allocator adjacency are checked again during
the real request. The entire writer denominator is retained even on failure.
This new mode also requires its output observer to remain the final local logits
hook, checked before and after that real call; a later return-transforming hook
is refused instead of comparing an intermediate output.
Missing or repeated selected occurrences, changed bindings/geometry, missing
consumer completion, failed lossless/OFF equality, and incomplete restoration stop
the plan. Existing pending leases retain their public storage owners and first
failure receipt. Explicit cleanup retries can restore only those still-current
objects; rebinding a Tensor to another storage refuses the write. A Native release
already completed before a failing acknowledgement is read back without writing
through a released lease.

`writer_inventory.json` contains the original full candidates, aliases, Native
events, callbacks, allocator artifact references, producer/next-consumer rows,
reference/lossless/OFF/fault/OFF outcomes, Native owner/generation, full output and
residual hashes, changed byte counts, and recovery/release receipts. Its
`single_site_c1` section names the actual mechanism and scope and links the frozen complete inventory and alias hashes to
the selected site and the freshly completed B/L/OFF outcomes. These JSON records
are audit evidence, never a serializable permission or live C2 proof.
`evidence_complete` becomes true only after the full plan and host cleanup;
`qualification_granted`, top-level `c1/c2/c3`, and all-eligible qualification remain
false pending independent review. The probe's successful status is
`SINGLE_SITE_C1_EFFECTS_EVIDENCE_READY_FOR_REVIEW_NOT_C2`. A local residual effect
does not require the final output to differ; the separate full-output comparison
is recorded. E5 raw-statistics acceptance remains independent.

Prepare a root-reviewed QMM case from an immutable installed source and Native bundle:

```sh
python3 -B experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py \
  --installed /reviewed/installed --native /reviewed/native \
  --probe /reviewed/providers26_group_probe.py --source-commit FULL_REVIEWED_COMMIT \
  --mechanisms 26.residual_stomp --stages c1_single_effects --card 2 \
  --out /reviewed/qmm-single-site-c1-cases.json
```

For typed FIA, use matching new Python packages and the reviewed typed flat-storage
Native bundle. The preparer validates its existing exact typed profile:

```sh
python3 -B "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py" \
  --installed "$INSTALLED" --native "$NATIVE_BUNDLE" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_group_probe.py" \
  --source-commit "$COMMIT" --mechanisms 26.fia_destination \
  --stages c1_single_effects --max-targets 1 --card 2 \
  --out "$R/providers26-fia-single-site-c1-cases.json"
```

The FIA case retains the existing SmolLM2 model, fixed fresh-prefill workload and
full denominator. Its scope is
`ONE_ACTUAL_TYPED_FIA_SITE_FIXED_FRESH_PREFILL_NOT_ALL_ELIGIBLE_NOT_C2_C3`;
the original QMM scope is unchanged. The Native implementation is reused without
an ABI, capacity or vendor change. A successful local fixture or prepare command
does not grant actual C1 acceptance.

This command only generates the case; root controls actual scheduling. Complete
observations, stdout/stderr, arrays and allocator details remain under server 124
`/evidence`; public Git contains source, contracts and summaries only. This mode
does not change the group all-eligible gates or claim 17/17, 9.x/A3, or private
downstream integration acceptance.

CPU validation uses real Torch module execution and hooks with explicit Native,
allocator, scheduler and metadata doubles. The original QMM suite passed 87 tests,
including complete 135-call rows, alias preservation, trace capacity, refusal of
changed current objects/formals/allocator pairs, failed B/L/OFF and effect receipts,
foreign completion owners, pending-storage rebinding, before/after restore and ack
failures, final hook ordering, and the unchanged candidate/group negative cases.
The typed FIA extension adds 12 contracts for full inventories and flat storage,
fresh complete-byte Reference, changed current descriptor refusal, capacity,
real standard serving requests and owner-preserving failed cleanup/retry:

```sh
python -B -m unittest test_writer26_fia_single_site test_writer26_fia_typed \
  test_writer26_single_site test_writer26_group_experiment \
  test_writer26_pending_consumer test_writer26_coverage test_writer26_public_return \
  test_writer26_inventory test_writer26_loader test_writer26_group_build \
  test_writer26_metadata_source_closure
```

This combined regression passed 122 CPU tests, including the existing typed,
QMM, all-candidate, group and metadata-source contracts.

Use the repository's existing CPU environment and put `src`, `fault39/src`, the
repository root, `tests`, `tests/worker_extension`, and `fault39/tests` on
`PYTHONPATH`; set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. These tests never establish a
real Native hit or C1/C2/C3 qualification.
