# Native26 QMM single-site C1 effects exploration

`providers26_group_probe.py --stage c1_single_effects --mechanism 26.residual_stomp`
uses the existing serving adapter, `Writer26Experiment`, Native v2 leases,
completion receipts, and stop/drain/restore/ack lifecycle. It adds no operator,
Worker creation, address-based recovery, or fault retry. This is a new source
snapshot requiring actual device execution and independent review; CPU contracts
do not establish Native qualification. FIA is rejected for this mode.

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
inventory. All registered module aliases are retained separately; aliases never
create extra target leases. Among the twice-healthy, uniquely resolved candidates,
the minimum actual target key selects exactly one site within that Worker
lifetime. The key binds the actual module object, forward implementation, Native
cache/registration, device, format and geometry. No target/layer/address/rank input
is accepted. No failed selection is replaced or silently skipped.

`max_targets` keeps its existing meaning for the legacy lossless/effects and group
paths: a complete resolved set exceeding that budget is rejected, never sliced.
The explicit new mode has an independent fixed one-site plan. Setting
`--max-targets 1` on the old `effects` stage still rejects a multi-site set. It is
not a selector. The new mode still checks the independent Native trace capacity
for all eight full requests, not just the selected calls.

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
`single_site_c1` section links the frozen complete inventory and alias hashes to
the selected site and the freshly completed B/L/OFF outcomes. These JSON records
are audit evidence, never a serializable permission or live C2 proof.
`evidence_complete` becomes true only after the full plan and host cleanup;
`qualification_granted`, top-level `c1/c2/c3`, and all-eligible qualification remain
false pending independent review. The probe's successful status is
`SINGLE_SITE_C1_EFFECTS_EVIDENCE_READY_FOR_REVIEW_NOT_C2`. A local residual effect
does not require the final output to differ; the separate full-output comparison
is recorded. E5 raw-statistics acceptance remains independent.

Prepare a root-reviewed case from an immutable installed source and Native bundle:

```sh
python3 -B experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py \
  --installed /reviewed/installed --native /reviewed/native \
  --probe /reviewed/providers26_group_probe.py --source-commit FULL_REVIEWED_COMMIT \
  --mechanisms 26.residual_stomp --stages c1_single_effects --card 2 \
  --out /reviewed/qmm-single-site-c1-cases.json
```

This command only generates the case; root controls actual scheduling. Complete
observations, stdout/stderr, arrays and allocator details remain under server 124
`/evidence`; public Git contains source, contracts and summaries only. This mode
does not change the group all-eligible gates or claim 17/17, 9.x/A3, or private
downstream integration acceptance.

CPU validation uses real Torch module execution and hooks with explicit Native,
allocator, scheduler and metadata doubles. The focused suite passed 87 tests,
including complete 135-call rows, alias preservation, trace capacity, refusal of
changed current objects/formals/allocator pairs, failed B/L/OFF and effect receipts,
foreign completion owners, pending-storage rebinding, before/after restore and ack
failures, final hook ordering, and the unchanged candidate/group negative cases:

```sh
python -B -m unittest test_writer26_single_site test_writer26_group_experiment \
  test_writer26_pending_consumer test_writer26_coverage test_writer26_public_return \
  test_writer26_inventory test_writer26_loader test_writer26_group_build
```

Use the repository's existing CPU environment and put `src`, `fault39/src`, the
repository root, `tests`, `tests/worker_extension`, and `fault39/tests` on
`PYTHONPATH`; set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. These tests never establish a
real Native hit or C1/C2/C3 qualification.
