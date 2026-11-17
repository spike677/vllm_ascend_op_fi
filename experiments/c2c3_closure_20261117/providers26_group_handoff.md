# Native26 group v2: offline implementation, natural qualification pending

The new `libdcat26_group_v2.so` implements serial transactions for the existing
QMM output-underflow and FIA wrong-destination mechanisms. It is a distinct ABI
and build fingerprint. The old v1 writer sources, bundles and C1 evidence remain
unchanged. The v2 bundle omits both old writer DSOs and refuses to install when
either v1 control ABI is already loaded. One group chooses one mechanism.

Every admission binds owner, generation, actual native thread, target SHA words,
one-use token, exact observed call, output/residual geometry, current producer,
and QMM cache registration. The independent trace records the call first. The
same trace supplies the actual next AddRms consumer for both mechanisms; the
shim does not invent producer or consumer sequence numbers. FIA executor
registration exists only from that actual GWS to Execute and is added to the
execution receipt, never treated as a persistent cache epoch. Invalid identity,
foreign thread, overlap and missing trace detail remain failures in the complete
denominator. The callback may explicitly pass an unresolved call unchanged.

QMM lossless now allocates a real temporary output. Both writers require
reference, lossless and healthy OFF for every target before sealing the exact
target set. QMM fault still writes at `output - (allocator_padding + 32)`; FIA
fault still writes at the beginning of the retained residual. QMM copyback
remains host staged because shifted source and destination overlap. Per-call
budgets are spent before execution. C2/C3 dispatch policy belongs to the shared
CoverageSession; the Native engine limits each target to eight fault attempts.

Each transaction keeps complete original residual/output/padding backups until
the actual consumer and full readback restoration succeed. Failed writer,
consumer, copy, sync, descriptor destruction or device free retains the lease
and every still-owned resource. OFF is explicit stop, drain, restore, exact
readback, acknowledge and remove; failures can be retried without discarding
backups. Full memcmp proves restoration and lossless behavior only. It is not
an E5 statistic or a first-difference verdict. A same-Worker healthy output
comparison after hooks/resources reach zero is still required.

CPU memory contracts passed on the host, without a container or NPU, at
`/home/17-fault-c2c3-20261117/providers26-group-cpu-a2/receipt.json`. These tests
exercise two targets for each mechanism, exact recovery and retry, failures,
duplicate/stale tokens, consumer sequence/stream mismatch and overlap. They
are explicitly host memory simulation, not Native qualification. A separate
control test compiles against the actual SDK and calls no CANN operation.

Root-only SDK build command after publishing this committed source as `$SOURCE`
and setting its actual commit `$COMMIT`:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_build_group.py" \
  --source "$SOURCE" --source-commit "$COMMIT" \
  --parent /home/17-fault-c2c3-20261117/native26-qmm-inventory-a2 \
  --parent-manifest-sha256 0bf039524de7c783b941aeda09988d04a50a41203265af665bf369c545dc7c1d \
  --out /home/17-fault-c2c3-20261117/native26-group-v2-a1
```

The builder verifies every parent file, copies only the trace/proxy dependency
set, compiles the new DSO against actual CANN 8.5.1 in the pinned vLLM image,
runs CPU/control contracts, records headers/source/image/command and output
hashes, and checks new ABI exports. It has no device mounts and explicitly uses
the runc runtime with `ASCEND_VISIBLE_DEVICES=void`. It preserves a failed build
log and requires a new output path for retries. No NPU case is enabled by this
build command.

The valid natural inventory is still missing after the externally stopped
attempts. No new target is qualified and no C2/C3 pass follows from this code.
Actual Python model observers must retain the real producer, writer output and
consumer inputs, verify exact public/native ownership and live allocator
adjacency for every QMM lease, bind the scope to one fresh prefill or one proven
decode boundary, and keep every unsupported/unverified row. Target scope cannot
be inferred from a structural module list or from pointers/shape alone. The
serial implementation cannot qualify a workload that naturally overlaps writers
before the first consumer; it must report that unresolved complete scope.
