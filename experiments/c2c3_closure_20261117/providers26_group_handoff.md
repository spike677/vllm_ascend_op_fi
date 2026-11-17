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

## Actual SDK result and runnable candidate entrance

Root built the new ABI successfully from source-a29 on 124. The frozen bundle is
`/home/17-fault-c2c3-20261117/native26-group-v2-a1`, with `build.json` SHA256
`5a6adbf86bd0fdbb5a18ab16953714db6a292ff315bdc0dce28c1a59654b7653`.
Its SDK/control and memory contracts passed; actual NPU calls remain zero. The
build now inspects the local image and runs its immutable image ID. A unique
container name, owner label and actual image are verified before any cleanup;
timeouts/exceptions retain the build receipt and log.

The Python v2 binding and standard Worker entry now expose three separate
candidate stages. `inventory` makes two complete healthy observations.
`lossless` then executes reference, actual temporary output lossless and OFF for
each completely resolved candidate. `effects` additionally executes F/OFF for
each of those independently qualified candidates. Every fresh request still
records the full independent native denominator, including all unselected,
unsupported and unverified calls. A budget that cannot contain the resolved set
blocks that whole plan; it never chooses a first target. Discovery binds the
actual module callable/source/code, native registration, complete geometry,
retained public residual, actual public writer-output storage and next real
consumer. Full allocator records distinguish a proven nonadjacent pair from an
unknown allocation. Any live-pointer uncertainty refuses dependent writes.

These one-current-target candidate windows deliberately keep C2/C3 false. They
are an entrance for actual qualification data, not an all-eligible certificate.
After Native stop/drain/release, another full-vocabulary healthy request runs in
the same Worker, and then all owned hooks and references are removed. If a
failure leaves a lease, retry OFF is permitted only while the actual public
writer output and residual objects are still strongly retained at the exact
addresses. Missing pointer liveness leaves the lease/backup pending and requires
quarantine; it never fabricates restoration through a fresh process.

Root can prepare two read-only cases first, using newly installed Python source:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py" \
  --installed "$INSTALLED" \
  --native /home/17-fault-c2c3-20261117/native26-group-v2-a1 \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_group_probe.py" \
  --source-commit "$COMMIT" --card 2 --stages inventory \
  --out /home/17-fault-c2c3-20261117/providers26-v2-inventory-a1-cases.json
```

The same command with `--stages lossless effects --statistics` prepares four
separate bounded candidate cases after review of the complete natural census.
The default target budget is eight and can be explicitly set with
`--max-targets`. Native event capacity is checked against the entire planned
healthy call count before qualification starts. The generator pins the exact
image ID from the Native build and preloads only the new writer DSO plus the
verified trace/proxies. Model mounts are DeepSeek W8A8 for QMM and SmolLM2-135m
for FIA. Every source/model/library mount is read-only. Root alone owns the
resource queue and any actual NPU/container start.

Seven new Python CPU tests cover both writers through two actual CPU modules,
all candidate phase transitions and zero-resource health, missing Native
callbacks, missing live writer outputs, unsupported versus unverified residuals,
and refusal to shrink the target set to fit the budget. Their native and
allocator fixtures are explicitly simulated. Remaining dependent work is the
valid natural inventory, source/lifetime review of unresolved edges, shared
CoverageSession all-eligible admission, and a live same-Worker C2 proof before
every-match. The candidate result reports every one of these remaining gates.

## Conditional complete-set C2/C3 execution

The additional `coverage` stage uses the shared `CoverageSession` and the same
unchanged Native v2 bundle. It first performs two complete healthy Native/public
censuses, including every missing callback, unsupported row and unresolved
writer/consumer lifetime. The complete scope must be stable and contain no
UNVERIFIED row. A budget overflow or overlap between eligible writer-to-consumer
intervals blocks the whole group. This implementation supports serial writers
within one fixed fresh-prefill request; it does not claim decode or multistream
coverage.

Three further complete healthy requests perform reference, temporary-output
lossless and OFF for **every** eligible target. Each actual boundary has its own
Native lease, writer/consumer receipt and complete output/residual readback;
packing these into three requests does not reuse one target's qualification for
another. Only then does `all_eligible + once` admit the shared fault request.
The independent native event precedes admission. An existing eligible target's
failed geometry remains an eligible failed opportunity. An unknown or newly
eligible lifetime before the first actual effect, or during OFF, fails the
scope. Post-effect additions are recorded separately and pass unchanged without
expanding the frozen target set. A missing eligible callback also fails rather
than disappearing from the denominator.

Recovery stops all logical targets, drains all of them before any release,
requires actual OFF readback for every target, and releases the Native runtime.
A subsequent full-vocabulary healthy request runs in the same Worker. The
original issued C2 proof is retained only after all owned hooks and public tensor
references are removed and the common zero-resource health proof passes.
`every_match` uses that same object, owner, process/model/module lifetimes and
scope with a higher generation; copied or relabelled proofs are rejected before
Native fault sealing. C3 observes every existing-target call in two bounded fault
requests. Unknown pointers and unresolved recovery retain the failed status and
logical reservations; restarting the container is never an OFF receipt.

Root can prepare two conditional full-set cases after publishing this Python
increment. This only writes the queue manifest; it does not launch a container:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py" \
  --installed "$INSTALLED" \
  --native /home/17-fault-c2c3-20261117/native26-group-v2-a1 \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_group_probe.py" \
  --source-commit "$COMMIT" --card 2 --stages coverage --max-targets 128 --statistics \
  --out /home/17-fault-c2c3-20261117/providers26-v2-coverage-a1-cases.json
```

Each case uses the immutable image ID recorded in the matched a29 Native build,
the fixed model paths above, TP1, fixed 64 KV blocks, 16 CPU/80 GiB/1800-second
queue budgets, 32,768 independent Native events, 64 MiB Native device and
512 MiB Native host budgets. A successful C2 session needs eight complete
requests and C3 nine. The shared native event capacity is checked before the
dependent plan, including events already used by C2. Results and raw statistics
remain under the new case's `c2/` and `c3/` evidence directories on 124.
The phase map records actual collector steps; compare raw reference-all against
fault for E5 with fixed rtol `1e-5`, atol `1e-8`, preserving null/NaN/alignment.
Restoration memcmp and runtime-selected targets are not an E5 first-difference
decision.

Nine group CPU contracts now exercise once followed by the original live C2
object's every-match, all-stop/all-drain/release ordering, both typed writer
identities, full-set unresolved/over-budget/overlap rejection, unknown-before-
effect and frozen-target geometry failures, and copied/relabelled proof
rejection with Native/hook cleanup. These use explicit mock Native memory and
cannot grant public C2/C3. The valid natural inventory and actual C2/C3 runs
remain unverified after the externally stopped attempts; no natural target
count is inferred here.
