# FIA NCL descriptor bridge: additive Native stage

This change adds an FIA-only typed descriptor contract. It does not qualify a
natural target or enable a fault. The existing QMM ND path, v1 inventory wire
layout, fault destination and Engine recovery/readback rules remain intact.
The actual source49 SmolLM2 observation had public ND format 2 and native NCL
format 47; format 47 is preserved throughout this bridge. It is not a TND tag.
The actual FIA `input_layout` string is separately recorded and compared.

The trace observes original `attentionOut` before GWS, original and effective
outputs after the real GWS, then the group's owned effective descriptor before
execute. Every typed record binds call, native thread/device, event sequence,
descriptor and raw address, dtype, format, complete view/storage ranks and
dimensions, strides, offset, element size and full byte spans. The public v1
denominator is recorded first and is not filtered by typed eligibility. Unknown
geometry and failed or incomplete observations remain refusals.

The specialized admission accepts only complete BF16 NCL rank-three geometry,
equal view/storage dimensions, strict contiguous strides, zero offset and the
existing bounded span. The original descriptor is read only inside its real
GWS stack. Every selected typed mode, including Reference and OFF, owns its
effective descriptor with actual tag 47; execute never re-reads the borrowed
original. Lossless and fault use the unchanged Engine replacement address.
Prepared executors that did not return successful execution retain descriptors,
backups and owner. An exception or rejected current snapshot cannot authorize
drain/release. SDK getter arrays use the existing scoped array owner on success,
early return and exception.

Additive layouts, checked against the actual SDK by the builder:

| Wire type | Version | Bytes |
| --- | ---: | ---: |
| FIA typed descriptor | 2 | 280 |
| FIA typed observation | 2 | 656 |
| FIA typed observer state | 2 | 104 |
| FIA typed admission | 3 | 912 |
| FIA typed call snapshots | 3 | 2048 |

The new trace exports are `dcat26_inventory_fia_{typed_state,typed_read,before,
after,owned_execute,end}_v2`. Group exports are
`dcat26_group_fia_abi_v3`, `dcat26_group_fia_snapshots_v3`, and
`dcat26_group_continue_fia_v3`. Existing v2 control layouts are unchanged.

Local validation: 43 focused Python contracts passed. The new C++ test includes
the actual group runtime and supplies host-memory SDK doubles. Its 16 cases
cover the selected Reference lifecycle, changed original/effective geometry or
identity, wrong call, v2/QMM rejection, getter failure after allocation and
exceptions. It asserts balanced array ownership and zero release after a
prepared-but-unexecuted failure. Root executed the earlier 15-case source57
parent/group-a2 SDK build successfully, with group manifest SHA256
`6ef225470c6a4701fb5edfe6429cfeadbb00cb1f2dbb5960e8858c611716cb50`
and parent manifest SHA256
`413cb5df91d742e8021a207df7a108f84a7102f5f902982a24de9a3228627f91`.
The additional unclassified-executor gate and 16th case require a fresh build;
the earlier a2 bundle is not authorized for the typed device case. This
worktree has no C++ compiler. Root must run the fresh SDK build below before
using the final ABI implementation.

## Root-only device-free build

Use a newly frozen `$SOURCE` containing this commit, its actual `$COMMIT`, and
fresh output directories. The old metadata parent cannot be reused: both trace
and proxy changed. These commands run CPU build containers with no NPU devices;
the agent preparing this change has not run them.

```sh
R=/home/17-fault-c2c3-20261117
python3 "$SOURCE/experiments/c2c3_closure_20261117/build_native26_parent_124.py" \
  --source "$SOURCE" --out "$R/build26-fia-typed-parent-a3"
python3 "$SOURCE/experiments/engineering_20261117/stage_native_bundle.py" \
  --source "$SOURCE" --build "$R/build26-fia-typed-parent-a3" \
  --kind 26 --out "$R/native26-fia-typed-parent-a3"
PARENT_SHA=$(sha256sum "$R/native26-fia-typed-parent-a3/build.json" | cut -d ' ' -f 1)
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_build_group.py" \
  --source "$SOURCE" --source-commit "$COMMIT" \
  --parent "$R/native26-fia-typed-parent-a3" --parent-manifest-sha256 "$PARENT_SHA" \
  --out "$R/native26-fia-typed-group-a3"
```

The first builder compiles the whole existing trace recipe, including FIA,
despite its QMM recipe selector. The second builder requires all typed trace
exports and the exact new control sizes, executes the 16 host runtime cases,
and binds source, header, image and DSO hashes in the new profile. It retains
failed-build receipts and cleans only its verified owned container.

## Remaining Python and natural gates

Python must bind the current typed snapshots and actual public Tensor lifetime;
it must not relabel a generic format allowlist as equivalence. Two unmodified
stock natural requests must retain full actual public output bytes. The owned
descriptor Reference must match those bytes completely before lossless/OFF,
sealing or F can be qualified. This is new bridge qualification; old v2
Reference evidence is not reusable. Each current call still needs the real
writer, still-live residual, actual consumer and same-Worker healthy OFF.

The Python follow-up implements those gates in `writer26_fia_typed.py` and the
existing experiment/provider. It does not add a controller or a new state
machine. A fresh typed profile binds exact ctypes/SDK sizes. Independent typed
records must account for every v1 FIA call, including unresolved candidates;
each selected call needs its additional owned-before-execute record. At the
public return and again before consumer acknowledgement, the current owner,
token, records and descriptor-release receipt must agree. A snapshot failure
cannot be repaired using an old record. A failed descriptor check at an
existing site remains that site's failed eligible call.

Actual public output bytes are retained as immutable host samples and separate
bounded binary artifacts. Equality compares the full bytes; hashes only name
the evidence. Both healthy samples and the new owned Reference must agree.
Stock mismatch refuses dependent modes and never falls back to another target.
The existing public Tensor/storage ownership is retained through the actual
consumer and recovery. The stock samples are cleared only after Native cleanup;
their host bytes also enter the Coverage health gate. Diagnostic artifacts are
not recovery authority. A missing public lifetime still refuses restoration.

Eighteen additional CPU Torch/Native-v3 mock tests cover the complete candidate
loop, complete-group once followed by the original live CPU proof, copied proof
refusal, changed before/after descriptors, omitted observations, final-byte stock
changes, consumer-time snapshot changes, and failed-lease cleanup. These are
fixture contracts, not CANN numerical or natural inference acceptance. Existing
QMM-only `c1_single_effects` remains QMM-only. All full candidate reasons and
independent call records remain available; unknown paths are not discarded.
The complete writer26 Python regression suite passed 108 tests, including the
existing QMM and QMM-only single-site Core path.

After root verifies the fresh SDK bundle and builds matching Python packages,
prepare one read-only natural FIA inventory case first:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py" \
  --installed "$INSTALLED" --native "$R/native26-fia-typed-group-a3" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_group_probe.py" \
  --source-commit "$COMMIT" --mechanisms 26.fia_destination --stages inventory \
  --max-targets 64 --card 2 --out "$R/providers26-fia-typed-inventory-a1-cases.json"
```

The case preserves the existing SmolLM2 path, TP1 eager BF16, fixed 64 KV blocks,
one fresh prefill per request and the root queue's resource/device gate. Later
`lossless`, `effects` or `coverage` cases each start their own complete two-stock
inventory and run the same strict gates. `max-targets` bounds the entire resolved
plan; it does not choose a subset. Only root may approve/run those device cases.
No C1, C2, C3, numerical Native acceptance or E5 conclusion follows from this
code or its CPU tests.
