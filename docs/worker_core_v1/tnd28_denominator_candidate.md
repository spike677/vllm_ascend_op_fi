# TND FIA denominator replay: kernel and host candidate

This increment implements the missing **TND kernel arithmetic and source-bound
host selection** for `28.fia.denominator_replay`. It is not a working main-project
fault entry yet. No successful SDK build, Native input binding, NPU effect, C1/C2/C3 or E5
qualification is claimed by this commit. FA, legal tiling and group execution are
outside this increment.

Source87 independently resolved 56 TND API scopes and 112 actual launches. That
observation is not a normal certificate. Source110 compiled the separate TND
normal candidate. Source113's journal retained zero Native calls; later symbol
visibility did not identify the actual cached FIA route. None supplies this
new variant's current-input authority.

## Exact change

The new variant is `tnd_prefill_denominator_replay`, vendor
`dcat_f28_tnd_replay_transformer`. It starts from pristine c77 source and verifies
16 source hashes, including the real `attention/` parser, TND GQA cube/vector,
dispatcher and original `SplitCore`. No historical object/kernel/vendor is read.

The accepted geometry is the source71/source87 actual TND prefill: one fresh
2048-token request, FP16 Q `[2048,16,128]`, K/V `[2048,8,128]`, G=2, int8 causal
mask, cumulative Q/KV lengths `[2048]`, sparse3, no paging/quant/rope/prefix.
The Python API block size 128 and parsed nonpaged block size 0 remain distinct.
The supplied pre/next attributes remain separate from the pinned parser's
effective `(INT_MAX,0)` pair.

| Stage | Host/kernel behavior |
| --- | --- |
| Native normal | Existing scoped M512/S2=512 NonQuant route, key `103000000030000003`; no replay |
| New control | M128/S2=128, TND causal tail, BRC, key `109000000030400003` |
| New fault | Same control template, key `119000000030400003`; multiply **old softmax sum only** by 2 before the second block's normal max rescale |
| OFF/unrelated calls | No replay selection; original source routing/arithmetic |

The keys are checked against the original `GenTilingKey` result
`103000000030400003` (FP16/TND/TND/BRC, no PA/FD/tree) before extending its trait.
That raw BRC specialization is also explicitly compiled and registered. The six
unchanged fallback entries from the TND normal build remain in the exact key set;
they are build coverage, not claims about this new package's observed launches.

`FiaBlockVecNonQuant::SoftmaxFlashV2Compute` repeats `inSumTensor` only when
`!info.isFirstSInnerLoop && info.s2Idx == 1`. BRC length is
`dealRowCount * brcbNum`. Maximum, numerator/PV, QK, cast and output stores are
unchanged. The oracle is `numerator / (denominator + first_block_mass)`, with
the mass expressed under the final maximum. This is not constant output scaling.
Queries 0..127 have no second causal block and therefore no replay.

The causal helper uses TND `(S1,G)` coordinates. At flattened row 2048, M128
covers queries 1024..1087, not 0..127. Selection precedes the original split and
workspace calculations. After the original `SplitCore`, every whole query tile
must belong to one core, all 256 `(KV head, M tile)` reductions must be covered,
and every FD producer/consumer array must be zero. A partial S2 assignment is
refused; no FD count or array is cleared to make it pass.

The first device-free build (`tnd28-denominator-build-a1`) compiled the CPU
contracts and executed the original splitter. All 11 physical scheduling
profiles 16..26 produced real split-KV/FD work, so the build stopped before SDK
compilation. Its matrix SHA is
`ab36cf4824ca295a43aca16bd7b2dc00119a782d624730cfdae1e896f728d86c`;
that refusal remains an original result, not a validator error.

The revised control and replay both pass **8 scheduling cores** to the original
splitter for the eight equal KV-head workloads. `aicNum_`, `aivNum_`, platform
identity and physical-sized `SplitResult` storage remain unchanged. The original
`AssignByBatch` can allocate one whole head to each scheduling core. Admission
requires its actual result to contain exactly eight endpoints `(1..8,0,0)`,
256 complete query-tile reductions and zero FD producer/consumer entries;
anything else is refused. No endpoint is authored by the adapter. Original
`SetSplitOutput`, `FillTiling`, `CalcNumBlocks(usedCoreNum_)` and workspace
calculation consume this result normally. This is a changed scheduling policy
for this candidate, and still requires a fresh stock/control normal bridge.

The host requires the current observer token, PID/native TID/start identity,
owner/generation, actual module object, request hash and input hash. A UUID is
consumed once even if subsequent tiling fails. Foreign thread/unscoped calls do
not acquire a replay. Replay additionally requires
`dcat_tnd28_denominator_current_input_v1` from a reviewed **live** input adapter.
This increment deliberately does not provide a dummy implementation. Missing
binding fails before M/key selection. In particular, the old empty journal,
saved normal report, copied bytes, same shape or an RPC boolean cannot pass it.

## Root-only build preparation

The root's existing device-free controller needs only one explicit choice:

```text
tnd28-denominator-replay -> apply_tnd28_denominator.py
```

No controller, registry, Host, Runtime, exporter or public action file is changed
here. Preserve the controller's pinned image, `runc`, no network/device mounts,
same reviewed offline dependencies, four CPU jobs and fresh output directory.
After that choice is integrated, the root command is:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/build_tnd33_124.py" \
  --source "$SOURCE" --interfaces "$REVIEWED_CANN851_INTERFACES" \
  --strategy tnd28-denominator-replay \
  --out /home/17-fault-c2c3-20261117/tnd28-denominator-build-a2
```

The controller runs the source recipe and the existing `build_fresh.py`. Before
SDK configuration, the builder compiles the SDK-independent owner/replay
contracts, a real missing-weak-symbol negative executable, and executes the
**unchanged** pinned `SplitCore` twice for all 11 AIC profiles 16..26: the original
physical schedule and the selected eight-core schedule (22 complete rows).
It preserves the original schedule's full arrays and refusals separately.
Every selected profile must pass the strict whole-head check before SDK
compilation. For a matrix-only early check in the same device-free
image, against the source checkout produced by the recipe:

```sh
python3 /driver/compile_tnd28_denominator_matrix.py \
  --source /input-source --out /build/matrix-only
```

Expected fresh outputs are `build/run/evidence/completion.json`,
`evidence/tnd28_replay_matrix/{matrix.jsonl,receipt.json}`,
`evidence/tnd28_schema/{fia_abi_schema.json,compile_receipt.json,macro_dependency.json}`,
and the new vendor under `build/run/private_stage`. The builder remeasures the
actual 65-field SDK serializer (63 semantic fields plus its two pinned PH padding
fields), requires the exact compiled key set and hashes every new artifact.
Build success remains `CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION`.

The source export must include all new `tnd28_denominator_*` files,
`apply_tnd28_denominator.py`, `compile_tnd28_denominator_matrix.py`,
`tnd28_causal_tile.h`, `test_tnd28_denominator_contract.cpp`, plus the existing
normal headers/recipe, `base_policy.py`, fresh builder/schema/compatibility and
registration closure. `build_fresh.py` now imports `tnd28_denominator_build.py`
for every variant, so omitting that helper would break even an old variant's
CLI import. The two shared builder/registration diffs only add the exact new
variant; old normal behavior and defaults remain unchanged.

## Remaining pass/stop conditions

1. Pass the actual CPU matrix and SDK host/kernel build. A template compile
   failure, no complete reduction profile, or wrong compiled key stops here.
2. Resolve the actual cached typed API route and public-to-prepared input
   lineage in one bounded real-target normal window. Source87 V has stride
   `[4096,128,1]`, offset 3072; a contiguous same-shape V is not its proof.
   Auxiliary Slice launches and current register/binder history cannot be
   discarded. The live gate implementation and main-project source connection
   are still missing and must remain explicit work, not “only acceptance”.
3. Establish this new package's stock/private native-normal bridge and its new
   M128 causal control on the same actual inputs. No source110 or old key/result
   is inherited. Keep complete current Q/K/V/mask/optional/scalar metadata,
   host/POD/scoped launch/registered ELF binding and normal outputs. An exact
   bridge mismatch remains a refusal, not blanket technical inapplicability.
4. Only a newly admitted actual target may execute one replay and OFF. Record
   the real denominator selection, immediate attention difference, downstream
   propagation and same-Worker recovery separately. Unchanged generated text
   does not erase an internal numerical effect. No C2/C3/E5 claim follows.

Local validation after the scheduling correction: 69 CPU Python contracts passed (new replay plus old TND
normal/build and shared builder/registration contracts). All 16 retained actual source hashes matched; the patch
applied to the seven actual source files, and removing the new denominator and
causal insertions restored every other vector/kernel byte exactly. Raw cached
sources and model/runtime artifacts are not committed. Root's a2 device-free
build has now completed: completion SHA
`93f5159839db7899c4725c182790566bd9364437f4cb734135f7ef51c49aa3b3`,
registration SHA `ca068750afcc3484fea16bf62a012cb2a9484b1149e2a5512833be746e82f637`.
Its actual 22-row SplitCore matrix and SDK host/kernel build passed; the ten
compiled keys include control `109000000030400003` and replay
`119000000030400003`. The kernel SHA is
`d99ce5e9435a3df59ff4432f53f997305638de40e4b306918f587faf8d96dd81`.
The measured public serializer has 63 semantic fields and two actual PH padding
fields. No NPU call, new normal bridge or F permission follows. The first
build's original-schedule refusal is retained, not converted into a pass.

The subsequently added [compiled-input normal observation](tnd28_prepared_input.md)
is not present in a2. It requires a fresh host/opapi build, new completion and
artifact identities, and a real normal observation. Its new C++ observer has
not been compiled on this Windows host; the next device-free builder compiles
and executes the dedicated fake-SDK contract before SDK compilation. Actual
replay input admission, safe rejection lifetime and main-project binding remain
unfinished implementation work.
