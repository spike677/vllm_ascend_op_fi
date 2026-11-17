# Current execution checkpoint

## Active checkpoint — source-a30 and live-proof review

- Source-a30 is frozen at5fef59cea4eb408c3bab5d22576c17fbc61f976e,596 files,
  archive8f42df70245120876c2513302467eb4df9a8c134352fa8db1c8e83d30c100e48.
  Remote extraction and every file hash passed. Subsequent review/documentation
  commits do not change that immutable snapshot.
- `fa28-launch-short-a1` passed the fresh card5 gate, but Docker exited125 before
  creating a process: a read-only `/build` parent prevented its `/build/run`
  nested mount. No NPU call or timeout occurred. Its original controller/logs
  remain under R; only its unstarted owned container was removed. Correct the
  manifest layout, then use a fresh attempt with the same90-second budget.
  Long Worker experiments still await external-cleanup coordination.
- C3 proof hardening9499d57+13955ef is integrated. The independent frozen-route
  auditc936aa3 (root70402592) verifies exact18/20/39 driver bytes and their original
  live-object handoffs. It does not retroactively run the new registry in old
  processes. Root's59 focused coverage/authority/identity contracts passed.
- The public vision weight download is still running in local session28838.
  Preserve its first receipt, then independently reverify all pinned bytes with
  the current helper before transfer. The original running helper predates edits:
  its completion-time script hash alone is not a startup-code attestation.
- Complete-set Native26 coverage/probe integration is pending final Agent5 review;
  its v2 SDK artifact remains the already verified CPU build. No new Native26
  NPU or whole-model qualification has been granted. C2/C3 stay3/17 each.

## Active checkpoint — further source review and device-free preparation

- Published checkpoint remains8f73f82 on `codex/17-fault-c2c3-closure-20261117`.
  Additional reviewed source is being integrated locally; C2/C3 stay3/17 each.
  No root NPU task is running. The external cleanup coordination question below
  is still pending; no expired host isolation or unrelated process kill is used.
- FA source-a28 CPU build completed0 at `R/fa28-fresh-build-a1`. Root independently
  verified all851 vendor artifact hashes. Completion SHA7189c077dad9411e53240d6a0cc9190477d00433d5b1325833c0fcb26c7cac5d.
  See `attention28_fa_cpu_build_summary.json`; no actual NPU launch is claimed.
- Natural MoE group adapter was integrated, then root review found nonselected
  calls still used equal copies of I/P. Fix d51eb34 (root61ac184f) now sends
  original objects/storage to the native operator for healthy/unselected/exhausted
  calls, and releases the independent observer even if trace close fails.
  `natural-moe8-group-a1-cases.json` still names source-a28 BEFORE this fix: do not
  run it as the corrected group. Prepare a new source/wheel and attempt instead.
- Full Qwen3-VL MODEL_LOCAL image census/group and original Conv3d padding adapter
  are integrated. Pinned public model89644892e4d85e24eaac8bacfd4f463576704203 is
  being prepared; official weight SHA7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0.
  No NPU/Worker qualification. Text-only Worker request context remains insufficient
  for actual multimodal scheduler/encoder ownership.
- Native26 group v2 is separate from legacy writer DSOs and C1 scope. Its CPU
  failure/retry contracts pass; root SDK build and Python/natural ownership binding
  are still being prepared. Old inventoried/unresolved rows cannot be dropped.
- FIA natural prefill observation is read-only, with no private vendor or policy
  changes. Actual TND/page16 and8.5.1 do not qualify the9.1 BNSD strategies.

The older checkpoint notes below retain exact earlier attempt identities.

### Reviewed source-a29 and follow-up preparation

- source-a29 commit1076720aca23ad55decaf2fe42e174176c0e0a5f, archive
  cacd68443029eff909f8bc02dfd002955dd5d3197d89e28f8ccfe389d076d25b,
  all578 files independently verified. packages-a11 build/install completed0.
- Native26 v2 SDK build `R/native26-group-v2-a1` passed without devices. Deployment
  SHA5a6adbf86bd0fdbb5a18ab16953714db6a292ff315bdc0dce28c1a59654b7653;
  new DSOe7c21c25eb228ee126b17c6ec76717861ba6942d043bd18302466fd5cd32174a.
  Independent audit `AGENT7_CPU_BUILDS_A29_REVIEW.md` preserves host GLIBCXX failure
  and matched-image CPU control success. Candidate B/L/F/OFF code is integrated;
  complete public C2/C3 dispatch is still being completed by Agent5 with Agent1.
- New `natural-moe8-group-a2-cases.json` uses source-a29 with the non-target
  identity fix; it is prepared, not executed. Vision17 census/group queue files
  also use source-a29; they require complete pinned weights and actual census
  review before dependent group work. All are single-card5 manifests.
- Device-free vision processor attempts a1/a2 both retain import failures:
  absent libascend_hal, first Torch autoload then Transformers explicit NPU import.
  No model or NPU execution occurred. Configuration files are verified; local
  official weight download is still in progress (outside the Git tree).
- Root source-a29 unittest:729 cases,2 skipped,no failures; separate Worker suite
  194 cases,no failures. Do not add these overlapping counts. New FA/FIA focused
  suite:48 cases,1 skipped,no failures; RoPE focused contracts also pass.
- The new FA final-argument reader is integrated. Long Worker runs still await
  external cleanup coordination. Previous four short FIA cases completed in
  14-23seconds, so root is reviewing a separately bounded90-second four-call
  standalone FA case; no implication that the long cleanup window was granted.
- Agent1 found copied/relabelled C2Proof objects can retain the private issuer.
  Issued-object identity plus immutable-field verification is being added before
  publishing the next checkpoint; no old scope or CPU proof is being promoted.

## Active checkpoint — Native39 accepted; resource coordination pending

This section supersedes the older running-session notes below. R is still
`/home/17-fault-c2c3-20261117`; no old evidence was overwritten.

- Native39 `providers39-a6` exited0. Fresh live C2 has113 targets/113 hits;
  subsequent C3 has113 targets/226 hits. Independent Agent7 review found342/568
  reference bytes matching and same-Worker recovery; root accepted this bounded
  C3 after checking113 actual output changes in each fault phase. Totals are
  **C2 3/17, C3 3/17**. C3 has no raw dump triplet, so C2 statistics cannot
  be presented as C3 E5. C2 first linear output5.1875->Inf, later NaNs retained.
- Natural DeepSeek MoE census `natural-moe8-a1` completed with two exact healthy
  replays:52 actual unpermute calls,26 per request, all five source-bound roles
  observed, no trace errors. Full permutation values and fault effects still
  require qualification; no C2/C3 claim follows from the census.
- Source-a24 SHA `e4c7ecde9dee180475b83401f32dc8f248d733ca014fff27a3e5a499cdd48367`
  and packages-a7 contain fresh Worker18 statistics and stock/private33 loading.
  Source-a25 SHA `5b2b8106e6599dc2c0c6b3caa8aae2ba4b20bbaf1e18225733e618f7cbaeb543`
  and packages-a8 additionally contain the whole Native quant inventory.
  Source-a26 SHA `33d4d0a2cf83d8aa58431ebad259005ec919ec26f5ec5ff77e7d54c2d7ee8c19`
  adds only a no-device stock extension diagnostic; no new wheels required yet.
- Both optimized26 inventory SDK builds passed (a2). The first QMM run under
  `reviewed-next8-a1` was externally stopped: Docker signal15 then9, exit137,
  OOMKilled=false, no controller timeout. Preserve `external_stop_events.jsonl`.
  No mechanism or same-process recovery conclusion is possible from that run.
- `reviewed-next8-a2` completed four28 FP64 B/F cases, then41 B was again stopped
  outside the controller after about92seconds (signal15 then9, noOOM, exit137).
  Both stop-event sidecars are preserved. There is no running root NPU task.
  A user coordination question is pending: pause external cleanup for30minutes;
  no expired host isolation or foreign-process kill has been performed.
  Fresh28 final-argument/launch review passes8/8. Independent raw replay first
  differs at FIA output Mean; separate Norm change1.7210312676%, six nulls retained.
  Legal has two exact statistic changes, neither exceeds fixed tolerance. Both
  remain standalone step0, not model C2/C3. Reviewed reports are now integrated.
- Source-a24 CPU-only quant Native build passed at `build-quant-inventory-a1`;
  `providers-quant-inventory-a1-cases.json` is ready with source-a25/packages-a8.
  `worker18-statistics-a1-cases.json` is ready with source-a24. Neither has run.
  `postwrite39-c2-a1-cases.json` also remains queued, not executed.
- Correction to the older33 note: private **libcust_opapi.so** shadowed the stock
  custom library's Norm entry, not libopapi.so. Actual absolute-path bootstrap
  and official platform vendor registration now verify Norm->stock/FIA->private.
  No-device full extension import aborts identically in both private and stock-only
  controls (`attention33-loading-cpu-a2`, `attention33-loading-stock-control-a1`).
  This does not establish NPU failure or full import success. New source-a27
  SHA `63558f149d0c6b54b46c8a0b91eea2c7a2057aab526c82ae9e1d10b868593340`
  and packages-a9 explicitly distinguish these scopes and require a real binding
  recheck after ordinary device-backed Worker initialization. CPU binding+platform
  check passed in `attention33-loading-cpu-a3` (no complete extension/NPU claim).
  `attention33-baseline-a2-cases.json` is ready, not run. Exact stock bridge beforeF.
- Integration a5db7141: main unittest discovery692 passed/2 skipped; separate
  Worker Extension discovery191 passed. Do not add these overlapping counts.

## Active checkpoint — integration f572c9f8 plus root review

This section supersedes the older source-a15 checkpoint below. Root resumed the
same task and preserved every earlier attempt. No new raw data is published.

- Independent review `AGENT7_ACTUAL_REVIEW.md` and its hash manifest are merged.
  Root accepts bounded standard-Worker C2 for `18.api_half` and
  `39.matmul_range`, and bounded standard-Worker C3 for `18.api_half`.
  With the additional Native20 review below these are **3/17 C2 and 2/17 C3**
  mechanism IDs in their exact reviewed scope,
  not whole-model-family, all-version or multi-Worker qualification. HF Fault18
  is a separately reviewed scope of the same mechanism, not an additional count.
- Standard Worker18: actual C2 113/113, C3 226/226, 9 scheduler requests,
  exact complete hidden/logits OFF, same generated healthy token, owned hooks0.
  Its standard-Worker E5 still needs fresh capture; HF statistics do not qualify it.
- Native39-a5: C2 113/113, target OFF/whole logits exact. C3 discovery refused
  persistent collector binding; strict source/code/globals-bound unwrapping is
  integrated. Fresh `providers39-a6-cases.json` uses source-a21/packages-a5 and
  must establish a new live C2 before C3. It is running in root session69759.
- Fault20 `providers20-group-a1` completed exit0 and released card5.
  Root independently reviewed C2 135/135 and C3 270/270,408/678 references
  matching actual bytes, all135 outputs changed, exact target/group OFF recovery.
  Independent raw C2 statistics: B self comparable; B/F first QMM output Max
  16.25->14.0, 132 preceding null fields, no preceding incomparable fields.
  F contains2 later NaN fields, so whole-capture E5 remains false. Sidecars are
  `R/providers20-group-a1-statistics-{self,bf}`. Root grants only the scoped C2/C3
  in `native20_group_summary.json`; full review sidecar is
  `R/providers20-group-a1-independent-review-a2`. Earlier audit attempts preserve
  their wrong directory/field-interpretation diagnostics; evidence is unchanged.
- Fault28 legal/replay: actual final argument/launch and same-process OFF checks
  passed. New raw pairs each have10 finite/14 null fields and no first difference.
  No model/C2/C3 or historical0.7968% inheritance. Agent4 is preparing a separately
  disclosed FP64 collector; no changes to original raw JSON.
- Fault33 real FAI host/kernel build completed, six entries compiled and public
  serializer checked. `attention33-{stock,baseline,fault}-a1-cases.json` prepared
  with source-a21/packages-a5. Execute stock, then B and independently compare
  complete payloads, then F only after that bridge. Stock a1 passed all complete
  payload repeat/OFF/healthy checks. Private B a1 failed at model initialization:
  private libopapi.so hides stock AddRmsNormBias. No F; Agent4 fixes coexistence.
- Collective25 both8.5.1/9.1 completed6 bounded cases,4 repeats, healthy collective,
  drain and normal exit. Every case NO_REPLAY_SAMPLE; no F/C2/C3.
- Native26 inventory, natural39 C2,41 group and disclosed FP64 collector are merged.
  Immutable source-a22 (511files,9d06f984) archive SHA
  `b7029d396a416f0d6c1e48eed361d8a5d567991151d55b84ff2bcceb707cda7a`.
  CPU-only26 QMM inventory-a1 passed after a slow global initializer compile.
  Root stopped owned FIA CPU duplicate (exit137) after an equivalent bounded
  byte-storage optimization passed CPU contracts. Fresh optimized builds needed.
  No NPU mounts; use postwrite-build-a2 for natural39/41.
- CPU integration after fixture-only b7754769:645 tests passed,2 platform skips.
  Earlier missing test-module invocation was a command typo, corrected by22
  targeted tests. No production identity check was relaxed.

R remains `/home/17-fault-c2c3-20261117`. Root owns all NPU starts, checks the
controller and idle gate before scheduling, and never treats cleanup as OFF.
Current NPU task is providers39-a6, source-a21/packages-a5, card5/session69759.
Fresh source/package snapshot needed for26/41/FP64 tools.

Read this section first. The initial checkpoint below is retained as history.
Latest checkpoint (source-a15, integration fec83416):

- CPU integration: 591 tests passed, 2 platform skips. No NPU qualification follows.
- `providers39-a4`: both healthy Worker observations contained113 native MatMul
  calls through `vllm.unquantized_gemm`. The public-pair adapter did not recognize
  that custom op and refused before injection. Hits0, unregister CLEAN, hooks0.
  Actual image source is in `R/gemm-source-a1`; do not drop unmatched calls from
  the denominator. a3 was a separate Resource_Busy failure before inference.
- `natural18-group-a1`: natural pretrained HF Qwen scope,113 eligible RMSNorm
  targets. Runtime C2:113/113 once; C3:226/226 across two forwards; same-process
  OFF exact and leases/hooks0. Standard Worker is NOT qualified. Raw statistics
  dump identities align but construct maps differ from observer stride queries.
  E5 remains refused pending fresh collection; never edit the old raw maps.
- `postwrite-natural-a1`: natural HF39.output_proxy B/F actual matmul output and
  consumer verified; exact OFF and leases0. Independent raw audit v2 finds
  Torch.matmul.0.forward/output/0.Max,536->Inf, with228 preceding null fields and0
  preceding incomparable fields. B self is comparable; F has7492 NaN statistics,
  so whole-dump E5 remains incomplete. This is a controlled output proxy, not
  native MatMul range qualification. The41 baseline failed wrapper cleanup;
  reviewed atomic collector-preserving restoration is integrated. Fresh B/F
  `postwrite41-natural-a2` is running card5; check before scheduling NPU.
- `attention-preflight-a3`: modern ACL observer compiled on8.5.1 and9.1. Actual
  attention28-launch-a2 captured successful launches but lacked binary/function
  identity and argument snapshots. Collector diagnostics add launches. NOT
  launch-qualified; use actual registration evidence, never expected profiles.
- `tnd33-build-a2` and schema-a1 compiled, but exact source analysis found natural
  Worker TND/page16 follows FAI rather than the built arch22 branch. Preserve as
  unentered investigation artifacts. Agent4 is adapting real FAI host/device
  KV stack512->256 with a new matching key and workspace. Actual chip16 AICores;
  original prefill/default routing must remain intact.
- `build20-v3-a1` actual8.5.1 SDK build passed; library SHA
  `0c3e3f5fc73401f7b6e9bcdba261c730cf8ba17d6389c90a3843399da6b3df30`.
  Full-request group glue is pending review; build success is not NPU acceptance.
- Immutable source-a15 archive SHA
  `1c1d1467d60a7adc0254c61414d46f4d7dffcf8c794391d822f423fef1617e6b`.
  Earlier running/prepared states below are superseded by this checkpoint.

Current scoped qualification remains **C2 0/17, C3 0/17** pending actual receipts.
Root alone schedules NPU, with shared lock/per-case gates and new attempts. Raw
data and binaries remain124; no expired Docker isolation or foreign-process kill.
Remote root R is `/home/17-fault-c2c3-20261117`; SSH `npu-dual` port2125.

Actual NPU results:

- `collective25-smoke-a1` rejected HCCL_CONNECT_TIMEOUT60 before calibration;
  actual SDK requires120. Corrected smoke-a2 and bounded-a1 completed both ranks'
  repeat/OFF, health AllReduce, barrier, sync, destroy and normal native shutdown,
  exit0. Six FP32/FP16/BF16 bounded cases all NO_REPLAY_SAMPLE. No F/C2/C3.
  Bounded supervisor SHA256:
  `a63d5ba69459b89a58f6f592be5445c20fe0b6781325c91aa289afe6e076ef1c`.
- `providers39-a1` failed in actual Worker discovery before injection: msprobe
  custom-op wrapper lacked `.default`. Native hits0, owned hooks/owner released.
  Exact collector packet compatibility is now in source-a11/packages-a2.
  `providers39-a2` is running C2 then conditional C3; inspect its actual result.
- `natural-census-a1` encountered a transient external card2 process after gate;
  aclInit Resource_Busy, no inference. It was not killed. New census-a2 completed
  two exact healthy forwards each for standard Qwen Worker and pretrained VAE.
  Worker uses native/custom RMSNorm, fused rotary and unified attention, not an
  ordinary index_select path. VAE format30 is base NCDHW, not private NZ; that
  select/copy chain cannot qualify Fault15. Stock VAE k3 Conv3d is not Fault17's
  patch16 geometry. These are inventories, not injected model qualifications.

Builds and immutable inputs:

- Final `build39-a3` native dir:
  `R/build39-a3/cache/2c83a5742eb393f7b7bd202f421b1b8abed5f42dc433e86dda23f40412f89ff7/attempt-a0743e472ccd4f5dbb32831f35c89123/native`.
  Parent receipt SHA256
  `aef9d4a0b84b32e6df11012678cab5d13efcc5f9d493ceae807c5bb4a264416a`;
  native library SHA256
  `04f834ef2e479f2578ec2c33a7e895097a2eb2a5c0e619dea33a497c5bfbc07c`.
- Attention preflight-a2 observer and real FIA TU compiled on both exact SDKs.
  `R/attention-preflight-a2/{cann851,cann910}/attention-preflight/result.json`:
  8.5.1 SHA `f540f6f088ecc5706fb2ea79ed4d100f3478076581ae872f05cc0ba355fecfb6`;
  9.1 SHA `19e85248eed2a754c4406bde442358dcb3822fa48336240cf29a95347f03cb83`.
  Preserve `/build/attention-preflight` absolute library paths when mounting.
- `tnd33-build-a1` full8.5.1 package rejected ascend950/mc62 OpDef registrations.
  a2 from source-a10 records exact-source registration adaptation, retaining
  ascend910b and tiling/kernel bytes. Read actual completion; build success alone
  cannot qualify Native launch. Official source remains pinned c77c0829.
- `postwrite-build-a1` new public ACL library built with no devices. Native dir:
  `R/postwrite-build-a1/cache/0b1ef8930beb40c4cbdb248b6f73b35796f279357111ee44946a4a33490ea754/attempt-717de3e550c44e4f94ce4e2c1fbe2ecc/native`.
  Library SHA `5c3f9c73233ca7336d1b419d7073dcfa05d80e99dbc2a62ec2a645afc79ad8b8`.
  `postwrite-natural-a1-cases.json` prepared B/F pairs, not yet executed here.
- Immutable source-a1..a11 remain remote. Source-a11 archive SHA
  `25989295e5de5ff6572210883eefcd4350a7ff57cb414bf4f59735c115f064ff`.
  Installed packages-a2 derive from it. New main-tree group18/22 adapters,
  statistics audit, saved coverage audit and quant inventory need a new snapshot.

Next: finish39 actual all-target group recovery,28 final launch,33 exact-image
  build and natural TND Worker, natural39/41 B/F, natural18/22 groups/statistics,
  DeepSeek all-projection inventory and Native group gates. Keep model-local
  pretrained evidence distinct from standard Worker and constructed evidence.
  E5 reads raw B/F only, fixed rtol1e-5/atol1e-8; null/NaN remain visible.

## Initial device-free checkpoint (superseded above)

Frozen base: `91decf5e`. This worktree/branch is the main integration checkout:
`codex/17-fault-c2c3-closure-20261117`. The parent worktree is dirty historical work;
do not edit it. All seven specialty worktrees exist independently from the frozen
base under sibling `.publish/c2c3-{coverage,tensors,collective,attention,nativeproviders,postwrite,evidence}-20261117`.

Qualification remains C2 0/17 and C3 0/17 pending real scoped receipts. Root owns
the NPU queue and final qualification. Raw artifacts stay on 124. No NPU experiment
has been executed in this checkpoint; only device-free compiler/ABI checks.

Integrated specialty commits: Coverage `6363182` plus whole-group drain barrier
`7361d44`; Native39 `3876c51`, independent pre-admission Run observer `3ee4fec`,
control-test fixture correction `2493744`; Attention observer/preflight `67ff8e8`.
Do not replay already cherry-picked commits. Check `git log` before integration.

124 root: `/home/17-fault-c2c3-20261117`, SSH `npu-dual` port2125.
Source archives a1–a4 are immutable snapshots with `source_manifest.json`.
Latest a4 archive SHA256:
`0a26c8ce660041c504d95606c358646c94aebe32e6c07674852cc5db67ef714f`.

Build39-a1 compiled but its new CPU ABI fixture failed because the previous invalid
input mask was not reset. No runtime guard was weakened. Build39-a2 passed after
the fixture correction and includes the independent observer; receipt:
`build39-a2/cache/ac03918acdeb15c38926b22e8cf9bff42d6098debbb510f259a7d24558a31087/attempt-a1721f51d4eb462697a4164ab4ba8b54/receipt.json`,
SHA256 `5a2e1ec18c3009184130cd6d46e12d8d7f6d5914248715319e7ff689f5e58780`.
The provider agent is adding actual temporary-device readback and the full natural
Worker dispatcher; its next Native change requires a fresh build before NPU use.

Attention CPU preflight a1:

- 8.5.1: real official FIA translation unit compiled; observer failed because
  `prof_common.h` was outside the supplied include directories. This is not proof
  of an unsupported version. Agent4 is fixing header discovery.
- 9.1: observer and FIA translation unit compiled. No final launch qualification.
- Logs/results: `attention-preflight-a1/{cann851,cann910}/attention-preflight/`.
- Official pristine source: `/home/17-fault-engineering-20261117/ops-transformer-pristine-a1`,
  commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`.

Available models confirmed by read-only inventory:

- `/home/gyl/models/Qwen3-0.6B`
- `/home/haoran/models/DeepSeek-V2-Lite-W8A8`
- `/home/worker-positive-20261117/models/GLM-4-9B-0414`
- `/home/fault39-integration-20261116-a1/models-smollm2-135m` (prior verified)
- `/home/hunyuan-vae-cann91-ab-20261022/models/vae` contains real
  `AutoencoderKLCausal3D` config and 986000558-byte `pytorch_model.pt`; historical
  scripts, sources, dependencies and review are preserved alongside it. Its
  natural Conv3d inference is a candidate for Agent2; not a standard vLLM scope.

CPU interpreter: parent workspace `.codex-tmp/unified-env/Scripts/python.exe`.
Use `-B`, `PYTHONPATH=src;fault39/src;.;tests`, unittest (pytest is not installed).
Coverage regression: 50 tests passed. Queue safety contract: 6 passed. These are
CPU results only. New raw statistics must not enter Git; publication review must
bind checked file bytes to staged Git blobs and preserve all prior receipts.
