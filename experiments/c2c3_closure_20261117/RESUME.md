# Current execution checkpoint

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
