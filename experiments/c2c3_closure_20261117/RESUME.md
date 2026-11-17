# Execution checkpoint

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
