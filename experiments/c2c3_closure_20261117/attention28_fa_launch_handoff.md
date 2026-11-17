# Fresh FA launch experiment: ready for root's queue

No container or NPU was started by this implementation/review. Do not resume an
externally stopped run. Queue only in the root-approved device window.

The actual fresh CPU build is
`/home/17-fault-c2c3-20261117/fa28-fresh-build-a1/build/run`.
Read-only hashing verified all 851 artifacts, 27,801,135 bytes, no mismatches
or extras. Exact receipts and important SHAs are in
`attention28_fa_build_review.json`. Completion SHA is
`7189c077dad9411e53240d6a0cc9190477d00433d5b1325833c0fcb26c7cac5d`;
the new ELF SHA is
`967265862973a761ae7de2c7b54600e246f067898cd2c97f297aa9a12e784eb0`.
Its real build image ID is
`sha256:9008234fbe4948ce9e0383c34972044b369e42dc7711d44d5878f52267b8614a`.

## Independent argument interpretation

The unchanged, exact-image runtime capture library can observe FA. The new
Python `fa/launch.py` independently binds the actual new FA entry declaration,
metadata and source-built opbase argument constructor. There are 25 declared
pointers, 24 compiler `parameters`, and an 8-byte hardware-sync prefix in this
910B MIX route. The resulting tiling pointer slot is **200**, derived from this
FA source, not FIA's 280. The public FA POD is **1464 bytes**; compiler
`opParaSize=1472` remains separately recorded and is not substituted for that
public struct size.

`launch.observe_launch` takes no host record, selected-factor profile or expected
key. It selects by independently captured registered ELF and actual function
binding, then uses the runtime's placeholder descriptor at the source-derived
FA slot. It requires successful launch and post-launch synchronization. Other
collector launches can be excluded only using their independently bound ELF.
Missing/ambiguous binding, a duplicate FA launch, another key/function, budget
overflow, wrong slot or host-input overlap fails. Finding the expected host
bytes elsewhere in an argument buffer never selects them.

After that extraction, FA compares complete actual bytes with the owned host
callback, decodes all 205 public fields using the **new build's schema**, and
runs the **new build's C++ POD decoder** against the actual captured bytes.
All 192 opaque public SDK section bytes are retained. The five exact audited
padding fields (17 bytes) remain in raw evidence and host/final equality; only
the explicit semantic OFF/repeat comparison excludes them. All 100 matmul
fields and the source's typed scalar projection are cross-checked. Actual FA
launch arguments have not yet been observed in this new generation.

The existing FIA verifier calls the same generic observation function and
then its own original FIA serializer checks. It does not acquire FA schema or
scope. No C++ capture source changed, so the already freshly compiled observer
at `attention-preflight-a4/cann910` is usable after its source/library/header
hash checks. Its library SHA is
`ef4cfd1962fbafd41edb600e26291fcf65740c68c64500dd232a2ca1c9a84a49`.

## Minimal root preparation command

After integrating this commit into `source-NEW`, run this CPU-only manifest
preparer, which starts no container:

```sh
R=/home/17-fault-c2c3-20261117
python3 -B "$R/source-NEW/experiments/c2c3_closure_20261117/prepare_attention28_fa_cases.py" \
  --source "$R/source-NEW" \
  --build-root "$R/fa28-fresh-build-a1/build/run" \
  --observer "$R/attention-preflight-a4/cann910" \
  --batch 8 --card 5 --selection-target joint \
  --out "$R/fa28-launch-cases-a1.json"
```

Root then passes this new manifest to its existing lock/resource controller.
It produces one bounded case by default: **B, Brepeat, F, truly unscoped OFF**.
The first call independently discovers this run's native M/nRatio from final
POD bytes. The default `joint` toggles both M=64/128 and nRatio=4/8 from that
actual pair, so the KV grouping also changes. `query_tile` and `key_group`
remain explicit single-selector alternatives. No historical
native pair or certificate is loaded. If the observed native pair is outside
the source's candidate set, the run refuses instead of forcing a route.

The probe uses the confirmed `torch_npu.npu_fusion_attention_v3` keyword
contract and creates new, deterministic modular FP16 Q/K/V plus a complete
causal bool mask. Every input byte is frozen and rechecked after each call.
The original API return object passes through. All tensor leaves, shapes,
dtypes, complete bytes and scalar/container leaves are retained after the
collector window; nonfinite output is a failure. Brepeat and OFF must restore
complete output bytes and final semantic POD/key/block/schedule. OFF has no
`NES_FA_*` environment at all and needs no host record to prove its launch.
Failures preserve the current call ID, requested plan, runtime/host evidence
and original exception. Cleanup never overwrites another environment owner.

The manifest mounts the observer directory at `/build` and the new FA build
at the nested `/build/run`, both read-only. The source snapshot is `/driver`,
which also preserves the decoder's original header paths. At runtime, all five
SDK interface hashes, all 217 decoder compile inputs and the exact observer
headers are checked before NPU work. The image is pinned to the fresh build ID.
`ACLNN_CACHE_LIMIT=0` and `TASK_QUEUE_ENABLE=0` are mandatory. The per-case
budget is 900 seconds, 8 CPUs, 32 GiB; only root schedules devices/locks.

## Optional subsequent raw B/F collection

Add `--include-statistics --statistics-backend cpu-fp64-v1` and
`--cpu-preflight "$R/attention-statistics-cpu-a1/probe/result.json"` to prepare
two additional, separate baseline/fault cases. Each first discovers native
geometry, then executes two identical measured calls, collecting only the
second. The prepared CPU FP64 adapter remains explicitly disclosed and pinned;
original collector config/package/support YAML identities are retained. No raw
dump, stack, construct, null, NaN or first-difference record is edited or chosen
by this probe. Add `--batch 1` as a second batch if desired; the preparer caps
the total at six cases. Initial launch debugging needs only the sequence case.

Successful status means new standalone FA launch/argument observations and
requested sequence checks completed. The original stock-package bridge,
natural Worker qualification, workspace allocation size observation and
independent statistical comparison remain separate. Reports keep C2/C3=false,
`public_worker_qualified=false` and no historical qualification inheritance.

CPU tests cover independent FA slot/POD selection; rejection of FIA schema or
slots; matching bytes at a wrong offset; opaque/padding preservation; fresh C++
decoder disagreement; host matmul mismatches; unscoped OFF; exact source and
MIX-argument contracts; and selection derived from the newly observed native
pair. Synthetic CPU traces are explicitly not NPU launch evidence.

Validation completed: the new bundle validator ran read-only on the actual
remote build and checked its 851 artifacts plus source-derived ABI, with zero
remote writes, containers or NPU calls. The targeted FA/FIA/TND CPU suite ran
48 tests successfully (one environment-dependent skip). Source hashes and
the exact test command are retained in the build review JSON.
