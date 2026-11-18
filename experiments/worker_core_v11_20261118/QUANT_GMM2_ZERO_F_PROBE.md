# One current WeightNz observation

This increment observes the single typed `aclnnGroupedMatmulWeightNzGetWorkspaceSize` / `aclnnGroupedMatmulWeightNz` pair identified in [the fixed-wheel static review](QUANT_GMM2_STATIC_ENTRY_REVIEW.md). It does not assign routed rows, alter quantization, run a numerical reference, or authorize C1/C2/C3. The next useful result is a current GWS-to-container-registration-to-executor join within the actual GMM2 source window. Four creator/destroy observations alone do not establish that join.

The fixed scope is one TP1 eager B8 first decode in the same standard vLLM Worker process. All 135 quant projection calls, 135 Native prepares and executes, and 270 public quant API calls remain in the census. The prior 10/135 request-row proofs must remain 10 in this experiment; the other 125 are retained. Twenty-six GMM2 windows must each contain exactly one observed prepare and execute. New typed observation errors do not suppress, repeat or change an original operator invocation. They refuse the diagnostic result.

## Separate Native and runtime sources

Root built the new Native observer from frozen **source106**, commit `922ff7cb1d343a3e16f28d43776552b00123428d`, source manifest SHA `2317f5947c73aa5ff7ae6644fd9b1ef26aaa36aab325d20557b870ff13db0bb4`.

- Actual SDK bundle: `/home/17-fault-c2c3-20261117/quant-gmm2-trace-sdk-a1`.
- `build.json` SHA: `6a2414e6e02a17d1abe06175a294b850f56908c24b31625b951708448996e620`.
- Build receipt SHA: `4b09364f8c847aa750548b389af21706936ec3cf30331b36c08c9b5e3989920d`.
- Observer DSO SHA: `51b0d92465232cbc26ad2d93416b028bd1601c3dcadc4282b25304512600526d`.
- Root reported 19 original and 28 new SDK-typed C++ CPU contracts passed, startup original resolution passed, and the owned build container absent after cleanup. This is CPU/SDK evidence only. No GMM2 observation on NPU has been accepted for this new pair.

The Python binding and probe require a later runtime snapshot/package. They do not change the 13 compiled source files or require pretending the SDK was compiled from that later runtime commit. The case records both `source_commit` and `native_compile_source_commit`; the startup original records the actual Native build/profile digest and compile commit. The old quant group bundle remains independently pinned at build SHA `b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`.

## Startup and observation

`VLLM_ENABLE_V1_MULTIPROCESSING=0` and `TASK_QUEUE_ENABLE=0` are mandatory. Before even importing vLLM, `prepare_startup` attaches to the exact already-preloaded observer with `RTLD_NOLOAD`. It validates both ABI layouts, calls the lookup-only resolver and checks all five resolved functions by actual pointer, canonical image path, full file SHA, exported symbol and offset. Native resolution also requires the original four container APIs and all seven typed descriptor getters. A failure prevents LLM construction; setup cannot repair it by loading a library.

The same installed Python module retains this actual pre-LLM object. After standard `DemoncatWorkerExtension` construction, an ordinary read-only Worker RPC binds the real PID, `/proc` boot/start identity, current thread, Worker/runner/model objects, and observed/configured logical device. `require_host_driven_core` independently requires the actual `InprocClient`. A separate Worker process is refused. No diagnostic Worker mixin, model initialization wrapper, scheduler override or recursive identity RPC is added. Physical card 5 is a controller assignment, not the Native logical device ordinal.

The observer remains uninstalled during warmup and healthy requests. It starts only after the actual B8 prefill and immediately before the one captured decode. Each actual GWS records all 23 formals, including the signed `quantGroupSize`; it is never filled from the public schema. Fourteen TensorList slots and the IntArray slot reference the live creator registration, entry/return sequence and copied metadata. The public size getters recheck those current objects. The current group-list tensor descriptor is read through the seven actual SDK getters, whose owned arrays are released. No opaque container layout or floating activation is read.

The prepare-return executor has a fresh registration. Execute records its actual executor, workspace pointer/capacity, stream and runtime logical device; it must join that prepare in the same scope/thread/device. Address reuse needs a new registration. A returned execute is not advertised as device drain. The observer has host journals only, and never consumes a prepared executor on behalf of the model.

Pure `query` reads counters. Explicit `advance(..., collect=True)` stops observation and collects the existing projection census. Explicit `collect` saves immutable originals, then recomputes pair counts, current container lifetimes, named-slot payload references and executor joins. Cleanup saves a separate immutable attempt even when stop or hook removal fails. The GMM journal releases before the creator journal; missing execute, pending/untracked executor, active scope or retained container blocks release. The first failure is preserved across retry. The preloaded libraries remain resident.

## Root-only deployment

Source-only export needs one new installed module, `fault_injection/quant_gmm2_native.py`. Its imported `quant_container_native.py` and `metadata_arrays.py` are already in the previous container diagnostic closure. Existing serving/row/source modules remain unchanged. The standalone probe additionally mounts the exact `probe_quant_containers.py` support file read-only and verifies/reports its SHA; it is not assumed to be in the installed Python package. The new probe is `probe_quant_gmm2.py`; the preparer is `prepare_quant_gmm2_case.py`.

Before any NPU case, root can run the new binding's `NativeGmm2(...).preflight()` in the same fixed image with no device nodes, the built observer supplied by startup `LD_PRELOAD`, its actual read-only driver dependencies, and the new installed package. This only checks marker/layout/NOLOAD resolution and `/proc` identity; it does not import torch/vLLM, call `aclrtGetDevice`, install a journal or invoke an operator. Library constructors may probe driver/platform information, so it is not a claim of zero driver function calls. Retain that separate load receipt, including a failure.

After review and fresh runtime packaging, prepare (not launch) one case:

```sh
python3 "$RUNTIME_SOURCE/experiments/worker_core_v11_20261118/prepare_quant_gmm2_case.py" \
  --installed "$NEW_INSTALLED_PACKAGE" \
  --quant "$EXISTING_REVIEWED_QUANT_ND_FLAT_BUNDLE" \
  --trace /home/17-fault-c2c3-20261117/quant-gmm2-trace-sdk-a1 \
  --source-commit "$RUNTIME_COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --card 5 \
  --out "$NEW_CASES_JSON"
```

The standard root queue still owns the live card gate, startup, timeouts and removal. Bounds remain case 1800 seconds, 80 GiB, 16 CPUs, creator calls 4096/events 8192/metadata 8 MiB, GMM calls 128/events 256/pending pairs 32/journal 1 MiB. There are no repeated route-index captures or synthetic helper quantizations.

Success requires complete current 26 pairs, the full 135 census with the same 10 proofs, unchanged before/after healthy tokens and zero owned observer resources. Missing/unknown registration, a different caller or original owner, any count/argument/lifetime mismatch, budget overflow or failed cleanup is a preserved refusal. Even success is `ZERO_F_CURRENT_WEIGHTNZ_ENTRY_EVIDENCE`, with `request_rows_proven=false` and C1/C2/C3 false. Only after reviewing those actual slots can a later implementation join routed token indices to their real current GMM inputs and outputs; this probe does not do that join.

## CPU validation

The local isolated worktree passed **93 tests, zero skips** with the following modules (43 new binding/probe tests and 50 existing builder/container regressions):

```sh
PYTHONPATH=src:fault39/src:.:tests:tests/worker_extension \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m unittest \
  test_quant_gmm2_native test_quant_gmm2_probe test_quant_gmm2_build \
  test_quant_container_build test_quant_containers test_quant_container_review
```

The new tests use CPU metadata/model/ctypes doubles, including a full preflight-before-constructor, actual enqueue/explicit-collect ordering and healthy-before/after orchestration. They cover every missing resolved function category, different PID/start/thread/model/device, pure query, old-registration/address reuse, swapped slots, negative scalar preservation, incomplete prepare/execute, workspace mismatch, first-error retention, partial begin/release, failed stop/removal, live getter arrays and immutable snapshot copies. They do not establish a real SDK-to-Python load or natural NPU call; those remain separate root checks. The pre-existing local Torch package emitted its optional Visual C++ redistributable warning after the successful unittest run; no test was skipped.
