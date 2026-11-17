# Native Build Manager

This package executes reviewed build scripts in a fresh, immutable directory. It selects an adapter from the mechanism and the **observed operator**. It does not select a shared object by CANN version, import torch-npu, initialize a device, or create a Worker. A successful build has `native_qualified=false` and `actual_npu_calls=0`.

Run from a complete source checkout in the target Linux aarch64 environment after its normal CANN environment has been selected:

```bash
python3 -m fault_injection.native_build recipes
python3 -m fault_injection.native_build inspect --mechanism 39.matmul_range
python3 -m fault_injection.native_build --cache /work/native-cache build --mechanism 39.matmul_range
python3 -m fault_injection.native_build --cache /work/native-cache build \
  --mechanism 20.buffer_substitute --operator aclnnQuantMatmulWeightNz
python3 -m fault_injection.native_build verify --artifact /work/native-cache/KEY/attempt-ID
python3 -m fault_injection.native_build loaded --artifact /work/native-cache/KEY/attempt-ID --pid WORKER_PID
```

The build result returns `native_directory`; no user-selected `.so` is needed. A deployment integrates the same `NativeBuildManager` API with its actual operator inventory. `20.buffer_substitute` without an operator is rejected because WeightNz and QuantMatmulWeightNz have distinct contracts. `--repository` can select a complete checkout when the package is installed elsewhere. Building from a wheel alone without the recorded C++ and shell sources is refused.

| Recipe | Execution boundary | Current build admission |
| --- | --- | --- |
| `fault39.matmul.abi0.v1` | MatMul OpCache, ABI0 | Actual CANN 8.x/9.x files, symbols, exact header signature compilation and identity tests |
| `fault20.weight_nz.abi0.v1` | WeightNz six-argument GWS | Existing CANN 8.5.1 and exact audited torch-npu PTA resolver fingerprint |
| `fault20.qmm.abi0.v1` | QuantMatmulWeightNz fifteen-argument GWS | Same audited installation restriction, separate selected API and exported state functions |
| `fault26.qmm.abi0.v1` | QMM output writer and live consumer | Existing 8.5.1 route, exact custom AddRmsNormBias header/library and compiled signatures |
| `fault26.fia.abi0.v1` | FIA V3 destination writer | Same actual installation checks, distinct FIA writer ABI |
| `postwrite.output.acl.v1` | `aten.mm.default` returned tensor with public ACL writes | Actual CANN 8.x/9.x public ACL headers, symbols and compilation; separate controlled proxy |
| `postwrite.cache_tail.acl.v1` | `aten.index_copy_.default` live cache slot | Same public ACL backend, distinct mechanism and selected operator |

These are executable build recipes. This table grants no NPU, model, A2/A3, C1, C2 or C3 qualification. The postwrite variants do not inherit MatMul OpCache or historical dispatcher acceptance. Fault26's two recipes currently compile the same supporting trace/proxy set, but select different writer runtimes; the unused writer is not required by the loaded-library check.

`inspect` fingerprints actual installation metadata, environment script, complete CANN header inventory, ELF files, exact required exports, compiler/tools, source bytes and packages. An initialized Worker's SoC observation is included if already available. Without it, runtime SoC qualification remains required. The tool never infers hardware compatibility from a CANN version string. Unknown private ABI fails at symbol/header/compile checks; the narrow Fault20 and Fault26 admission rules remain intact.

The custom Fault26 API is resolved as a paired `libcust_opapi.so` and `aclnn_add_rms_norm_bias.h` from installed vLLM-Ascend/build layouts. An actually mapped library wins. Distinct unselected artifact copies are ambiguous; byte-identical installed/build copies may share one identity. No custom ABI is inferred from size alone.

The cache key binds these inputs and the operator-specific recipe. Each build holds an exclusive lock, pins compiler/tool executables, records its full log, recompiles ABI probes, runs the existing recipe checks and rereads all inputs before issuing a receipt. Source or installation changes invalidate the result. Failed attempts stay in place with `failure.json`. Timeouts terminate only the build's subprocess group. A cache hit rechecks artifact bytes and the current inputs. No vendor binary, full tensor or new raw statistics should be committed.

`loaded` reads the target process's actual `/proc` maps and process start identity, verifies inode/file identity and hashes against the receipt, and rejects changed, missing or ambiguous mappings. Run it in the same filesystem namespace as the Worker. It establishes mapped-file identity only. Native in-memory attestation, actual API/slot/storage binding, original execution, fault effect, OFF and Recovery still belong to each Provider's live qualification.

CPU tests (`python -m unittest discover -s tests -p test_native_build.py`) exercise recipe separation, architecture rejection, ELF validation, immutable fingerprints, lock ownership, path containment, cache/source invalidation and rejection of a false NPU claim. Fixture ELF files in those tests are parser inputs, not compiled NPU libraries.
