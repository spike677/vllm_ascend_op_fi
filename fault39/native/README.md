# Fault39 native v1

This directory extracts the real CANN 8.5.1 `aclnnMatmul` lifecycle from
`fault20_fast_validation/source/l4_metadata_lifecycle_proxy` and the private
`OpExecCache::Run`, `AddOpExecCache`, and both `AddParamToBuf` overloads from
`fault20_fast_validation/source/l4_buffer_substitute_allhit_v1`.
`metadata_capture.inc` preserves required-tensor metadata access; no optional
QMM/null rule is used. The original files are not modified. No old fault control
symbols, shared memory, routes, or global reload/reset are included.

`bash build.sh --cpu-test` compiles mathematical, layout, bound, alias, pattern,
padding-preservation, and finite/nonfinite observer tests without CANN. A plain
`bash build.sh` additionally builds the aarch64 ABI0 interposer and metadata
proxy against the real 8.5.1 installation. `FAULT39_ASCEND_TOOLKIT`,
`FAULT39_REAL_LIBOPAPI`, and `FAULT39_OUTPUT_DIR` select paths; a 9.1 installation
cannot be enabled by changing an expected-path string. The default toolkit is
`/usr/local/Ascend/cann-8.5.1/aarch64-linux`; `pkg_inc` is required.

The build produces `libfault39_native_v1.so`, `libopapi.so`, and
`real/libopapi_real.so`. The first must be preloaded into the actual worker, and
the proxy must precede the real operator API in that worker's library search.
`VLLM_SERVER_DEV_MODE=1` is required. Merely loading the DSO is not evidence of
Run interception. Check `interposed_run_calls`, `typed_metadata_calls`, actual
candidate observations, and a successful live lossless gate.

## Public C ABI

`fault39_api.h` is the single ABI definition: version 1, 368-byte configuration,
natural C alignment. `fault39_abi_version_v1()` returns 1. The ctypes provider
uses symbols from the already loaded process rather than opening a second
runtime instance.

1. Bind the worker's actual rank with `fault39_bind_worker_rank_v1(rank)` during
   worker setup. Binding is immutable except for an identical repeated value.
2. Configure mode 1 (discover), with an owner, rank/device filters, byte cap,
   positive bounded `max_hits`, and `input_mask=3`. Every discovery clears the
   prior candidate inventory. Two actual observed Runs are required for target
   selection; the Python layer must additionally compare repeated workload
   inventories. Shape-only silent first-match selection is not supported.
3. Disable the exact owner/generation and wait for its drain acknowledgment.
4. Configure a discovered canonical signature for mode 2 (inject), 3 (lossless),
   or 4 (observer). A signature shared by multiple observed cache identities is
   `AMBIGUOUS_TARGET`. Rank is independently checked against boot binding and
   `aclrtGetDevice` on the Run thread. `input_mask` supports self-only (1),
   mat2-only (2), or dual (3) in lossless mode; other modes require 3.
5. Around actual scheduler work, set thread-local scope with
   `fault39_set_context_v1(owner, actual_request_ids_json)` and clear it on every
   exit. The JSON must be a nonempty array of nonempty strings. Missing scope
   produces `UNPROVEN_REQUEST_MAPPING` and forwards the original Run unchanged.
   Configuration never supplies or fabricates request identity.
6. Disable/clean supply exact owner/generation and a bounded timeout. Neither
   changes another owner. Clean clears only this runtime's Fault39 state.

Configuration starts disabled. It never mutates original model input storage.
The first finite/nonfinite observation with a nonfinite output disarms further
hits. A single selected Run is allowed in flight at a time. Copy/synchronization
failure disables injection, marks `RECOVERY_FAILED`, and retains every lease.
An explicit clean may retry recovery only on the owning thread, device, and ACL
context after successful synchronization. Timeout or failed recovery never
reports a successful drain. The DSO does not free retained leases on shutdown.

## Numerical and evidence contract

Only rank-2 ND BF16 matmuls with positive, injective strides, positive M/K/N,
even K, consistent storage extents, in-bounds offsets, and nonoverlapping
original/temporary storage are eligible. Inputs and output are all mandatory.
The structural signature contains API, cube math type, rank/device, dtype,
format, logical shapes, strides, offsets, and storage sizes, with no raw pointer.

The exponent is historical 58 or auto (`-1`). Auto selects the smallest
nonnegative integer e <= 63 for which a finite single product `2^(2e)` and
`(K/2)*2^(2e) > FP32_MAX` hold. The comparison is exact integer arithmetic;
K=2,4,6 have no solution. All five patterns in a comparison share the selected
amplitude. The block patterns mirror the second K half with the opposite sign,
including incomplete blocks. At small K, differently named patterns can be
identical; their names alone are not evidence of distinct reduction order.

`fault39_snapshot_v1` returns required JSON bytes including NUL; insufficient
buffers are not partially written. `last_observation` is tied to owner,
generation, hit and the actual thread request list. It contains original input
counts from D2H, temporary input counts from the **host constructed** buffers,
and output counts from post-Run D2H, plus shape/stride/slot/address proof and
copy/Run/sync/free statuses. Temporary device contents are not independently
read back: `input_device_readback_verified=false`. Finite counts include zero;
positive and negative counts exclude zero. NaN/+Inf/-Inf are separate.
Finite min/max/max_abs/mean_abs are null if no finite elements were observed.

The production runtime has no fake NPU backend. CPU tests validate only the
pure transformation and validation core, not vendor ABI compatibility, actual
worker interception, device stream lifetime, or intermediate overflow location.
Those require root-controlled live 8.5.1 validation. A nonfinite result after
the mathematically zero cancellation construction is evidence of an
intermediate-range failure, not proof of a specific accumulator implementation.
