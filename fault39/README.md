# Fault39 standalone native integration

This package implements only `FLOAT_MATMUL_CANCELLATION`. The blue environment
does not contain the current internal `src/demoncat` framework. Its integration
status remains **`INNER_FRAMEWORK_INTEGRATION_PENDING`**: the document-contract
adapter and tests are not evidence that the inaccessible catalog, bindings,
controller or CLI have been changed.

Historical Blue-124 results and independent statistics are indexed in
[the acceptance report](reports/README.md); its source fingerprint refers to
the historical `build-a10`, not the current version-adapted build.
[The 2026-11-16 acceptance](../experiments/provider_adaptation_20261116/README.md)
separately verifies the current `source-a5` / per-version `build-*-a5` on actual
CANN 8.5.1, 9.0.0, 9.1.0 and 9.2.0 installations at operator level, plus Qwen3
TP1/TP2 on 8.5.1 and TP2 on 9.1.0. The 9.1 model runner uses Python `source-a7`
with the same accepted native `build-91-a5`.
Qwen3 and a real Llama implementation retain separate historical acceptance
records; no arbitrary-model or arbitrary-8.x/9.x certification is implied.

The native core derives from the historical CANN 8.5.1 metadata/cache runtime.
See [source audit](docs/historical_source_audit.md),
[frozen contract](contracts/fault39_contract.yaml), and
[historical reference profiles](contracts/reference_profiles.json).
The historical A2/A3 distributions are references, not required output counts.
`finite` includes `zero`; the two must not be added together.

## Actual injection boundary

```text
real aclnnMatmulGetWorkspaceSize
  -> typed self/mat2/out metadata
  -> AddParamToBuf formal identities + AddOpExecCache association
  -> actual OpExecCache::Run addresses
  -> owner/rank/device/signature/request gates
  -> temporary self and/or mat2 storage
  -> original CANN Run with substituted input addresses
  -> stream synchronization + full logical output observation
  -> release temporary storage
```

No original model parameter is modified. No returned result is replaced or
forced to Inf. In cancellation mode, `self[m,k]=+A` and
`mat2[k,n]=sign(k)*A`. Each column has exactly K/2 positive and K/2 negative
values. The mathematical result is zero; the real operator may produce finite
values or a nonfinite range anomaly depending on its reduction implementation.
This does not identify a particular accumulator instruction or guarantee an
anomaly for every legal shape.

`half`, `alternating`, and mirrored `block256/512/1024` are parameters of one
primitive. All patterns in a matrix use the same exponent. Historical mode uses
`A=2^58`; auto mode chooses an integer exponent satisfying the documented FP32
reference product/range conditions. That reference is not a claim that a kernel
internally uses FP32 throughout.

## Generic interface and current limits

| Component | Implemented behavior | Acceptance boundary |
|---|---|---|
| `Fault39Config` | Pattern, amplitude policy, actual rank/device selection, hit and memory limits | No user-specified layer names |
| Native metadata | Dynamic M/K/N, BF16 ND, positive injective strides, storage bounds, alias rejection | Other dtype/API/format routes are rejected |
| Target discovery | Stable real signature and cache identity across repeated workloads | Ambiguous candidate signatures/caches are refused; cache uniqueness does not establish model-layer uniqueness |
| `NativeBackend` | prepare/register/discover/enable/disable/query/clean/unregister | Owner-scoped; no global `reload_weights()` |
| Worker extension | vLLM 0.18/0.23 actual rank and scheduler request scope | Eager, synchronous host task queue; `TASK_QUEUE_ENABLE=0`; actual runner contract remains checked |
| Inspector | Full native logical statistics and independent raw msprobe comparison | Native hit logs cannot select the msprobe first difference |
| Host adapter | Versioned document-contract adapter + mocks | Yellow framework wiring still required |

The interface has no model-name dispatch and does not hard-code Qwen dimensions.
This is a design property, not arbitrary-model certification. P5 has separately
validated Qwen3 and Llama with real weights; their exact scopes are listed in the
acceptance report. TP4 is a mock-control test until separately validated on real
hardware.

The strict `unique` target policy remains the default. Real Qwen3 discovery
found five legitimate geometries and correctly refused to choose one. An
explicit optional `max_k_min_m_unique` policy was subsequently added: maximize
the actual K (reduction pressure), then minimize M (self-buffer working set),
and still require a single stable signature/cache. This rule does not inspect
model names or layer strings. Ties and changes to the selected target across
workloads are refused. Changes to nonselected geometries are not rejected by
this optional policy.
Both the original refusal and any subsequent policy validation must be reported.

A CANN cache can be reused by several model layers. The current model runner
uses `max_hits=1`, which substitutes the first matching Run in the armed request
scope. This establishes a selected cache/geometry and a bounded actual hit;
it does not establish a unique layer or coverage of every matching layer.

Dynamic positive injective strides and nonzero storage offsets are implemented
and covered by CPU tests. The current real operator probe uses zero-offset,
complete dense backing storage, including a transposed `mat2` view. Nonzero
offsets and other accepted stride layouts still require separate NPU acceptance.

## Recovery semantics

Register starts disabled. Disable closes admission for the exact owner and
generation, waits for in-flight Run completion, synchronizes, and confirms zero
buffer leases. Timeout yields `DISABLE_PENDING`. Copy/synchronization/free
failure yields `RECOVERY_FAILED` and retains leases rather than freeing memory
still in use. Recovery retries require the owning thread/device/context.

Each hit clears its previous observation and binds new evidence to its own
generation and hit number. Inspector acceptance also requires a terminal native
state; an older successful observation cannot validate a later failed hit.

Successful real OFF runs demonstrate the normal disable/re-enable path only.
The live native ABI owner-isolation probe rejects foreign configure/disable/clean
and stale-generation disable, preserves the active owner's complete state, and
then verifies that the owner's real MatMul still executes normally. This does
not establish composition with another internal-framework fault. Forced timeout,
synchronization/copy/free failure recovery and integrated multi-fault coexistence
remain mock/contract-only. TP4 is also mock-only. These scopes must
remain separate from any reported real TP1/TP2 results.

## Building and running on 124

Historical NPU acceptance used `quay.io/ascend/vllm-ascend:v0.18.0` with CANN 8.5.1.
The current source supports capability-checked, per-installation builds rather
than rejecting every version except 8.5.1. CPU builds against actual 8.5.1,
9.0.0, 9.1.0 and 9.2.0 and their operator NPU B/F/OFF have passed. Real Qwen3
inference passed on 8.5.1 (TP1/TP2) and 9.1.0 (TP2); 9.0/9.2 model inference
remains unvalidated. See [version differences and exact scope](docs/cann_version_adaptation.md).
`native/build.sh` uses the installed headers, symbol/signature checks and an
immutable build profile. Vendor libraries are copied only into the private
runtime build directory; vendor binaries and headers must not be published.

The installed torch-npu 2.9 `torch.mm` path calls **aclnnMm**, which this Fault39
does not target. The operator probe therefore uses `torch.matmul`, whose actual
Run identity must be observed as `aclnnMatmul`. The installed split
`libopapi_nn.so` path also needs the private metadata proxy. This mapping is
restricted to owned experiment containers; the shared host libraries are not
modified. Loading a DSO alone is not interception evidence.

`tools/package_source.py` makes immutable source archives with SHA-256 manifests.
`tools/run_124.py` takes the existing shared resource lock and checks real NPU
processes and open device handles before every case. Generic Docker/OCI device
mappings are recorded as access permissions. A running dedicated `ptq_` ATK
container mapped to a selected card is additionally treated as a reservation,
even before its NPU process appears. Only task-owned containers are cleaned by
the controller. Each batch contains at most eight cases.

Run stages in order:

1. `contracts/p2_cases.json`: disabled warmup, repeated discovery, then an
   observer-only Run. All three native original addresses must exactly match
   the live PyTorch backing storage addresses. No buffer substitution occurs.
2. `contracts/p3_cases.json`: self-only, mat2-only and dual lossless substitution;
   compare full storage hashes, full output bytes and OFF recovery.
3. `contracts/p4_cases.json`: five separate cancellation cases. Each fresh
   process repeats all earlier gates before changing inputs.
4. `tools/vllm_validate.py`: real model discovery, lossless, pattern and recovery
   stages. Model ambiguity is reported, not bypassed by selecting a layer.

The controller maps physical cards through Docker device cgroups. Within the
single-card container the device is logical 0; physical card IDs are not passed
as `ASCEND_RT_VISIBLE_DEVICES` logical indices.

## Evidence interpretation

Native evidence records original input D2H statistics, host-constructed temporary
statistics, actual address relations, Run/synchronization/free status and output
D2H statistics. `input_device_readback_verified=false` is intentional: temporary
device buffers are not independently read back. Do not relabel host construction
as a device-readback measurement.

The separate msprobe path keeps raw `dump.json`, `stack.json`, `construct.json`,
configuration and collector identity with `task=statistics`, `level=mix`,
`summary_mode=statistics`. `tools/compare_msprobe.py` reads only those artifacts,
validates alignment and uses `abs(F-B) > atol + rtol*abs(B)`, with explicit
defaults `rtol=1e-5`, `atol=1e-8`. Null remains unobservable. NaN, signed Inf and
finite values remain distinct. The report can identify the first observable
statistical difference, not necessarily a unique physical cause.

Replacement occurs below Python: msprobe can legitimately record equal caller
inputs while native evidence proves different temporary inputs reached MatMul.
The independent findings must subsequently be linked using real request scope;
unproven relationships retain `UNPROVEN_REQUEST_MAPPING`.

The operator lossless probe compares every output byte and hashes the complete
input backing storages for self-only, mat2-only, dual and OFF controls. Model
lossless validation instead checks native execution/release evidence and the
generated token/text response digest against baseline. Matching model responses
do not prove that every model intermediate tensor or logit is bitwise equal.

## Tests and handoff

Run `python -m unittest discover -s fault39/tests -v` with `fault39/src` on
`PYTHONPATH`, plus `tools/test_worker_scope.py` and
`tools/test_native_probe_slots.py`. Native pure-core and production-ABI guard
tests run during `native/build.sh`. These tests do not substitute for real NPU
acceptance.

See [the concrete yellow wiring guide](docs/yellow_integration.md). Yellow
integration should implement `demoncat_fault39.host_contract.v1` against
the actual catalog/bindings/controller and compose the Worker Extension with
its existing extension. Do not pretend this document protocol is the inaccessible
framework's actual API. Re-run rank selection, request mapping, owner isolation,
disable/drain and B/F statistics acceptance after wiring.
