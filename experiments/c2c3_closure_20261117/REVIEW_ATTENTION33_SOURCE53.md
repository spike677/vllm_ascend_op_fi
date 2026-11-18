# Source53 first-boundary offline diagnosis

This reader accepts only the two preserved source53 diagnostic results. It does
not edit them, replay inference, open a device, load a runtime library, execute
subprocesses, compute E5, authorize a fault or grant C2/C3. NumPy interprets CPU
byte files. The existing frozen contract/bridge byte readers are loaded only
after their files match the exact source53 manifest. The old bridge validator is
not weakened or relabeled to accept the diagnostic schema.

Source commit: `b017a43ac8cabc48b9090e9db25914739f9730b5`.
Manifest SHA256: `8cf340e5d51bd9fbafa2437a9e9fa128ab1d178596e46e3da4f41fde9572ed36`.
Stock result SHA256: `fde2b7717c93162aeb2d4d1e3fe5b56ed1901b775d9976a574ea899e1a6b5077`.
Private result SHA256: `68157b6970a94498cf9a91cfd718593b6d7ddbb930cbc7f08a6a508d561a69a3`.

The root coordinator runs this on preserved host files, outside any container:

```sh
python3 -B /path/to/review_attention33_first_boundary_source53.py \
  --source-root /home/17-fault-c2c3-20261117/source-a53 \
  --stock-run /home/17-fault-c2c3-20261117/attention33-first-boundary-source53-card2-a1/attention33-fd-first-boundary-stock/run \
  --baseline-run /home/17-fault-c2c3-20261117/attention33-first-boundary-source53-card2-a1/attention33-fd-first-boundary-baseline/run \
  > /path/to/new-source53-first-boundary-offline-a1.json
```

The script emits a bounded JSON report: hashes, shapes, counts, paths and numeric
difference summaries; no tensor arrays or raw launch arguments. Exit 0 means
the diagnostic was reconciled, including any stock/private differences. Exit 2
means an evidence join or preserved-byte check failed. Both outcomes keep
`bridge_pass`, `fault_authorized`, `c2`, `c3`, `e5` and public qualification false.

For all five phases in each arm it verifies the duplicate result/boundary files,
actual unique request mappings, the standard Worker capture source, complete
28-layer metadata, unchanged argument objects at recorded checkpoints, exactly
one original API invocation and cleanup receipts. It rehashes all existing 87
complete payloads in each phase, plus every new complete query and immediate
output bin. It compares raw query/output bytes numerically, including signed-zero
bit differences and empty optional outputs. Full block-table values are encoded
as their recorded little-endian integer dtype and rehashed, including unused
table entries. Scalar/optional inputs, mask hashes, schema/defaults and logical
K/V hashes form a complete comparison signature without process-specific object
identities. Actual object/storage joins remain separately enforced within each
phase.

The original `runtime_observation` is retained verbatim. The old helper rejected
more than one successful historical binding for the same function, even when
each binding was identical. The new `new_offline_runtime_resolution` requires:

- One original begin/end pair for the exact token, PID/TID and time window.
- The launch's actual registered binary handle/generation and stable binary
  file, with matching original registration and no newer handle generation.
- For `aclrtLaunchKernelWithHostArgs`, exactly one matching bind in this scope.
  It must be the latest effective bind before the launch. Every earlier bind of
  that function handle in the same process must have the same binary handle,
  generation and key; conflicting history cannot be discarded by filtering.
- A successful unchanged-argument launch, valid full raw argument snapshot,
  observed blocks/schedule, and successful completion synchronization.

Identical historical bindings are counted, not deleted. Future bindings cannot
retroactively alter the earlier call. Unrelated earlier registration gaps remain
reported; they never establish complete coverage of runtime entrypoints. The
private first launch also joins its saved adapter call and raw argument/binary
hashes. Stock arguments are never decoded using the private ABI, key or layout.

Evidence limits remain explicit:

- Logical K/V **before** the API call have complete hashes, not preserved bins.
  Their identity/page mapping/checkpoint hashes are compared with the rehashed
  **after-consumer** logical K/V files. Those files are never relabeled as
  independent pre-call byte captures.
- Mask/optional tensor snapshots have full hashes but no raw files. Their
  before/after snapshots and cross-arm hashes can be compared, not independently
  reconstructed. Scalar after-values are source-bound check results. Full
  block-table values and query/output bins are independently rehashed.
- Equality at checkpoints cannot rule out an intervening write followed by
  restoration when inference tensors lack version counters.
- Registered binary fingerprints identify the intercepted launches. Mapped DSO
  hashes remain recorded fingerprints, not freshly rehashed external files or
  evidence of which ACLNN API-library owner was selected. That owner and complete
  entrypoint coverage remain unobserved.
- A stable immediate layer-0 return difference with equal captured inputs
  localizes the first observed discrepancy to that original API boundary. It
  does not establish the kernel/ABI/algorithmic cause. Different launch profiles
  are reported as observations, not a causal proof.
- Private `healthy_after` retains the private vendor process with the scoped
  adapter removed. It is not recovery to stock libraries. The two arms remain
  diagnostic-only; a non-exact bridge continues to reject F.

CPU contracts cover the repeated-history resolution, conflicting keys/handles,
generation changes, wrong scope/thread, missing sync, raw snapshot corruption,
object replacement, complete block-table padding, K/V pre/post distinction,
delayed output mutation, signed zero and empty outputs. They use CPU-generated
byte fixtures, never NPU evidence.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:fault39/src:.:tests \
  python3 -B -m unittest tests.test_attention33_first_boundary_review -v
```
