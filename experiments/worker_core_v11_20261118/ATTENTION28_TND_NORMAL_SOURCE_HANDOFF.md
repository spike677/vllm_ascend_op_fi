# Fault28 TND2048 normal source candidate

This increment prepares one independent normal implementation for the actual
source87 prefill scope. It does not add a public mechanism, fault/replay path,
normal-equivalence certificate or C1/C2/C3 qualification. No new Native or NPU
call was executed by this change.

apply_tnd28_prefill_normal.py consumes a clean c77 checkout and verifies
12 independently supplied source hashes. It adds one include, one final host
capture, and an exact geometry exception to the old CheckSpecConditions
route preference. Removing those three additions restores the original source
bytes. M=512, S2=512, split-core logic, mask/softmax arithmetic, serializer
definition and every kernel file remain unchanged.

The new private names are tnd_prefill_arch22_normal and
dcat_f28_tnd_normal. The receipt's 28.fia.normal_candidate label is private
build metadata; it is not a registered FaultConfig mechanism.

## Evidence boundaries

tnd28_prefill_geometry.public_candidate is an offline metadata contract. It
accepts the actual source87 V view, including stride [4096,128,1], offset 3072
and storage 16777216 bytes. A contiguous replacement is rejected. It preserves
all four current object/storage identities, null inference versions, tensor
hashes, the actual fresh request and the distinction between supplied
arguments and declared schema defaults. It does not verify a device pointer,
mask values, current allocation liveness or a Native temporary copy.

The host predicate reads actual prepared gert shapes, descriptors, lengths
and attributes. These are not the public Tensor view. It requires the current
910B platform's real AIC/AIV pair with CV1:2, rather than deriving it from a
stock launch block count. Final capture checks the actual used-core count and
the same SDK CalcTschBlockDim result. The source array bound is AIC<=26.

The pinned parser establishes three distinctions:

- TND Q/KV length arrays [2048] are cumulative; final flags must both be 1.
- sparse_mode3 canonicalizes Native preToken=INT_MAX,nextToken=0.
  Raw actual attributes are retained separately, not replaced by Python
  defaults.
- With no block table, GetS2SizeForBatchContinuous does not call
  GetBlockSize. The initialized parser block size remains 0, while the
  actual raw API attribute is 128. Both values are separately checked and
  recorded. Requiring typed 128 here would reject the reviewed parser.

The host record explicitly leaves public_prepared_pointer_stride_join=false,
prepared_stride_values_verified=false, mask_values_verified=false, and
kernel_argument_raw_observed=false. isTensorV1 and presence of the parser's
K/V stride metadata are recorded; missing values are never labelled
contiguous. A later Worker binding must prove the original public V view and
actual prepared pointer/copy/stride chain before accepting normal equivalence.
The current header's candidate route is not an authorization to load the
package into serving.

The normal route is independent of observation environment values. Capture
requires a matching real dcat_attention_observer_scope_matches_v1 token,
PID, native thread, current /proc/self/stat start ticks, owner, generation,
site ordinal/object and target-set digest. Only baseline, baseline_repeat,
off and recovery_off are accepted. Capture cannot select M/S2 or a fault key.
Owner identity is checked again before bounded output is appended. Runtime
owner reservation, draining and health remain the existing Core's responsibility.

## CPU results and limits

The focused Python suite has 19 tests, including all 4096 (S1,G) rows,
the complete 32 unpruned M/S2 tiles per KV head, every public shape/stride/
offset/storage/dtype, optional features, fresh-request identity, repeated
site order, no dropped site, and source-pin/patch-anchor refusal.
The analytical causal pair count is 4,196,352 per KV head and 33,570,816 across
eight KV heads. This is not an executed split-core traversal.

check_tnd28_normal_materials.py independently rehashes source87's actual
36.7MB result, all five exact c77 kernel git blobs, both real parser sources
and the host material. It checks all 56 calls from the two original requests,
without copying private payloads or deleting any of 28 sites. The new small
CPU summary records the hashes and explicit negative qualification fields.
Its SHA256 is 98d2d3b1d181be9e594e07cf2462df2a5acea306fef59368510da2eeabce2e5c.
That immutable summary belongs to the original d79831d6 source-only input;
use that frozen source root to recompute it. The later scope fix and builder
integration below require their own new receipts and do not relabel it.

The original source-only d79831d6 local run passed all 44 tests: the 19 new contracts plus the existing
10 prefill-boundary and 15 stock-census contracts. Earlier regression attempts
had missing test-module search paths; the corrected full command passed:

    PYTHONPATH=src:fault39/src:.:tests:tests/worker_extension:experiments/c2c3_closure_20261117 \
      python -m unittest test_tnd28_prefill_normal test_attention28_prefill_boundary \
      test_attention28_stock_census -q

test_tnd28_prefill_contract.cpp exercises the actual SDK-independent
postcondition header, including changed M/S2, FD/BRC, key, canonical token/
block fields, core bounds and all 4096 coordinates. There is no C++ compiler
in this local Windows environment, so this C++ binary and the real SDK header
compile are **pending**. Python passes do not stand in for them.

## Additive builder integration, still not loadable

The new explicit build_fresh.py / binary_registration.py branch accepts only
`tnd_prefill_arch22_normal` and vendor `dcat_f28_tnd_normal`. Old legal/replay
and every fault33 variant/default remain separate. The root-owned
build_tnd33_124.py controller adds only strategy `tnd28-prefill-normal`;
this work adds no Docker controller and performs no container operation.

Before compiling, the builder reconstructs all 17 source materials from the
actual c77 Git blobs plus the current normal recipe/headers. It verifies the
12 fixed source pins, every original/updated byte, generated typed projection,
complete unique manifest, exact M512/S2=512 scope and seven normal/fallback
keys. A forged manifest cannot approve altered kernels or generated headers.
The new branch requires the actual reviewed CANN8.5.1 interface manifest SHA
`1a9c9159e9d1d84b82b754b2db806be1b878e05d06d45d3df07295c1280878e2` and
`ascend910b-cann851` compatibility. The existing source/dependency/SDK checks
still run, including all actual interface file hashes.

The build runs the strict C++ contract with `-pthread`, compiles the complete
host header in the fresh SDK host translation unit, and derives the public
serializer through that actual compile_commands entry. Its field names,
types, widths, counts, non-overlapping bounds, generated projection, decoder
binary and CPU sentinel receipt must agree. The completion receipt rehashes
all new CPU/schema originals, frozen patched source and the complete vendor
byte inventory. Exact seven-key registration and binary SHA are mandatory.
No old fault33 normal certificate is an input.

A review of the original capture placement found that process-wide mode
could make unrelated host calls throw. The fixed production Capture uses
CaptureCurrentScope, shared directly with the executable C++ contract. Its
initial thread/PID/token/observer and geometry probe is non-throwing. Foreign
threads, absent/wrong tokens and different geometry return without capture.
Only after both current scope and geometry match does strict validation run;
owner/generation/token changes then reject, including a change during the
geometry probe. The scope is rechecked before the bounded append. The CPU
contract creates an actual foreign std::thread and tests this production
helper, not a separate model of its predicate. Geometry values in this CPU
contract remain explicitly synthetic; actual SDK/NPU geometry is not proven.

The new tests exercise source reconstruction, missing/duplicate materials,
wrong variants/keys/SDKs, modified generated bytes, actual schema-field joins,
compile/selftest failure, stale output bytes, and a prior authorization flag
that must never be silently overwritten to false by completion. The full
focused regression command passed 110 tests with no skips or failures:

    PYTHONPATH=src:fault39/src:.:tests:tests/worker_extension:experiments/c2c3_closure_20261117 \
      python -m unittest test_tnd28_prefill_build test_tnd28_prefill_normal \
      test_attention28_fresh_build test_fia_binary_registration \
      test_paged33_arch22_s2_recipe test_paged33_arch22_bridge \
      test_arch22_all_normal test_arch22_all_normal_review \
      test_attention28_prefill_boundary test_attention28_stock_census -q

The first new registration fixture omitted the existing required MIX/support
metadata and was corrected; the production registration guard was unchanged.
Local Python source/compiler fixtures are not actual SDK compilation. There
is no local C++ compiler, so the executable C++ contract and the actual SDK
host/kernel build must run in a new root-controlled attempt. A previous
build or its ABI certificate cannot be reused for these changed source bytes.

### Exact no-device build handoff

After root integrates the strategy mapping and freezes the new source, run
the following on 124 with that frozen source directory as the only argument.
It recovers the *recorded* SDK manifest path from the existing controller,
rehashes it, and invokes the existing controlled builder. No library is
loaded in a Worker and no NPU request is authorized by this command.

```sh
python3 - /home/17-fault-c2c3-20261117/<new-frozen-source> <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
source = Path(sys.argv[1]).resolve(strict=True)
root = Path('/home/17-fault-c2c3-20261117')
old = json.loads((root/'tnd33-arch22-normal-build-a1/controller.json').read_bytes())
mounts = [arg for arg in old['command'] if 'dst=/sdk_interfaces.json' in arg]
assert len(mounts) == 1
options = dict(part.split('=', 1) for part in mounts[0].split(',') if '=' in part)
interfaces = Path(options['src']).resolve(strict=True)
assert hashlib.sha256(interfaces.read_bytes()).hexdigest() == (
    '1a9c9159e9d1d84b82b754b2db806be1b878e05d06d45d3df07295c1280878e2')
subprocess.run(['python3', str(source/'experiments/c2c3_closure_20261117/build_tnd33_124.py'),
    '--source', str(source), '--out', str(root/'tnd28-prefill-normal-build-a1'),
    '--interfaces', str(interfaces), '--strategy', 'tnd28-prefill-normal'], check=True)
PY
```

The controller retains its pinned actual image, runc, network=none, no
NPU devices, readonly input mounts, owned-container cleanup and 7200-second
build deadline. An existing output directory is refused; a retry uses a new
attempt name. All build failures remain separate originals. Completion is
only `CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION`, with `loading_allowed`,
`fault_execution_ready`, `stock_private_exact_proven`, public/prepared lineage
and C1/C2/C3 all false.

After a successful fresh build and the missing live-view binding, root can
queue the complete stock-repeat/private-normal-repeat/full-input-and-output
join/same-private-Worker-OFF matrix in
ATTENTION28_TND_PREFILL_ADAPTATION_REVIEW.md. It must use the actual strided
V and int8 mask, fixed fresh 2048-token workload, new package identity and real
platform/serializer evidence. Unknown descriptor lineage, changed bytes,
extra/missing launches, FD/BRC, budget exhaustion or cleanup failure stop
normal acceptance. There is no F queue in this increment.
