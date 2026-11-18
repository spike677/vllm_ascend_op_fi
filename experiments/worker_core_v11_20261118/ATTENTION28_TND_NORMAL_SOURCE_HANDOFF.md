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

The final local run passed all 44 tests: the 19 new contracts plus the existing
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

## Root integration delta

No shared builder, binary registry, fault33 helper, Host, Runtime or provider
was edited. Root needs these explicit additions in its separate integration:

| Component | Required additive change |
| --- | --- |
| build_fresh.configure_policy | Recognize only the new source receipt schema/variant/vendor; verify this recipe, base generator, normal header and contract header hashes. Never route it through old legal/replay admission. |
| SDK compatibility | Permit only the exact reviewed CANN8.5.1 compatibility/profile for this new variant, preserving all old variant guards. Record actual interface/dependency/source hashes. |
| Compile and binary registration | Compile and validate exactly recipe.KEYS: normal key 103000000030000003 plus the six unchanged fallback entries listed by the recipe. Do not borrow fault33's key list or add an unobserved fault/FD key to satisfy a mismatch. |
| ABI schema | Run the existing real-SDK compile_public_schema.py against this fresh source; retain generated layout/size/field/type/offset and source/hash receipts. Do not inherit an old schema qualification. |
| CPU build | Compile/run the portable normal contract below; additionally compile the complete new host header through the actual host build. Preserve failures as separate build attempts. |
| Completion | Preserve the new normal receipt, recipe/header/contract/projection hashes, complete vendor byte inventory, actual compiled key metadata and both SDK/CPU receipts. loading_allowed, fault_execution_ready, stock_private_exact_proven remain false. |
| Worker admission | Separate new boundary/native binding; exact public V and mask objects plus actual prepared descriptor/copy lineage; no public_candidate result used as live authority. Retain every Slice/FIA launch, join the exact current serializer/stream/kernel-argument record, and preserve the complete 28-site inventory. |

The preparation commands are concrete, but build_fresh.py intentionally
does not yet accept the new variant:

    python /source/fault_injection/native_adapters/attention28/fia/apply_tnd28_prefill_normal.py \
      --source /build/new-pristine-c77 --output /build/new-normal-source-receipt
    g++ -std=c++17 -Wall -Wextra -Werror -pedantic \
      /source/fault_injection/native_adapters/attention28/fia/test_tnd28_prefill_contract.cpp \
      -o /build/tnd28-normal-contract
    /build/tnd28-normal-contract
    python /source/fault_injection/native_adapters/attention28/fia/compile_public_schema.py \
      --source /build/new-pristine-c77 --sdk /usr/local/Ascend/cann-8.5.1 \
      --out /build/new-normal-schema

After a successful fresh build and the missing live-view binding, root can
queue the complete stock-repeat/private-normal-repeat/full-input-and-output
join/same-private-Worker-OFF matrix in
ATTENTION28_TND_PREFILL_ADAPTATION_REVIEW.md. It must use the actual strided
V and int8 mask, fixed fresh 2048-token workload, new package identity and real
platform/serializer evidence. Unknown descriptor lineage, changed bytes,
extra/missing launches, FD/BRC, budget exhaustion or cleanup failure stop
normal acceptance. There is no F queue in this increment.
