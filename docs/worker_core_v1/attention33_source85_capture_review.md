# Source85 normal-pair capture review

This is a CPU-only review and capture correction. Source85 originals, the
Native implementation, strict mapped-file identity gates and F admission are
unchanged. No F, C1/C2/C3 or E5 qualification is issued.

The preserved source85 independent report has SHA256
`26dc3be024e9dfc57adbbfb00a0bb5ffa7c7e20b9dd76f70b44febdf982077f7`.
It records exact bytes for all 435 cross-arm payload pairs and two remaining
`ORIGINAL_RUN_FAILURES_PRESERVED` issues, one per arm. Each arm's five original
boundaries retain both `BOUNDARY_ACTUAL_FUNCTION_JOIN_MISSING` and
`BOUNDARY_MAPPED_LIBRARY_IDENTITY_CHANGED`. The independent reader already
resolves the first gap from raw events; it does not rewrite the capture or
discard the mapped-library failures. Original evidence is not committed.

## Current-scope function binding

`paged33_boundary_probe.read_scoped_launches` previously required exactly one
matching function-bind in the whole recorded history. Repeated successful
bindings during earlier warm-up/layer calls therefore caused a false missing
join. The live capture now follows the independent reader's narrower rule:

- The actual launch must join the latest registered handle generation and
  retained binary bytes. The actual PID and native TID must match the Python
  boundary's separately captured native-thread domain.
- All bindings preceding the launch for that PID/function handle are inspected
  before selection. Every record must be successful and agree on registered
  handle, generation and tiling key. Conflicts are never filtered away.
- Exactly one binding belongs to the current scope. It must be the unique
  latest effective binding, from the same native thread, between actual begin
  and launch records in both raw order and monotonic time.

The report retains the selected raw line and prior-identical-history count.
Missing/duplicate current bindings, later foreign bindings, conflicting old
keys/handles/generations, failed bindings, stale registrations, incorrect raw
order and substituting Python ident for native TID remain refusals. No stock
kernel POD is decoded using a private ABI.

## Mapped-library failure evidence

The capture checks exactly these basenames: `libcust_opapi.so`, `libopapi.so`,
`liboptiling.so`, `libcust_opmaster_rt2.0.so`, and `libtorch_npu.so`.
The a2 ten-file stage contained the first, second and fifth, but not the two
tiling libraries. The root's subsequent CPU image scan identified production
paths for additional staging:

- `/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so`
  (49,089,352 bytes).
- `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so`
  (23,091,888 bytes).

Those paths are deployment candidates, not proof of the original failing
mapping. The old private boundary contains only the generic error. That error
combined inode, device and the 512 MiB file-size bound, so it cannot identify
which file or condition failed. Neither staging candidate is deemed fixed
without a new actual capture.

On failure, `mapped_api_libraries.diagnostic` now retains all matching raw
maps rows, raw/resolved paths, inode and device fields, current-file stat,
individual `failed_checks`, successful earlier file hashes, and hash-before/
after stat where available. Deleted or unavailable files retain their raw
map row. The top-level error preserves the original exception text.

Every mapping row for a resolved path must agree with stat; a later row cannot
hide an earlier conflict. The original exact inode/device checks and 512 MiB
bound remain. Hashing also requires unchanged device/inode/size/mtime afterward.
Any failure stays `UNOBSERVED_OR_INCOMPLETE`; it does not become accepted file
evidence. Even complete file fingerprints prove mapped presence only, not the
selected ACLNN resolver owner.

CPU contracts exercise those failures, including a successful file preceding
the failure, hash-time mutation, and the actual session's diagnostic save path
with exactly one call to the original API and its original return object.
The separate cold-probe-import test correction is commit
`f5c406a9f0dbd78a0b5920dc2bf37321b9dd8386`; it changes no production code.
