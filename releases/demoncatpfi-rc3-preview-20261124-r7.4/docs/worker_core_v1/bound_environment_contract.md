# Explicit mapped function environment

The default `environment_fingerprint` is unchanged: any unavailable,
identity-mismatched or ambiguous mapped image prevents its fingerprint. There
is no inode-only, overlay, pathname-hash or installed-version exception.

Source71 exposed an actual deployment mismatch: six public image libraries had
the same inode but executable maps used device `253:2` while opening the path
reported the overlay device `0:47`; `map_files` could be readlinked but not
opened. The old attempt remains failed. Root can stage the exact original
library bytes on a separately mounted filesystem and bind them read-only to
their original paths before creating a new Worker. New maps/stat/file hashes
must then pass normally. This source does not stage or load those libraries.

The new explicit `inspect_bound_environment` profile accepts live
`NativeSymbolHolder` objects from a reviewed deployment. It verifies the
getter's actual ctypes address, `dladdr` symbol and unique executable segment
against that holder's exact mapped file and manifest SHA before calling it.
The getter reads an existing address slot. The returned address must resolve
to the exact expected symbol in one verified executable mapping. Getter and
target identities are read twice; mappings and pathname identities are then
rechecked. An unresolved cache, zero/foreign address, source mismatch, changed
mapping or unverified additional image refuses the whole profile.

Duplicate basenames retain every image and its file-verification status. A
collision must have an actual held getter or target in its complete group.
Other images are retained and fingerprinted as additional verified images;
they are not asserted to be used by that function. The scope material includes
all image paths/hashes/build IDs and the exact getter-to-target edges. A new
lookup, a path string or serialized JSON cannot stand in for a live holder.
ASLR addresses and PID remain in audit evidence, outside the stable digest.
This is loaded-image evidence, not C1/C2/C3/E5 or runtime compatibility.

## Existing holders and new builds

- Native26's `dcat26_group_cached_address_v1` reads the existing fourteen ACL
  function-pointer fields. It refuses an unresolved cache, unknown/foreign
  owner thread or active callback. It never invokes the resolver or ACL.
- Its custom proxy's `dcat26_custom_cached_address_v1` exposes the two actual
  `g_real_add_rms_norm_bias*` pointers used by the original continuation. It
  never calls `pthread_once`, `dlsym` or the operator. The existing serial,
  quiescent Worker boundary is required by the reader.
- Postwrite39 has no custom dynamic cache. `dcat_pw_linked_address_v1` reports
  the three direct linked ACL relocations already used by its transfer code.
  It neither resolves a replacement nor opens a lease. This is explicitly
  `LINKED_FUNCTION_RELOCATION`, not Torch/custom GEMM implementation proof.

The Native data/ownership ABIs and existing numerical paths are unchanged;
these are additional read-only symbols. The writer group build recompiles its
custom proxy into the **new output bundle** and runs an actual-source SDK CPU
getter contract. The parent trace build's source/metadata receipt must be fresh
because it pins the group runtime source. No old bundle or evidence is rewritten.
The postwrite build also runs its actual-source SDK CPU getter contract before
creating the new library. Both builds reject absent exports. Native26 records
`bound_symbol_profile='demoncat.writer26.cached_addresses.v1'`; old bundles keep
the old strict environment route. The public 39 group requires the new export.
Default C1 and old source-bound receipts do not inherit these new group scopes.

Root uses the existing NativeBuildManager/trace parent procedure followed by
`providers26_build_group.py` for 26, and the existing `build_postwrite.sh` recipe
for 39. Those commands are root-owned SDK builds. This agent's Windows CPU tests
exercise real ctypes getter signatures and synthetic map ownership, not the
Linux loader, actual SDK C++ binaries, CANN numerical behavior or NPU acceptance.
The included C++ tests are executed by those build entries, without device calls.

Final local validation ran 134 CPU contracts against the independently
deliverable source, with all 134 passing. This includes the real Host/Runtime
26 C2 and 39 C2/C3 callbacks, default environment refusal, actual ctypes getters,
identity/collision/race negatives, original-provenance failures, Native cleanup
regressions and the distinct output digest boundaries. It does not include
SDK compilation or an actual Worker/NPU acceptance run.

## Output digest boundaries

New public39 originals label two distinct hashes. `forward_artifacts` and
`actual_health` hash the PyTree containing `(actual root hidden, complete logits)`
with per-tensor geometry/byte digests. `actual_requests.output_sha256` hashes the
single final logits tensor's dtype, shape and bytes. Compare corresponding
B/OFF/health within the same boundary, workload and group generation. Those two
algorithms and value boundaries must never be compared directly. Old originals
are unchanged; the new labels do not affect data, recovery or qualification.
