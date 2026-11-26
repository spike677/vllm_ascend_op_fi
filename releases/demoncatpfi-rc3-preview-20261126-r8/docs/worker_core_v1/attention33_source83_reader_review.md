# Source83 normal bridge: thread domains and cold CPU reader

This change fixes observation/review contracts for a new normal-only pair.
It changes no Native code, kernel choice, geometry gate, F authorization or
mapped-library identity check. Source83 originals remain rejected evidence.

The retained a1 report has SHA256
`bb7e3c794edcc594aa6d833ed816f485f56f7419192ce8638c2052bea324b7bd`.
Its 137 issues comprise 112 thread comparisons, 20 blocked CPU-reader imports,
two preserved original failures and three downstream completeness failures.
The report's frozen capture/reviewer source hashes match the pre-fix files
reviewed here. The original result, metadata extracts and report are not
committed or rewritten.

`FirstDecodeBoundaryCapture.thread` was `threading.get_ident()`, while the
Native observer's `tid` is OS `gettid`. The private first boundary recorded
Python ident `281463120013952`; its matching host token recorded PID/TID 1/1.
Comparing those integers directly was a reader error. This does not establish
an absent mapping: source83 did not capture the boundary's OS TID separately.

New captures retain the legacy Python `thread` field and add an explicit
`thread_identity` containing PID, Python ident and `get_native_id()` together.
Capture and stock-scope owner guards check both thread domains before use or
release. The reviewer requires the pair, verifies the legacy Python alias,
joins every actual scope begin/launch/end to its captured PID/Native TID, and
continues checking the private host and complete Native receipt against that
same raw launch. All five phases must keep the same pair and Worker identity.
Old results missing the pair report `BOUNDARY_NATIVE_THREAD_IDENTITY_NOT_CAPTURED`;
neither a large integer nor matching PID is used to manufacture a Native TID.

The CLI previously installed the no-dynamic-load audit hook before the first
lazy NumPy import. A cold NumPy/ctypes initialization then hit that hook and
prevented complete payload review. The CLI now first initializes only NumPy's
CPU byte reader, refuses ambient device frameworks, and exercises FP16 signed
zero, BF16 decoding/nonfinite classification and FP32/FP64 differences. It
records NumPy version and package/core file paths and hashes, then immediately
installs the existing permanent audit hook. The report labels this CPU-reader
purpose; these file identities do not certify a device library. No Torch,
torch_npu or model runtime is imported during this initialization.

Cold-process contracts read real synthetic files through all three reused
readers, verify finite values and signed-zero differences, reject nonfinite
BF16 bytes, and verify a subsequent real `ctypes.CDLL` call and Torch import
are rejected by the hook. Thread tests deliberately use unequal Python/native
IDs, reject missing or inconsistent pairs, and reject foreign begin, launch or
end records. Capture tests retain references on a changed Native TID and allow
the original owner to finish after the condition is removed. The complete
two-arm CPU fixture retains 112 private Native receipts and all 435 payload
pairs; synthetic execution does not grant a device qualification.

Both source83 boundary extracts also retain
`BOUNDARY_MAPPED_LIBRARY_IDENTITY_CHANGED`. That original failure is independent
of these fixes and remains a refusal. Root will provide the actual production
same-byte library mounts for a new case. The new capture, reviewer and packages
need a new source freeze; the unchanged fresh Native normal build may be reused
only through its existing exact-source/ABI checks. Full stock/private numeric
equality has not been established by this CPU patch. F, C1/C2/C3 and E5 remain
ungranted.

## Normal repeats with actual page relocation

The separate source83 partial numeric a3 report has SHA256
`1329b1982c51c821a18ce6867dd4f21bd0f0626c83567f524af4ded5d8241e76`.
It records exact stock/private payloads for all five same-phase pairs
(435 payload pairs), while preserving the original thread/mapped-library
failures. It is a partial CPU diagnosis, not an accepted bridge certificate.
The unchanged ten-boundary metadata archive has SHA256
`b26c19866e58c930b24c73996783ad7700d95e68ca0804a7538a9eb0b7c4a3f7`;
its member hashes identify the boundaries used for the following diagnosis.
These raw materials remain outside the repository.

Within either arm, `baseline_repeat`, `measured` and `off_after` change
respectively 64, 62 and 64 of the 65 effective physical page IDs. The last
15 unused table entries do not change. The actual last-token slots are
1008, 1040, 1008, 1040 and 1008 across the five phases. These are real page
relocations, not raw-identical input replays or unused-tail noise. The query,
immediate outputs and complete logical K/V hashes are unchanged in the
metadata; the partial numeric report separately compares their saved bytes.

The new reviewer retains `complete_first_input_signature_equal` and
`raw_input_exact`; `exact` continues to mean raw signature plus query,
immediate outputs and all 87 complete payloads are exact. Cross-arm
same-phase comparisons always require that strict result, without any page
normalization. A logical within-arm result cannot satisfy a cross-arm pair.

Only within an arm, `logical_input_equivalent` and `normal_repeat_valid`
provide a separate, narrowly scoped judgment. Each phase must independently
rehash the complete 80-entry int32 table, join its first 65 entries to the
observed used pages, prove those pages unique and within both actual caches,
and bind the actual writer/consumer slot to the page containing position
1024. The recorded 3D API K/V views must share the actual 4D cache storage,
offset, stride, dtype and capacity, including the writer's retained cache
objects. Both full logical K/V checkpoints must match, and the existing raw
reader must rehash their complete matching later files. Before-API K/V is
still hash-only; this change does not claim preserved before-API bins or
prove the absence of intervening writes.

The first-boundary snapshot's `consumer_returned=false` is its actual
pre-call state. Completion is joined separately to the final independent
audit row for the same module objects, arguments, caches, page map, slots,
request and writer. A missing completion or a different audit boundary
rejects the proof; the original pre-call flag is retained in the report.

After those checks, only the first 65 table values and their corresponding
full-table hash may differ in the comparison. The unused tail, mask, query,
scalar/default semantics, tensor geometry and complete logical K/V signatures
remain exact. The report keeps both table hashes, page lists, changed-page
count and actual slots. Missing evidence, unknown actual inputs, duplicate or
out-of-range pages, wrong slots or storage views, and different logical K/V
fail. Exact output alone never substitutes for the input proof. Query,
immediate outputs and all 87 payloads must also be exact for a normal repeat.

CPU contracts exercise those refusals with actual synthetic files, including
equal immediate outputs with a changed logical K/V byte, a missing later KV
file, changed unused table entries and a legal relocation rejected by the
strict cross-arm comparator. No Native mechanism, production capture, old
result, library-identity gate or F authorization changes in this increment.
The source83 thread and mapped-library gaps remain; a fresh complete pair
must use the already reviewed dual-thread capture and actual production
same-byte library mappings. C1/C2/C3 and E5 remain ungranted.
