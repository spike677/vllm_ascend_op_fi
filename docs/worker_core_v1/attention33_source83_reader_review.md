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
