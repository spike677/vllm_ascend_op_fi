# Source111: FIA input journal SDK/CPU review

The new default-off, read-only FIA journal compiled with the actual CANN
8.5.1 SDK on124 in the pinned A2 production image. All38 Python/export
contracts and all26 production-translation-unit C++ contracts passed.
The controller used `runc`, no network and no NPU device mounts. It removed
only its own container after exit0, with `OOMKilled=false`.

This proves compilation and the bounded CPU contracts. It does not prove
actual NPU function ownership, a natural model input join, normal numerical
equivalence, Slice/copy lineage, C1/C2/C3, or statistics E5. None is granted.
No device computation or model inference was performed by this build.

## What was independently checked

The C++ fixture includes the production journal TU and the exact SDK types;
only original numerical functions/getters and owner callbacks are CPU doubles.
The cases cover inactive pass-through, noncontiguous view metadata, getter
failure/exception, wrong owner/caller/token/thread, missing/reused containers,
executor pairing and budgets, original GWS/execute errors and exceptions,
uninitialized failure outputs, workspace bounds, and empty-window exhaustion.
Successful fake execution is not reported as a real device call.

Independent source review found and repaired four issues before this build:

- Missing or multiple K/V members must remain in the public-role denominator.
- A public window without GWS records must not be labeled fully observed.
- Partial installation and partial release need staged cleanup with actual
  return codes and state readback. Unknown pending resources remain owned;
  a release retry continues only the unfinished component.
- Empty public windows also consume a bounded budget. Defaults allow64
  windows, enough for the planned56 calls, without enlarging the16 MiB event
  limit or discarding earlier records.

The new FIA GWS/execute wrappers preserve their original exceptions. The
unchanged reused four-container TU converts original C++ exceptions to
null/status161001; that historical boundary is disclosed, not described as
exception-transparent. No shared Runtime or writer callback was changed.

Root then rehashed all35 build artifacts, all17 recorded source dependencies,
the executed C++ binary and every one of the26 case logs. Exact case output,
exit status, source/profile identity and the separate controller originals
matched. This second read did not load the DSO or modify any original.

## Immutable identities

Originals remain at
`/home/17-fault-c2c3-20261117/fia-input-journal-source111-a1`.

| Item | SHA256 or commit |
| --- | --- |
| Frozen source commit | `0d9eab9d860d04bfcefd2404a2e8e5b75fe6dec9` |
| Source manifest | `12b37b01f130733844878bf69af8c922e02392ccc407733c6e3f06a535b01143` |
| Image | `35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0` |
| Controller | `4812e45e69006ef2c967eee2b942e90af17e4167b9db52c3f95e4ddee8a0ed5e` |
| Build manifest | `96abfb8f3de35c1a19a00d9db35bbd62bdb3a43a7e635bafede89f6ba508a4b0` |
| CPU receipt | `724d3128ee8959a73373b96bbd560e75ea7b28ebef2b837469c56c7d0db50968` |
| Canonical profile | `33102edbd8c3a8b98f410123a67e07ba0655431b2cfdf215b8927e085801a31c` |
| Executed CPU binary | `9c7c4e510e6abea196ccb37fcb646ad7547f192781d81f1f1d2a62f3d18e7c19` |
| Compiled journal DSO | `def7285eb9346b2cd6c4500065dba6a635692e1e7a1d431064dbc86c62e6c41d` |
| Observer DSO | `bcad8a17191db79b1ab0c89a110ff557b3a3b4243ff8d8f2fe353beeeb3a908b` |
| Python/export process log | `d7dfcf0308081c34dabf85eb534e50ae61326cb0f2d5cc0e550e8a9a75432721` |
| Independent original-file review | `2d5d709d2e1b6f8b3ad4b8b4e93b83de20058603c7f28817db558bb897b52ac0` |

Next is an explicitly scoped, zero-fault natural input observation with
pre-LLM resolution, the actual Worker/model objects, and the existing launch
token. Public/GWS geometry may be unequal or missing; those outcomes must
remain visible. Even equality cannot establish internal prepared-tensor or
Slice operands. Source111 does not authorize the separate Fault33 C2 candidate.

Publication contains implementation, CPU tests and this summary only; no new
raw statistics, private tensors, SDK headers or compiled binaries are included.
