# Worker Core source integration bundle

`tools/export_integration_bundle.py` exports a source-only handoff from an
explicitly selected reviewed source tree. It preserves the three existing
Python distributions and the existing NativeBuildManager, HostContracts,
WorkerFaultRuntime and provider implementations. No implementation is rewritten
or imported during packaging. The exporter invokes no compiler, native loader,
serving lifecycle operation, remote host or device query.

## Freeze, export and verify

Run these commands with Python 3.10+ from the reviewed tree. Replace the source
root, output folder and review reference with the actual reviewed snapshot.
The source root must contain the same exporter version used by the command.

```text
python -B tools/export_integration_bundle.py plan --source-root REVIEWED_ROOT --review-id COMMIT_OR_REVIEW_REFERENCE --output OUTPUT/review.json
python -B tools/export_integration_bundle.py export --source-root REVIEWED_ROOT --review OUTPUT/review.json --output OUTPUT/demoncatpfi-integration.tar.gz
python -B tools/export_integration_bundle.py verify --archive OUTPUT/demoncatpfi-integration.tar.gz --review OUTPUT/review.json
```

`plan` records exact candidate paths, byte counts and SHA256 values. The review
reference is supplied by the caller; creating the file does not establish code
review or runtime qualification. Review the resulting source list before export.
`export` refuses changed bytes, added/removed selected source files, linked
sources, unsupported file types and existing output artifacts. Every source
byte is preserved; even shell script line endings are not rewritten. New
reviewed Worker discovery/census modules and `test_core_*.py` contracts are
included when present in the selected root. Packaging does not require a Git
history scan or execute a referenced module.

Source policy revision 2 includes the exact existing observer closure:
`engineering_closure/__init__.py`, `path_discovery.py`, and `native_observer.py`,
plus `tests/worker_extension/test_serving_context.py`. If the selected snapshot
contains `call_census.py`, its current Worker Core README and census contract
are required too. Other engineering history, queues, execution receipts and
reports remain excluded. This packaging-policy revision does not implement
Worker Coverage V2. Verify older policy-1 artifacts with their original bundled
exporter rather than interpreting them as a complete census distribution.
The exporter also refuses a census snapshot unless its unchanged root
pyproject declares `engineering_closure*` in package discovery; copying its
observer source files alone is insufficient.

Keep `review.json` and the returned archive SHA256 separately from the archive.
Verification reads archive members without extraction and rejects traversal,
links, unexpected or duplicate entries, missing files, changed source bytes,
changed generated contracts and a changed source manifest. Hashes establish
integrity against supplied review bytes, not authorship or review approval.
Source-only archives are deterministic for the same reviewed snapshot.

## Layout and reuse

```text
demoncatpfi-integration/
    source_review.json
    source_manifest.json
    sources/                   # one original repository-relative source tree
    contracts/                 # HostContract/RPC/ABI entrypoints and source index
    worker_extension/          # current Worker source index
    providers/                 # existing Provider/deployment source index
    native_build/              # existing manager/recipe/build-input source index
    adapters/                  # existing adapter source index
    tests/                     # CPU contract index and commands
    integration_guide/         # deployment handoff, no serving lifecycle manager
    compatibility_matrix/      # separate acceptance states, no inherited pass
```

Each index points into `sources/` and records the original relative path and
SHA256. The implementation is stored once. The original distribution roots are:

| Distribution | Preserved build root |
| --- | --- |
| `demoncatpfi` | `sources/` |
| `demoncat-worker-extension` | `sources/src/` |
| `demoncat-fault39-core` | `sources/fault39/` |

Build or install these with the host's approved packaging tools. For an offline
CPU wheel check with build dependencies already installed, use
`python -m pip wheel --no-deps --no-build-isolation --wheel-dir WHEEL_OUTPUT ROOT`
once per root. The exporter itself ships no wheels or native binaries.
On Windows, keep both the extracted source path and the packaging subprocess's
TEMP/TMP path short. The desktop sandbox's long temporary prefix can exceed
Win32 path limits during setuptools or pip wheel copying even when archive
verification and Python imports succeed. This does not require source changes.

Native recipes retain their existing source-relative dependencies. Current
Fault20/26/39 build inputs in four legacy source directories are selected by
explicit filename; container scripts, analysis programs and historical reports
from those directories are excluded. When packages are installed elsewhere,
point the existing NativeBuildManager `--repository` at `sources/`. CANN files,
vendor operator source and environment-specific dependencies remain external.
Their absence must preserve the existing refusal, not trigger an alternative
unreviewed build or fallback provider.

Raw tensors, statistics, traces, evidence/report directories, history, model
weights, vendor libraries, binaries, bytecode and build outputs are excluded.
Only an explicit small set of source contract/schema JSON files is eligible;
an arbitrary JSON under a runtime directory is excluded. Historical native
strategy certificates are excluded with their evidence. Fresh explicit
certificates and bound evidence remain required for those arming paths.

## Acceptance boundary

The actual inner-framework Registry/Controller remains unavailable, so yellow-
zone integration is **PENDING**. This bundle offers the existing document
HostContracts and Worker Core HostAdapter for that team to implement against
their real serving system. It cannot certify that wiring.

CANN 8.5.1/A2 prior mechanism-specific results are not imported into bundle
acceptance. CANN 9.x and A3 remain **NOT_ACCEPTED_BY_THIS_BUNDLE**; a source
archive or successful wheel build does not establish that compatibility.
Runtime identity, build/ABI, actual call binding, C1, C2, C3, resource recovery,
healthy-output recovery and E5 require separate current evidence. Exporting
all source entries does not declare 17/17 fault acceptance.

`tests/test_integration_bundle.py` checks real source relocation, source drift,
explicit build dependencies, three unchanged distribution roots, archive
integrity, exclusions and deterministic generation. The exported host contracts
can then be run from the relocated tree using the instructions in
`tests/README.md` inside the bundle.

For a reviewed tree with census, run the complete relocated Core contract set:
`python -B -m unittest discover -s tests/worker_extension -p "test_core_*.py"`.
The three wheels must come from that same source review, including the root
distribution's `engineering_closure` package declaration. Verify imports from
installed wheel contents as well as the relocated source tree; source imports
alone cannot establish installed-package closure.

Recorded CPU validation on 2026-11-18:
17 exporter tests ran, 16 passed, and one source-symlink creation test was
skipped because this Windows host does not grant that operation. Archive-link
refusal was tested independently and passed. The current root snapshot
`11607cda4d30a402b4e69dc4318fd6ff927053a2` was copied read-only into a bounded
reviewed staging tree; only this exporter, its test and this guide came from
the agent overlay. Every other source hash was checked against the root tree.
Its 305-source-file / 319-member source-only bundle verified successfully.
All 67 relocated `test_core_*` contracts passed with no skips. All three
original distributions built wheels offline with
`PIP_NO_INDEX=1`, `--no-deps`, and `--no-build-isolation`; wheel inspection found
no native binary, tensor, raw statistics, bytecode, evidence or history payload.
The root wheel contained exactly the three existing observer modules, byte for
byte equal to the root snapshot. After installing all three wheels into a fresh
directory, the same 67 Core contracts passed again using installed runtime
imports, with no runtime source-tree fallback.
These are packaging and CPU contract results, with zero Native builds and zero
NPU calls. Source counts may change when the main reviewed tree adds discovery
or census modules; each final export must use its own source-review digest.
