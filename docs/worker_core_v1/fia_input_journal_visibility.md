# FIA journal loader visibility diagnostic

This additive probe investigates source113's **56 public windows / zero Native
journal calls**. It does not change the journal, its ABI, its runtime adapter,
the deployment preparer, any operator, or any tensor. It grants no input
lineage, C1/C2/C3, fault execution, or E5 authority.

## What the existing evidence establishes

The local source113 archive is
`fia-input-journal-source113-a1-json-originals.tar.gz`, SHA-256
`483a0cb7c91d04bbaf10807ee615aa2c3fd2b08873526d014cf2a811b5e53573`.
Its pre-LLM original-function receipt reports successful V3 resolution in
`libopapi.so` SHA-256
`b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c`:
prepare offset `30251552`, execute offset `30252216`.
The final retained journal state has `resolver_ready=1`, 56 accepted windows,
zero total/outside-scope calls, zero events and zero prepares/executes.
Therefore this attempt did not suffer the earlier hypothesized missing
pre-LLM SDK dependency. Window entry is not evidence that a typed wrapper ran.

There is a concrete deployment difference worth testing. The successful
quant container/GMM2 setup used a private **named `libcust_opapi.so`** and
prepended its vendor directory to `ASCEND_CUSTOM_OPP_PATH`. The Fault26 setup
also supplied a named `libopapi.so` containing the V3 exports. In contrast,
the new FIA journal is an independently named `libfia_input_journal.so` in
`LD_PRELOAD`; its dependencies are the original SDK/custom images.

Sources for this comparison are `prepare_quant_container_case.py:21`,
`fault26_readonly_trace_v1/build.sh:96`,
`providers26_prepare_group_cases.py:86`,
`natural_attention28_input_journal_probe.py:318`, and
`fia_input_journal_build.py:95`. The inspected vLLM custom resolver source
`csrc/aclnn_torch_adapter/op_api_common.h`, SHA-256
`142023f0f0e682b3b79aeb987044ecbd9b7db0b63b20de967447ba6f9e3d6b16`,
uses `dlsym` on a custom/main library handle. That source alone does **not**
establish which resolver or cached address the installed torch_npu FIA call
actually used. Neither handle bypass nor a V4 alternative is proven.

## Exact scope of the new probe

`diagnose_fia_journal_visibility.py` is standard-library-only before an
explicit `import torch_npu`. It captures the following six names through
`RTLD_DEFAULT` and existing named/path handles, once before import and once
after import:

* `aclnnFusedInferAttentionScoreV3GetWorkspaceSize`
* `aclnnFusedInferAttentionScoreV3`
* `aclCreateTensorList` / `aclDestroyTensorList`
* `aclCreateIntArray` / `aclDestroyIntArray`

All non-default handles use `RTLD_NOLOAD | RTLD_LAZY`. Missing handles and
missing symbols remain explicit entries; the probe never attempts an ordinary
`dlopen` fallback. The candidate names are a visibility matrix, **not an
asserted resolver priority or selected-function trace**. Existing mapped
aliases of those exact names are included, with a hard 96-handle bound.
It does not install a journal, invoke a looked-up symbol, construct a Worker,
or make an explicit device query or operator request.

The old source111 profile canonical digest is fixed in the probe:
`33102edbd8c3a8b98f410123a67e07ba0655431b2cfdf215b8927e085801a31c`.
The supplied build SHA binds every bundle file, including the journal DSO.
All 17 compile-source bytes are checked against that profile and a separately
SHA-bound frozen source manifest. Original SDK/observer/custom/torch_npu files
must still match their profile hashes and sizes. The diagnostic source and
the exact installed `torch_npu/__init__.py` each have a required SHA argument.
The latter is checked before and after the explicit import.
`--native-source-root` may also point to source113 when all these same 17 files
still match the source111 profile and its supplied manifest hash. This checks
unchanged compile-source bytes; it does not relabel source113 as the original
Native build. The separately mounted new diagnostic needs only its own SHA.

Each non-null lookup records the actual `dladdr` path/base/symbol/offset,
canonical file identity and exact profile-role matches. Its already-loaded
owner is temporarily held with another NOLOAD reference while readable,
non-writable executable mappings are hashed. Their bytes must match the
on-disk ELF executable segments, and pinned owners' segment descriptions must
match the old profile. Raw mapping lines retain overlay device/inode details;
the probe does not reject every system mapping merely because overlay `stat`
has a different device number. It does not infer executable identity from a
basename. Unknown owners retain their actual identity rather than inheriting
a profile role. All temporary NOLOAD references are released and errors are
retained. Limits are 256 MiB per file, 128 MiB executable image, 512 MiB charged
executable-byte reads per phase, and 64 KiB read chunks. These are diagnostic
limits, not increases to a provider's runtime resource authorization.

The two immutable phase files contain every lookup/error and are SHA-referenced
by the result. A failed import retains the first phase and also inspects the
partially imported process, labelled as such. A missing required startup
handle refuses before import. The script does not repair an incomplete
startup by loading CANN later. Actual PID/thread/start identity must remain
unchanged across the two phases.
The profile's torch_npu/custom-extension files are hashed even when not mapped.
Only torch_npu is imported. A still-unmapped `custom_extension` is recorded as
not loaded and is not an error or an invitation to load vLLM or that DSO.

## Root-controlled execution

Use the pinned production image, `runc`, no network and **no device mounts**.
Mount the genuine required driver libraries read-only, the unchanged source111
journal bundle, its exact observer alias and its frozen compile source.
The external controller must record image ID, mounts, absence of devices,
command/environment, stdout/stderr, exit/cleanup and every result hash.
The script deliberately reports device isolation as
`REQUIRES_EXTERNAL_CONTROLLER_EVIDENCE`.

Loading SDK libraries and importing torch_npu can execute constructors or
driver-discovery code. Their warnings/errors must be preserved. This probe
does not claim zero driver function calls; it reports zero explicit operator,
device API, Worker and journal-install calls in its own source path.

After loading the exact CANN environment, a root-reviewed command is:

```sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_PRELOAD=/observer/libdcat_attention_observer.so:/journal/libfia_input_journal.so
python3 /probe/diagnose_fia_journal_visibility.py \
  --journal /journal \
  --build-sha256 96abfb8f3de35c1a19a00d9db35bbd62bdb3a43a7e635bafede89f6ba508a4b0 \
  --native-source-root /native-source111 \
  --native-source-manifest-sha256 "$REVIEWED_SOURCE111_MANIFEST_SHA256" \
  --probe-sha256 "$REVIEWED_PROBE_SHA256" \
  --torch-npu-init-sha256 "$REVIEWED_INSTALLED_INIT_SHA256" \
  --out /evidence/fia-loader-visibility
```

Use a fresh evidence directory. The first two preload objects must be exactly
observer then journal; all existing SDK dependencies remain the journal's
unchanged startup dependencies. The root supplies actual SHA values, not
placeholders, before execution. No new Native build is needed for this probe.
It must not be inserted into an ongoing Worker or existing NPU attempt.

## Interpretation and stop conditions

If default lookup reaches the journal while a named handle reaches original
opapi/nnopbase, that proves a **visibility difference only**. It does not prove
the model called that handle, used V3, or retained any particular cached
pointer. Missing optional handles/symbols remain complete negative lookup
observations; an owner whose actual bytes cannot be established, import
failures, byte changes, gaps, bounds or unbalanced NOLOAD references make the
diagnostic incomplete. A fully recorded owner outside the pinned profile
remains explicitly unmatched and grants no authority. Do not retry
an inference or enable F merely because a visibility matrix is complete.

If visibility supports the named-handle hypothesis, the minimal next
candidate is to assess reuse of the already-tested quant custom-op proxy
deployment: expose only the existing V3/container journal functions through
the appropriate named handle, retain the original DSO's identity/SONAME and
its exact one-call forwarding, and keep all tensors and arguments unchanged.
That would be a **separate reviewed deployment/build**, not a modification
performed here. Actual cached-function ownership and natural-call coverage
would still need evidence. If current typed selection cannot be established,
retain UNKNOWN; do not add V4 exports or guess an ABI.

## CPU checks and limits of validation

```text
PYTHONPATH=src;fault39/src;.;tests
python -B -m unittest test_fia_journal_visibility -v
```

These local fixtures check byte/source drift, exact six-slot/missing-handle
reporting, bounded mapped-code reads, NOLOAD-only lookup, reference release,
partial import, startup refusal and immutable phase receipts. They use CPU
ELF/memory/loader doubles; they are not a real Linux SDK loading result and do
not upgrade the source113 56/0 result. Root must execute the device-free
production-image command before interpreting actual lookup addresses.
