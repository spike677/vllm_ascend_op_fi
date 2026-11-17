# Fault20 / native39 metadata array ownership repair

The earlier numerical effects, independent hit denominators and exact OFF
payloads remain historical evidence. Their resource counters did not include
the host arrays returned by `aclGetViewShape`, `aclGetStorageShape` and
`aclGetViewStrides`; those callers omitted the required `delete[]`.
The [CANN 8.5 getter contract](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/aolapi/operatorlist_00038.html)
states caller ownership and shows the three releases. Full-resource C2/C3
qualification therefore requires a new build and fresh in-process validation.
No historical receipt or raw statistics is rewritten by this repair.

The current source changes cover Fault20's `capture_x1_metadata`,
`capture_tensor_metadata`, and `log_tensor`, plus native39's shared
`capture_tensor_metadata`. `metadata_array_owner.h` owns every returned
non-null array immediately, including a getter that assigns an array before
returning an error or throwing. Destruction frees the arrays after success,
validation rejection, later-getter failure and exception unwind. The logging
path now returns on getter errors instead of dereferencing missing arrays.
Capture/log exceptions are contained and recorded; they cannot create success.

Each actual metadata proxy has a separate read-only counter instance. The
additive `dcat20_metadata_arrays_v1` or `fault39_metadata_arrays_v1` export
reports calls, active captures/getters, acquired/released/live arrays, errors
and a monotonic revision. There is no reset or caller-supplied release API.
Internal linkage prevents a main/split proxy from silently sharing counters
through ELF symbol interposition. These counters cover only SDK getter arrays,
so they supplement existing owned workspace/storage/descriptor evidence rather
than replacing it. Admission must inspect every loaded proxy bound by the
actual build manifest and reject missing/ambiguous exports, non-quiescent or
inconsistent state, errors, live arrays, or unequal acquired/released totals.

Both build recipes include a CANN-free test that inserts the exact production
capture and log function bodies into an allocating mock SDK. The mock returns
actual `new[]` arrays, while independent global allocation/deallocation tracking
checks the real `delete[]` calls. Cases cover all getter failure points before
and after allocation, exceptions, invalid dimensions, missing arrays, zero
element sizes, logging truncation and a throwing log sink. Query ABI failures
are also checked. The build profile fingerprints the shared ownership header,
test source and counter capability; existing bundles lack this capability.

The authoring Windows host has no C++ compiler. Local source generation reports
`NOT_RUN_COMPILER_UNAVAILABLE`. The root owner subsequently ran the exact
production-body fixture on the 124 host, without NPU calls: the first attempt
exposed strict `%p` vararg diagnostics; the production logging arguments were
fixed without weakening compiler warnings. The fresh `source-a38` attempt at
`/home/17-fault-c2c3-20261117/metadata-ownership-cpu-a2/receipt.json` reports
`CPU_CONTRACT_PASS`: 144 scenarios, 201 real `new[]`, 201 real `delete[]`, zero
live arrays. This CPU result is not an SDK or NPU acceptance. Reproduce with:

```sh
python3 -B "$SOURCE/fault39/native/test_metadata_ownership.py" \
  --output "$NEW_ATTEMPT/metadata_array_cpu_receipt.json"
```

Only after those CPU contracts, fresh exact-SDK proxy builds, loaded-image
identity checks and same-Worker before/after metadata counter readbacks pass
may the root rerun the full natural B/L/F/OFF and live C3 sequence. No change here
grants either mechanism new C2/C3 status.

The group runtimes now enforce `fault_injection/metadata_arrays.py`. It checks
the actual mapped inode/device, one unambiguous instance of every profiled
proxy, the self-hashed build profile, manifest proxy bytes and exact `dladdr`
origin of each counter export. Handles use `RTLD_NOLOAD`; no CANN image is
loaded by this observer. Old profiles, missing counters, mixed builds and
symbol resolution into a different DSO are rejected. Each request and the
final Worker health artifact retain raw counter readbacks. The implementation
digest is bound into the group coverage scope.

Before/after readbacks require zero active/live arrays, equal acquired/released
counts, zero getter/capture/ownership errors, stable independent counter
instances and monotonic counts. Final health additionally requires positive
metadata array activity during this actual workload; an unused split proxy
may remain zero. Missing metadata evidence cannot be converted into literal
zero host bytes in a Worker health proof. Counter tests include nonzero
leaks/errors, stale identities, missing APIs, wrong symbol providers, array
counter resets and no-activity group finalization. These Python tests are CPU
contracts only. Existing buffer, workspace, hook and lease checks remain
mandatory alongside these SDK-array checks.
