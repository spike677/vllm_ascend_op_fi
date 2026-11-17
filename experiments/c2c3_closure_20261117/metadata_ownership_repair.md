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

The authoring Windows host has no C++ compiler. Local source generation succeeds
and reports `NOT_RUN_COMPILER_UNAVAILABLE`; this is not a passing CPU or SDK
build. The root owner should run, without a device or container requirement:

```sh
python3 -B "$SOURCE/fault39/native/test_metadata_ownership.py" \
  --output "$NEW_ATTEMPT/metadata_array_cpu_receipt.json"
```

Only after those CPU contracts, fresh exact-SDK proxy builds, loaded-image
identity checks and same-Worker before/after metadata counter readbacks pass
may the root rerun the full natural B/L/F/OFF and live C3 sequence. This commit
does not grant either mechanism new C2/C3 status. The following Python/runtime
integration must enforce the new counter gate; source ownership alone is not
a complete serving acceptance result.
