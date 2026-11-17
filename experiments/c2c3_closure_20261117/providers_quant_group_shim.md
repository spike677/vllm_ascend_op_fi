# Quant group typed shim: SDK build handoff

This increment implements the independent group's typed DynamicQuantV2 shim
and control ABI. Legacy C1 source/DSO is unchanged. Production SDK compilation,
real numerical calibration, Python loader isolation and Serving/Coverage
integration are not inferred from the host CPU tests.

`batch_scale_group_runtime.cpp` records each public GWS entry as a model parent
before admission, and records the actual helper/effect GWS and execute arguments
immediately before each real typed function call. Optional smooth/group/offset
descriptors, destination dtype, real Q/scale storage, workspace, executor,
registration, original-call counts and status are retained in the independent
Journal. Helper calls through the resolved function pointer have their own raw
events and never masquerade as a single model execution. The only exported
operator variant is V2. Raw `preparing`, `pending`, `executing`, overflow,
unclassified calls and errors remain visible.

The shim serializes its state and callbacks with a recursive mutex. Actual
thread/device are read from the runtime. Entry overlap is counted before the
mutex; a competing public call is rejected rather than silently treated as
part of the selected call. The group also refuses a mapped DSO exposing legacy
`dcat_dq_arm`/`dcat_dq_finish`, including locally loaded libraries, at install,
public scope, admission and real public Native boundaries. Managed Python
check/load/arm exclusion is a separate required increment, not completed by
this Native scan.

## Storage lifetime

The public scope records an input association and the actual GWS executor
handshake. It does not grant ownership of a raw pointer. The future managed
Python scope must strongly retain the input and call the reviewed actual
public operation. The actual Native execute stack is the only window during
which the shim can read/restore the caller's public Q/scale under the caller's
allocation-lifetime contract. Staged helper/effect outputs and workspace are
owned by the group.

Only an internal witness tied to the active execute stack, matching public
scope, thread/device and actual parent is usable for recovery. There is no
export accepting an arbitrary `LiveWitness` or a user-supplied lifetime token
to restore memory. Success is synchronized, compared, published, exactly read
back and acknowledged before leaving that Native stack. If a failure occurs,
the shim attempts all-target stop, all-target/child drain, exact rollback and
release while the original call is still active. A prepared executor without
an observed successful execute cannot pass that barrier and is never forgotten.
A real execute returning nonzero retains its raw pending registration; later
stream synchronization cannot prove that the executor was consumed. The CPU
mock now retains the executor on that failure instead of inventing a release.

Native, admission and completion stack flags use RAII, including exceptional
unwinding. Every exported C ABI catches C++ exceptions and poisons the runtime
without attempting a late memory recovery. Even before the public envelope
ends, a returned Native execute cannot leave a usable stack witness. Newly
created descriptors are tracked in fixed slots before a potentially allocating
core ownership transfer, so a vector allocation exception does not lose them.

The CANN 8.5 getter arrays are caller-owned and must be deleted after use;
`view()` now releases shape, strides and storage shape with `delete[]` on every
success, error and exception exit. This follows the official
[aclGetViewShape](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/aolapi/operatorlist_00038.html)
and [aclGetViewStrides](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/aolapi/operatorlist_00039.html)
contracts. The mock allocates real arrays and counts allocation/deallocation;
it no longer returns descriptor-owned arrays. Earlier C1/20/39 sources and
receipts are not rewritten by this correction; their audit is separate.

On public return, the scope is revoked. A retained transaction is quarantined;
external recover refuses to read or write the former input/Q/scale. A later
successful stream sync does not revive those allocations. This intentionally
cannot claim same-Worker OFF after an unrecoverable returned-call failure.
Unknown live storage is never reconstructed from a matching address/shape.
Native byte rollback is not a healthy OFF output or E5.

No public output Tensor is claimed to have a Python strong owner before the
real operation returns. The minimal binding and subsequent provider must
validate the actual returned Q/scale identities, preserve callbacks/resources
on failure, and retain the original live C2 proof at the shared API boundary.

## CPU evidence and root-only SDK build

Host attempt `providers-quant-group-shim-cpu-a1` failed `-Werror` for an unused
test-fixture variable and is preserved. Attempt `providers-quant-group-shim-cpu-a2`
passed 9 typed-shim memory contracts and the existing 19 core contracts with
`g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic`.
After the executor/exception/metadata fixes, fresh host attempt
`providers-quant-group-shim-cpu-a3` passed 14 typed-shim contracts and the
updated 19 core contracts. Added cases cover execute-error ownership despite
successful sync, exceptions at start/completion and a control ABI, and metadata
arrays on success, partial getter error, rejected rank and exception exits.

Those host tests explicitly use `tests/native_quant_group_mock_sdk` and injected
CPU memory functions inside the test translation unit. They prove control-flow
contracts, not real CANN ABI, actual allocator reuse, or quantization rounding.
The production DSO contains no API for injecting those mocks. Raw receipts
remain under `/home/17-fault-c2c3-20261117/`; no NPU or container was started by
the provider agent.

Root's actual SDK attempt `native-quant-group-v1-a1` failed in the vendor
`acl_dump.h` zero-length `dataBuf[0]` declaration under `-pedantic`. Its receipt
SHA256 is `43f422d3758258ecad211e7421087ba29d6fdc004c53ed28412d75bd9680b61d`;
the log SHA256 is `d6bb01d9169919e13e4d6ef31ddb7c6df92b024243480f1e56192d7a44a7e026`.
That failed attempt is preserved. The builder now marks only the actual vendor
SDK include directories as `-isystem`; all four strict warning flags remain
enabled for our source. Official headers are unchanged. Retry uses a new
output directory rather than rewriting attempt a1.

Root's actual SDK attempt `native-quant-group-v1-a2` passed for the earlier
source revision. Its manifest SHA256 is
`5004df441af3f542f2ac241f01825448bd0b7432c10c9157854ea3fc1922bd01`.
It does not validate the subsequent three ownership fixes. A fresh a3 bundle
with a new source/build fingerprint is required before managed use.

Root should run the following only after reviewing the commit and creating a
fresh immutable source snapshot. This builds in the actual reviewed CANN 8.5.1
image with its real headers, no device/driver mount, `runc`, no network, 4 CPUs,
4 GiB and a 300-second bound. It records the inspected immutable image ID,
actual SDK/opapi/header hashes, source hashes, exported symbols, ABI sizes and
CPU contract results. Timeout/error receipts are retained and only the unique
container with the matching name/owner label/image can be removed.

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_build_group.py" \
  --source "$SOURCE" --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --out /home/17-fault-c2c3-20261117/native-quant-group-v1-a3
```

The bundle's `build.json` and its SHA are required inputs to the forthcoming
verified Python binding. `native_qualified_targets=0`, `c2=false`, `c3=false`
remain explicit even after a successful SDK build. Unknown Attention/MoE row
provenance and all unresolved projection candidates still block the whole
natural group; no model mechanism or denominator is narrowed here.
