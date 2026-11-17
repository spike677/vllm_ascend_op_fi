# Fault26: explicit split dependency startup and retained mapping diagnostics

The repaired QMM attempt stopped during Worker attachment, before any group
phase or fault injection. The original result remains failed:

`/home/17-fault-c2c3-20261117/native26-repaired-groups-card2-a1/providers26-v2-qmm-deepseek-coverage/result.json`

SHA256: `8e70dba60c09058a99ead9e1c3ed4e125b57c68aa4c687267f0392a26d81001c`.
It has healthy-before token `[185]`, `phases=[]`, zero qualified targets and
`METADATA_ARRAYS_INCOMPLETE_WRITER_PROXY_MAPPING`. The FIA case did not run.

Read-only inspection verified the actual queue command, immutable bundle and
ELF dynamic metadata. The command preloads the v2 shim, trace, FIA proxy, main
opapi and custom opapi. It bind-mounts `libopapi_nn.so` at the SDK path but does
not explicitly load that split DSO. The actual v2 shim's RTLD_NOLOAD lookup
had succeeded; the native trace has its actual resolution record. The trace
file SHA256 is
`100b08e007ff4f6a41ddaffc65af9090f0af972d08fd1fb1dea3cb09d1e1f0dc`.

Actual bundle `native26-metadata-group-a1` has manifest SHA256
`382b849b95a070cbfa64b22794696023784815d20f49865f7a37963aaaddf67c`:

| Image | SHA256 |
| --- | --- |
| `libdcat26_group_v2.so` | `9746decca6570f8e9a6944972a89c991ace89773759655e11bf0aac75773fd4b` |
| `libfault26_readonly_trace_abi0.so` | `df36a932fe10e25818e2b485fef74eaf04724e29712e3a93b01cd4497f3aabc7` |
| `libopapi.so` | `e480bf9b55fcb510ef6b43eb7c36f1d2ef01383ec3b71f0aff608e3e1504e430` |
| `libopapi_nn.so` | `1babe2a881124740e2b4f217024902d8149560444a84b3d21b64e4d6f26090d3` |

The old error did not retain `/proc/self/maps` or the missing-name set. Thus
the split is the source/launch-supported suspect, **not a retrospectively
observed missing-image identity**. New errors retain expected, missing and
actual mapped names/paths. A no-device preflight must confirm the exact cause
before another long model run. No old evidence is rewritten.

## Deployment change

`fault_injection/deployments/writer26_loader.py::StartupDependencies` is an
explicit startup operation, separate from the read-only metadata observer.
Before importing Torch/vLLM it verifies the immutable group manifest, all
artifact bytes, the three required preloaded images, their inode/bytes and
absence of legacy writers. Only the expected split dependency may be absent.
If so it loads that exact manifest file with `RTLD_NOW|RTLD_LOCAL` and retains
the handle. It then runs the unchanged four-image observer, including independent
counter instances, SHA/inode/dladdr identity, zero active/live/error counts and
equal acquisition/release totals. Wrong bytes, another missing image, foreign
preloaded copies, late initialization or any failed observer check still refuse.

The existing global LD_PRELOAD order is unchanged. The main/split proxy uses
RTLD_NEXT forwarding, so adding the split to the global preload chain would
require separate routing analysis; the explicit local load avoids that change.
The observer itself still opens only already-loaded images via RTLD_NOLOAD.
An unused split counter may stay zero, but the library and counter are mandatory.

The full `providers26_group_probe.py` performs this startup before importing
vLLM and stores its receipt. On failure, the report retains startup diagnostics
without constructing a model. Model qualification, real hits, same-Worker OFF
and current workload metadata activity remain separate mandatory gates.

## Root-owned no-device verification

First run the actual host ELF fixture:

```sh
python3 -B "$SOURCE/tests/native/test_metadata_arrays_elf.py" --writer26-startup
```

It starts with only three counter images, verifies the missing split is named,
performs the explicit local bootstrap, checks four independent instances and
that a unique split symbol has not become globally visible, and still refuses
an identical-byte foreign image. The separate `--writer26` mode retains the
SDK-alias test. Neither fixture loads CANN or uses NPU devices.

Next use the exact original image and bundle in a root-owned CPU container:
no device mounts, `--runtime=runc`, `--network=none`,
`ASCEND_VISIBLE_DEVICES=void`, and an explicit shell entrypoint. Mount the fresh
installed package at `/installed`, the immutable bundle at `/native`, and a
new evidence directory writable. Preserve the original SDK split bind mount,
set-env, custom OPP path and original five-library LD_PRELOAD/LD_LIBRARY_PATH.
Do not mount model weights or start a Worker. In two fresh processes run:

```sh
python3 -B -m fault_injection.deployments.writer26_loader \
  --bundle /native --manifest-sha256 ACTUAL_MANIFEST_SHA \
  --observe-only --out /evidence/observe-only.json

python3 -B -m fault_injection.deployments.writer26_loader \
  --bundle /native --manifest-sha256 ACTUAL_MANIFEST_SHA \
  --out /evidence/explicit-local-startup.json
```

The first preserves any actual missing-image refusal (exit 2); inspect it rather
than assuming the split is absent. The second must show all four exact loaded
images/instances and quiescent counters. It issues no operator, device, model
or writer-callback calls. Status `CPU_LOADED_FOUR_IMAGE_CONTRACT_PASS_NOT_NATIVE_ACCEPTANCE`
does not grant C1/C2/C3. Only after actual loader acceptance should root schedule
fresh natural QMM/FIA B/F/OFF and coverage attempts.

Local mock contracts cover successful explicit local loading, already-loaded
split, another missing dependency, altered artifact bytes, foreign existing
copies, late initialization, observer failure and complete missing-map diagnostics.
The Windows authoring host cannot run the real Linux ELF or actual SDK loader
checks; those remain root-owned. New raw statistics are not published here.
