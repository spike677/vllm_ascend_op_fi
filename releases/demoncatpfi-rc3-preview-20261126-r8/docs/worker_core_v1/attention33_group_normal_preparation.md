# Separate group package: normal preparation only

This increment builds `tnd_page16_arch22_group_s2_256` in the separate vendor
`dcat_f33_arch22_group_s2_transformer`. It preserves the reviewed original
SplitCore, workspace, serializer, and kernel arithmetic. The host candidate can
select S2 256 only with the distinct fault phase, current observer scope,
owner/thread/target fields, and one-use admission fields. **The adapter and the
two cases delivered here only run normal S2 512 and OFF. They cannot issue a
fault admission.** No C1, C2, C3, or E5 result is inherited.

The source105 single-site function/OFF result remains in its accepted original
scope. The source108 private normal run was externally terminated before its
last completed phase; it is neither a normal prerequisite nor a cost permit.

`GroupNormalWorkerSession` discovers the same 28 actual Attention objects and
retains all 56 prefill/decode calls per request. It uses the opt-in owned runtime
index, keeps the full original log and every binary, and records prefix/cost
receipts. All 28 current Q/valid logical KV/API output/consumer bindings and the
complete 87 model payloads remain distinct original files for every phase.

Only this new group's result uses `finish.capture_artifact`: exact path, SHA,
size, and compact payload identities. The full capture remains at
`<phase>/runtime_result.json`. The reader requires all five completed phases,
at most 128 MiB per capture and 640 MiB total JSON captures per arm, and rejects
an incomplete phase with its named path. Old all-normal result shapes remain
unchanged. Per-phase monotonic time includes reset, setup, real request,
capture, cleanup and its result checkpoint; initialization/warm request and
the final ordinary health request are timed separately.

Root adds the explicit controller mapping `arch22-group-s2` to
`apply_tnd33_arch22_group_s2.py`. CPU/SDK build command (root only):

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/build_tnd33_124.py" \
  --source "$SOURCE" --out "$BUILD_ATTEMPT" --interfaces "$SDK_INTERFACES" \
  --strategy arch22-group-s2
```

The recipe pins the same pristine official source closure, emits the independent
vendor/header/patch receipt, and builds a new public ABI schema and complete
SplitCore CPU matrix. Root must preserve the separate group's branches when
merging `fia/build_fresh.py` and `fia/binary_registration.py` alongside 28 work.

Prepare two normal-only cases after the final package and production CPU checks:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/prepare_arch22_group_normal_cases.py" \
  --source "$SOURCE" --source-manifest-sha256 "$SOURCE_SHA" --installed "$INSTALLED" \
  --build-root "$BUILD_ATTEMPT/build/run" --observer "$OBSERVER" --model "$MODEL" \
  --card 5 --timeout 1800 --out "$NEW_CASES_JSON"
```

The independent `review_arch22_group_normal.py` takes the same source/build/
observer identities plus `--stock-run`, `--stock-sha256`, `--baseline-run`,
`--baseline-sha256`, and fresh `--out`. It rechecks all 140 target pairs, all 435
payload pairs, original Native/ELF/POD evidence, and cleanup. A saved report
alone is never admission. Group C2 wiring, its time/log-capacity gate, and its
independent final reader are a subsequent increment; C3 remains design only.
