# Safe cancellation of an empty installed quant owner

`dcat_dqg_cancel_empty_v1` closes an owner that has admitted no targets. It
requires the actual owner/thread/device, no active public/Native/callback
scope, no admitted transaction/target, and no ambiguous raw executor, orphan
descriptor, exception or quarantine. It stops admission before checking drain.
Every actual successful passthrough execute stream observed since this owner
was installed must synchronize; raw pending/executing counts of zero alone do
not establish drain. A failed synchronization leaves the owner stopped and
installed so cancellation can be retried. Target recovery uses the existing
whole-group stop/drain/recover path and cannot use this shortcut.

Cancellation does not remove callbacks. The existing `remove` export still
requires the exact retained callback identities and zero owned resources.
Python keeps the owner/callback strong references after a pending cancellation
and until removal succeeds. No cancellation path reads or restores former
public inputs/outputs. Observer events, counters, errors and generation history
are preserved. A new generation does not resynchronize stale streams from a
previous owner.

The host-only C++ attempt
`/home/17-fault-c2c3-20261117/providers-quant-group-shim-cpu-a4/receipt.json`
passed 21 core contracts and 17 typed-shim contracts with strict warnings.
The tests use explicit CPU/mock SDK functions. Added cases exercise pending
prepare/executor, failed stream then retry, no new admission after stop, wrong
owner, existing target, live public scope and exact callback removal. No
container or NPU was started. Python binding tests additionally verify retained
callbacks across cancellation and independent removal.

The new export and source hashes require a fresh reviewed SDK bundle. The
earlier a3 bundle must not be used with the updated loader. Root build command:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_build_group.py" \
  --source "$SOURCE" --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --out /home/17-fault-c2c3-20261117/native-quant-group-v1-a4
```

Use that new bundle and its actual approved `build.json` digest for the
device-free load probe and later serving inventory. Cancellation is a lifecycle
contract; it confers no Native target, row provenance, C2 or C3 qualification.
