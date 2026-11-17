# Explicit scope for stale checkpoint review

This incremental correction makes live stale-checkpoint review opt-in. The
earlier read-only evidence remains historical evidence; it does not authorize
another attempt or another Pod. `idle_gate(cards)` now preserves retained
checkpoint allocations without querying the live exemption path.

The root-owned queue accepts an explicitly approved JSON file with exactly
these four fields (values below are examples, not an approval):

```json
{
  "schema": "demoncat.stale_allocation_scope.v1",
  "attempt_scope": "/home/example/exact-fresh-attempt",
  "pod_uid": "11111111-2222-3333-4444-555555555555",
  "checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
}
```

The shared helper exports
`validate_stale_allocation_scope(scope, attempt_scope) -> dict`. It requires
exact keys/schema, a UUID-shaped lowercase Pod UID, a lowercase SHA-256, and
a normalized absolute POSIX attempt directory equal to the actual caller's
attempt directory. Unknown fields, relative/root/traversing paths, control
characters and mismatch are rejected. It returns an independent dictionary.

The gate signature is:

```python
idle_gate(cards, *, review_stale=False,
          stale_allocation_scope=None, attempt_scope=None)
```

An approved queue invocation passes the same validated scope and actual
absolute attempt path to every pre/post gate:

```python
scope = validate_stale_allocation_scope(document, actual_attempt_path)
gate = idle_gate(cards, review_stale=True,
                 stale_allocation_scope=scope,
                 attempt_scope=actual_attempt_path)
```

The root queue is responsible for reading the explicitly requested bounded
scope file without following a symlink, recording its original bytes/hash,
and exclusively creating the fresh attempt directory. The gate does not
infer authorization from a Pod disappearing or from a successful earlier
review. The queue wiring is a separate root-owned change.

The shared reviewer independently revalidates the scope and requires **every**
original checkpoint blocker to match its one approved UID and checkpoint SHA
before any live API/socket query. Any additional unapproved owner or SHA keeps
the whole set blocking. The canonical four-field scope and its SHA-256 are
included in the successful no-current-owner proof; its evidence digest covers
both. The internal final gate carries the same scope/path while suppressing
recursive review. All existing live identity, complete-set, runtime, final
gate and unknown-state constraints remain in force.

CPU validation passed all 52 contracts:
`python -B -m unittest test_device_allocation_gate test_kubelet_live_allocation_gate`.
The added tests cover default denial, boolean-only denial, absent/unknown or
malformed scopes, another attempt, another UID/checkpoint SHA, an additional
unapproved owner, proof binding and unchanged final-gate context. No remote
host, container or NPU operation was performed for this correction. No older
receipt was relabeled as carrying the new explicit scope.
