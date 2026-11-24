# Output39 actual direct-forward discovery

This is a source/CPU adaptation for a new public Worker attempt. It does not
accept source80, grant Native/C2/C3/E5 qualification, or rewrite its failure.
Source80 retained 372 structural objects: 28 supported, 341 unsupported and
three unobserved objects. They were the inner model, its ModuleList, and one
ApplyRotaryEmb object shared through 28 aliases. Those actual diagnostics stay
unchanged. The next attempt must rediscover its own complete denominator.

## Actual boundaries

`serving39_output_discovery.Output39ObservedForward` extends the existing
`PostwriteObservedForward` only for the public 39 window. It never calls a
model, runs an RPC forward, changes an operator, or installs an execute_model
hook. The original ModelLocal observer and Native algorithm remain unchanged.

Within the already executing root/logits request it owns one current-thread
Python profile callback. The hot path first filters by code identity. For a
held module's actual bound forward or CustomOp delegate, it checks the actual
frame's code, globals and self against the current bound function. Actual
function and wrapper closure bindings are checked on hits; source files are
read/compiled at installation and checked again at the complete boundary,
never hashed for every Python event. Only the existing pinned collector
unwrapper can reveal an original beneath an msprobe wrapper.

A direct inherited forward creates a real module entry. An entry already
created by a real Module hook or the root/logits seam is reused; the observer
does not synthesize hooks or double count that invocation. An unverified
direct callable adds no observation. Its object remains in the universe.
Known code with a foreign module/globals is refused. An inherited super body
on a held object is not misreported as another bound module invocation.

The profile return event with `None` is ambiguous between a normal return and
exception unwind; it cannot prove a completed forward. It refuses completion
instead of treating an always-call hook or finally block as a healthy return.
Frame records, gaps and failures are retained with frozen owner/generation,
PID/thread/model identities. They contain metadata, not tensor payloads.

## Only the reviewed rotary non-call is excluded

The ModuleList rule is unchanged: exact unmodified ModuleList, observed parent,
and every direct child actually observed. A real direct parent frame supplies
the previously missing observation; the list itself is not claimed to run.

The ApplyRotaryEmb non-call rule requires all of the following:

- Exact currently loaded AscendRotaryEmbedding and AscendApplyRotaryEmb types,
  actual parent/helper objects and the full alias sets.
- Current bound `_forward_method` is that class's `forward_oot`; its executed
  code equals the current source, whose reviewed SHA256 is
  `74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7`.
- Every actual parent invocation has a non-null returned delegate frame and
  exactly one actual return from the held
  `vllm.npu_rotary_embedding.default` operator inside that frame.
- No helper invocation was observed. Both healthy inventories yield the same
  bounded proof, and every group forward reproduces it. Current bindings and
  aliases are rechecked before the next group request.

This proves only that this reviewed branch did not call that helper in the
observed workload. Wrong source, another delegate, missing parent/call/return,
changed aliases, or any unknown unused executable module remain unresolved.
An actually called helper is counted as a real module, never as a non-call.
Projection ordinals, all module objects, all aliases, operators, consumers and
the original supported/unsupported/unobserved denominator are preserved.

## Ownership and validation

A foreign profile at installation is untouched. Thread mismatch, profile
replacement, late callbacks and uncertain removal cannot free the owner.
Removal reads back the actual current profile: after-side-effect failure
retains the original error but does not remove twice; before-side-effect or
foreign-owner failure retains the handle for explicit recovery. The existing
Native OFF, hooks, leases, logical reservations and Runtime gates still apply.
Query reads resource state and never advances discovery or writes an artifact.

CPU tests in `test_core_output39_direct_discovery.py` use real Torch hooks,
frames, operator returns, Host and Runtime. Their rotary source profile and
Native transport are explicit test doubles, not a production profile. They
cover a complete C2/C3 protocol, inherited direct forward, hook deduplication,
shared aliases, unresolved source/branch/module, actual helper calls, foreign
objects/globals/thread/profile, exception/None return, source/delegate drift,
late callbacks, and before/after removal failures with explicit retry.

The formal natural probe's `MODULES`/`SOURCES` and group implementation pins
include `serving39_output_discovery.py`. Root's source-only exporter must add
`tests/worker_extension/test_core_output39_direct_discovery.py` and this
document to its explicit contract list; the existing Worker extension source
tree already includes the new module. The test's existing fixture closure is
`test_core_output39_group`, `test_core_output39`, `test_postwrite_providers`,
`test_serving_session_lifecycle`, `test_core_host`, `test_extension` and
`test_call_binding`. No private originals, production source copies, binaries
or vendor modifications belong in the handoff bundle.

Frozen local verification: 90 tests passed in 30.769 seconds on Torch 2.10.0+cpu,
including 18 new direct-discovery contracts and the original late-hook gate.
The dormant-parent contract confirms that a shared operator's return joins
only its actual active parent frame. The final command was:

```text
python -B -m unittest test_core_output39_direct_discovery test_core_output39 test_core_output39_group test_core_output39_discovery_evidence test_core_output39_custom test_core_output39_group_custom test_output39_group_worker_core_probe test_output39_worker_core_probe test_postwrite_coverage test_postwrite_native_recovery -q
```

Use the normal repository PYTHONPATH and `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.
No local test is a Torch 2.9 production, NPU, yellow integration, CANN 9.x, or
A3 acceptance receipt. The existing preparation command/stage is unchanged;
root must freeze a new source snapshot and review new actual originals.
