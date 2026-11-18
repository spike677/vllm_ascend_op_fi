# FIA public return lifetime repair (CPU contract only)

The new source44/pkg18 SmolLM2 census reported the actual FIA writer as TND
`[13, 9, 64]`, stride `[576, 64, 1]`, bfloat16 / dtype 27, format 47, and
14976 bytes. Its residual was `[13, 576]` with the same byte count. The missing
public output was an ownership boundary problem, not a byte-count mismatch.

The offline, exact-source excerpts provided by the root establish this path:

| Source | SHA-256 | Relevant behavior |
| --- | --- | --- |
| `attention.py` | `990ead07c828e26192ca46b4b8abae7a6792f67c27510705ad7b483035f3631c` | `Attention.forward` allocates its own output, calls `unified_attention_with_output`, and returns a flattened view of that buffer. |
| `ascend_attention_v1.py` | `bbe69c3015ee15789dea0d6ec9795243458b68494294c4c023ed5a5d665c550b` | `forward_fused_infer_attention` gets the real torch_npu API return, views it as TND, then copies it into the caller's output. |
| `torch_npu_init.py` | `967690b75576c1dfb76ed3cef3bd8a876fe94ec3195e4afa3116ef7d31e9b160` | The public API aliases are populated from the actual `torch.ops.npu` dispatcher packets. |

Previously, `Writer26Experiment._post` could see only the distinct module output.
It could not retain the original Native writer tensor. The existing consumer
lifetime check consequently failed even when the residual producer was known.

`writer26_public.FiaPublicOutputObserver` now observes only the already loaded
`npu_fused_infer_attention_score` / `_v2` aliases. It invokes the original outer
callable exactly once and retains the literal returned tensor, without creating
a view, copy, allocation, or reconstructed tensor. It accepts an exact dispatcher
packet, or the existing SHA-pinned msprobe registry wrapper chain whose closure
terminates at that exact packet. Arbitrary outer wrappers and `__wrapped__`
attributes confer no authority. Unsupported aliases remain in the census.

Each public return requires one actual Native callback and one ordered
`fia_before -> fia_gws -> fia_enter -> fia_exit` chain in the same synchronous
request/module/thread scope. The original complete metadata is checked against
`fia_before` and the callback, including shape, strides, extent, dtype and format.
GWS and enter must describe the same effective destination and executor. The
real exit record contains only call/thread/executor/status; it does not repeat
tensor metadata. A qualified L/F lease can intentionally change the effective
destination, while the retained public object remains the original formal.

The retained writer must survive to the actual public residual consumer. The
consumer's residual must share the retained producer's actual StorageImpl
identity; an actual view is accepted and a same-value clone is refused. Existing
Native completion receipts, full-byte output comparisons, restored residual
checks, per-call lease acknowledgments, OFF recovery and same-Worker health
checks remain required. A prior public-return proof cannot authorize a later
Native callback outside an owned current public API call.

Aliases are installed after the serving collector starts. They are restored
before the collector stops, including the failed-request path. Cleanup checks
the actual current alias, never removes a foreign wrapper, and retains a
retryable owner after failure. A setter that restores the original and then
raises is distinguished from a setter that fails before changing it. Public
alias counts are included in the Worker removal gate. The implementation hash
and deployment source list include the new observer.

Validation used real CPU Torch dispatcher packets and CPU tensor lifetimes,
with explicit Native-v2 mocks. The mock mirrors the real trace's sparse exit
record. Sixteen new tests cover copied module output, strong lifetime through
the consumer, actual residual views, same-value output/consumer clones, changed
shape/format/token, multiple Native calls, a later unobserved API call, reviewed
collector coexistence, foreign wrappers, and cleanup failures before/after
alias mutation and a failed owned hook removal. The new and existing focused
suites passed 47 tests:

```text
python -B -m unittest test_writer26_public_return test_writer26_group_experiment
  test_writer26_inventory test_writer26_coverage test_collector_dispatch
```

This change has no NPU execution evidence and grants no C1/C2/C3 qualification.
The complete C2 denominator is unchanged. Existing bounded QMM candidate effects
remain a separate stage; unresolved QMM census entries are not reclassified or
removed by this repair. Actual FIA admission still requires a new root-controlled
deployment and its real Native, metadata-ownership, consumer, OFF and health
receipts.
