# Source116: independent single-task Fault39 review

The 124 card-5 run completed the ordinary public Host/Worker path with four
actual scheduler requests: normal, fault, OFF and health. The scoped effect,
consumer, numerical recovery and resource records are consistent. This is a
blue-zone deployment check on Qwen3-0.6B, not verified inner-main-project
integration or new C1/C2/C3/E5 qualification.

## What actually happened

Normal examined **3 actual projection returns**, within the fixed limit of 16,
then bound `model.layers.0.mlp.gate_up_proj`. Fault and OFF retained that exact
module/function/weight/consumer binding; no alternate target was substituted.
The actual producer was `vllm.unquantized_gemm.default`, returning an NPU BF16
tensor of shape `[14, 6144]` (86,016 elements). This is one target in one request,
not coverage of all layers or requests.

All four requests had distinct actual scheduler ids, the same input digest and
the same Worker/model lifetime. The full normal preimage digest matches the
fault preimage. Native mode 0 changed zero elements. The single mode-1 write
changed flat element 0 only:

| Item | Normal | Fault |
| --- | --- | --- |
| BF16 bits | `0x3f77` | `0x7f80` |
| Decoded value | `0.96484375` | positive infinity |
| Changed elements | 0 | 1 |
| Actual consumer | `npu.npu_swiglu.default` | same operator and semantics |

The consumer's actual input geometry and storage identity join the producer's
current return and the Native opened span. It reads element 0 and returns once.
Normal/F/OFF use the same consumer dispatch ordinal and `dim=-1`. Consumer
output values were not separately dumped; subsequent hidden/logits hashes
changed in F. These are runtime records, **not msprobe statistics**.

| Request | Generated token | Text | Complete hidden/logits vs normal |
| --- | --- | --- | --- |
| normal | 576 | ` The` | reference |
| fault | 0 | `!` | different |
| OFF | 576 | ` The` | exact hash match |
| health | 576 | ` The` | exact hash match |

Only one generated token was requested. This establishes the observed first
token change; it does not assess multi-token generation or answer quality.
Health deliberately has no injection report. Its independently recorded actual
request, input digest, output hashes, token and text supply the health check.
An absent health injection report is not a missing fault observation.

## Native and cleanup checks

Both opened leases contain the complete sequence
`open → apply → restore → release → live`. Every operation returned zero; the
normal apply uses mode 0 and changed 0, while the fault apply uses mode 1 and
changed 1 at index 0. Geometry, bytes, selected indices, handle, generation and
Native owner are consistent across each lease. Native owner is derived from
the recorded **Runtime provider owner**, not the distinct Host ticket owner.
The final live query is false for both leases, with no first failure.

The execution records exact owned-span restoration, separately corroborated
at model level by actual OFF/health hidden and logits hash equality. This
review did not obtain the full private tensors or independently reread device
memory. It cross-checked recorded Native events, bounded typed readback, hashes
and request/consumer joins; it does not promote a hash into a new tensor dump.

The original Host receipt contains 17 actions, including exactly four pairs of
advance/query. Its cumulative actual-request records equal the per-request
records in the result. Finish has zero hooks, leases and inflight calls, an
empty Native lease list and no pending replay. Unregister clears registration
and binding reservation; detach releases ownership. The deployment binding
was removed. The controller separately records exit 0, no OOM, owned-container
removal and a clear post-cleanup gate. Container removal is not used as proof
of numerical recovery.

The fault query reports a current effect of 1; OFF and health report no current
effect. Their retained last-fault summary remains 1, correctly distinguished
from a new injection. No extra recovery requests, full census, dump or msprobe
collection were used.

## Identity and limits

Execution source: `134e11886c47f969bd11de498affd73e8ce3c3f3` (Source116).
Manifest: `aa40d9908ebae4b5063f32dadf6fc998a2afcb4b5db4afc182ad4ff6633d15e4`.
The previously reviewed Source101 Native DSO and build key are listed in the
adjacent summary; they are not relabelled as a Source116 Native build.

This review read the original result, full bounded Host receipt and controller
using standard-library code, without importing the provider or running a model.
It compared their records directly rather than accepting the probe's top-level
success status. Original metadata remains private; only small facts and hashes
are published in [the summary](output39_source116_single_task_summary.json).

The Source114 first-normal closure refusal remains a separate preserved failed
attempt with 0 F and successful ownership cleanup, not numerical OFF success.
The new source's implicit-class-cell adaptation was exercised here through
`AscendMergedColumnParallelLinear.forward → ColumnParallelLinear.forward`.
Native finite-original checks were unchanged. This controlled output proxy is
not proof of a historical native out-of-bounds defect, cumulative all-target
C3 behavior, multi-model support or successful inner-project wiring.
