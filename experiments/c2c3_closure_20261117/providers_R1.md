# Provider workflow: implementation and queue handoff

Frozen baseline: `91decf5efe48dfdf8c5494e52d258ccde788cc5d`. This work does not
change the historical C1, C2=0/17, C3=0/17 or E5 qualification ledger.

| Mechanism | Reused real scope | Missing gate at R1 |
| --- | --- | --- |
| 20.buffer_substitute | DeepSeek natural first embedding/norm dynamic-QMM projection, exact Native x1 substitution and OFF | 135 structural modules are not an eligible denominator. Need all actual quant→QMM lineages, per-call exact native identity and independent observation, then each lossless/OFF. Existing private QMM ABI remains distinct from MatMul. |
| 26.residual_stomp | DeepSeek real adjacent down-projection writer/live residual/consumer and exact restore | Every allocator-dependent live writer/consumer must be independently observed and separately leased. Native all_hits is not a C3 capability. Earlier KV slot 61→77 and NaN remain E5 blockers. |
| 26.fia_destination | SmolLM2/GLM actual fresh-prefill FIA writer/consumer/OFF | Complete prefill/decode/request/cache lifetime inventory and persistent group drain are missing. Do not extend prefill qualification by changing a count. |
| quant.batch_shared_scale | DeepSeek one B8 first-decode projection and row mapping | Independent complete executed projection inventory, per-request row lineage, separate target leases and repeated-request OFF. Existing first-selected-projection context is retained. |
| 39.matmul_range | Qwen single actual output projection C1 | Full root/logits native denominator, same-shape cache disambiguation, all-target lossless modes, effect and group OFF. New implementation below is awaiting root-queued actual Worker validation. |

The first executable C2 sample is 39 on natural Qwen3-0.6B, TP1, eager BF16/ND,
CANN8.5.1 on 910B4. It observes every Torch dispatcher boundary in the complete
root and actual LogitsProcessor, and reconciles native eligible Run counters
independently of enabled state, owner, mode, hits, or dispatch admission. Calls
without eligible typed Native MatMul are retained as transparent observations.
Unwrapped native calls, multiple native Runs at one public boundary, unknown
public input layouts, missing outcomes, changed cache registrations, overflow,
or missing effects refuse the attempt; they are not removed from the denominator.

The additive `fault39_configure_cache_v2` requires exact observed native
signature, cache address and registration epoch, two actual observations, a
current typed registration, actual request/thread ownership, live input storage
guards, and `max_hits=1`. The original `fault39_configure_v1` and its
unique-signature rule remain unchanged. Registration epochs advance on each
`AddOpExecCache`, including address reuse. `registration_live` describes that
registry state; it is not an allocator-liveness certificate. An actual matching
Run and exact input/output evidence remain required.

Every target receives reference plus self-only, mat2-only and dual lossless
substitution. V2 additionally copies each temporary device allocation back into
the existing pinned host buffer and checks the complete byte SHA before Run;
the old C1 receipt's false readback field is not relabeled. Actual original input
storage hashes remain protected. Per-call native generations are separate from
the logical target group. Both source identities appear in the evidence.

After C2’s all-once request, all-target read-only OFF and disabled OFF must
restore complete logits. The common recovery controller stops all, drains all,
then allows any restore/release. All owned hooks are removed before same-Worker
health attestation. A live, nonserializable C2 proof may open a new bounded C3
group in that Worker. Every matching call in its two fault requests is counted.
No proof is imported from JSON or another process. Propagated nonfinite original
inputs remain visible; substituted inputs must still have verified finite exact
cancellation construction. New native targets fail discovery closure, not pass
as covered.

CPU contracts use real CPU Tensor operations and explicit mock native receipts.
They do not grant NPU or natural model qualification. Tests cover same-shape
separate cache identities, lost/unwrapped native denominators, stale registration
epochs, missing native hits, all-target OFF, C2-gated repeated calls and actual
whole-request module/dispatcher hooks without core method replacements.

Root-only execution order:

1. Snapshot the merged final source and run `engineering_20261117/build_manager_124.py`
   with `--mechanism 39.matmul_range --operator aclnnMatmul` into a new build
   directory. This uses no NPU devices. `build39-a1` was a retained CPU fixture
   failure (an intentionally invalid mask had not been reset); `build39-a2`
   passed its intermediate source before the additional v2 readback. Neither
   is an execution certificate for final source.
2. Build/install fresh wheels using `engineering_20261117/build_packages_124.py`.
3. Run `providers39_prepare_cases.py --installed INSTALLED --native NATIVE
   --probe SOURCE/experiments/c2c3_closure_20261117/providers39_worker_probe.py
   --build-sha ACTUAL_BUILD_SHA --source-commit MERGED_COMMIT --statistics --c3
   --out NEW_CASES_JSON`. This writes one reviewed case and starts no device.
4. Root executes its `run_queue_124.py` under `/var/tmp/fault22_npu_2_5.lock`
   with fresh card-idle inspection. All raw statistics/native artifacts stay
   under `/home/17-fault-c2c3-20261117`; only reviewed summaries enter Git.

The raw unfiltered statistics/mix B is phase2, F phase6 for each group. E5 uses
only raw B/F collector/config/stack/construct evidence, fixed rtol=1e-5 and
atol=1e-8. Native hits and full-tensor hashes do not select its first difference;
null, NaN, stack/alignment failures remain failures. No A3, multi-rank, decode,
other model or other CANN qualification is inferred from this probe.
