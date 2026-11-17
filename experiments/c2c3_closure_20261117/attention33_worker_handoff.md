# Natural TND FAI Worker queue

Root owns every container/device launch and the existing resource lock. No
Agent4 NPU call or container start occurred. Use the exact reviewed8.5.1
`quay.io/ascend/vllm-ascend:v0.18.0` image and Qwen3-0.6B model. A new wheel
must include `paged33_tnd.py`, `paged33_tnd_probe.py`, the updated launch
verifier, and root's reviewed `collector_dispatch.py` compatibility module.
Keep all package imports under `/installed`; the driver verifies that location.

The root-run new FAI build must finish its actual public serializer/geometry
check, complete host package, all six matching kernel entries and registration
before native execution. `tnd33-build-a2` is not admitted. Observer preflight must
report successful bounded memory read, and its original absolute `/build`
library/header paths must be retained in the execution container.

Suggested fresh mounts:

| Host artifact | Container mount |
| --- | --- |
| Integrated new driver source | `/repo`, read-only |
| New installed wheels | `/installed`, read-only |
| Actual FAI build `build/run` | `/native`, read-only |
| Exact8.5.1 observer preflight parent | `/build`, read-only |
| Qwen3-0.6B | `/model`, read-only |
| Fresh per-case output | `/evidence`, writable |
| Reviewed collector package | `/msprobe_packages`, read-only |

Use `PYTHONPATH=/installed:/msprobe_packages`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, offline model settings and normal root
NPU/driver mounts. Do not set vendor or LD_PRELOAD externally: the parent
verifies clean state and execs the child with exactly the reviewed private
vendor, observer and synchronous uncached settings before importing torch.

First case, independent stock reference:

```text
python3 /repo/experiments/c2c3_closure_20261117/attention33_worker_probe.py
  --phase stock --model /model --out /evidence/run --experimental-qualification
```

Then separate identical-path native baseline and fault cases:

```text
python3 /repo/experiments/c2c3_closure_20261117/attention33_worker_probe.py
  --phase baseline --model /model --out /evidence/run --experimental-qualification --statistics
  --completion /native/evidence/completion.json --completion-sha256 ACTUAL_NEW_COMPLETION_SHA
  --vendor /native/stage/vendors/dcat_f33_fai_transformer
  --public-schema /native/evidence/fai_schema/fai_abi_schema.json --schema-sha256 ACTUAL_FAI_SCHEMA_SHA
  --observer-receipt /build/attention-preflight/result.json --observer-receipt-sha256 ACTUAL_851_OBSERVER_SHA
```

Change only `--phase fault` for F. Do not use the9.1 observer or vendor. Each
case runs one warm request followed by baseline-before, baseline-repeat,
measured, OFF-after and read-only healthy-after. Each request has a legitimate
fixed1024-token text prefill and two generated tokens, yielding exactly one
natural KV1025 first decode across all28 layers. There is no core-method
replacement, synthetic token-row creation, cache repacking or output edit.

The measured native phase observes every actual28 calls. Other native stages
use the original OFF key. Hooks retain actual tensor references during inference;
full hidden/logit/layer-output and valid logical KV readbacks occur after the
collector stops. Logical readback page order tolerates allocator permutation
while preserving each actual used element; it is never fed back into an
operator. Source references, real request IDs, actual storage/offset/shape,
writer/page/slot lineage and final raw kernel arguments are captured separately.

Baseline-repeat, OFF-after and healthy-after must have exact represented bytes
against baseline-before. Baseline measured must also be exact. Fault difference
is reported honestly, including no difference. A separate stock case supplies
the stock reference; equality to it is not implied by same-vendor OFF equality.
Failed capture, nonfinite results or ownership changes preserve evidence and
cannot grant recovery. Cleanup aborts only the driver's own enqueue IDs and
restores only owned hooks/API wrappers and native environment values.

Only measured stages collect unmodified statistics/mix/high with fixed step0,
same path/configuration/construction and complete natural request. Root's
independent raw B/F comparator must apply rtol1e-5/atol1e-8, preserving null/NaN
and rejecting missing/construction/stack differences. Runtime/full tensor files
must not select E5. Every result remains C2=false/C3=false until separately
reviewed; stock, old BNSD, unentered arch22 and new FAI evidence stay distinct.
