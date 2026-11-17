# Natural Worker preparation for both BNSD FIA strategies

`worker_extension/attention28_prefill.py` supplies an independent **read-only**
prefill boundary. It is not a common Provider or a NativeAdapter admission.
The two requested policies remain separate:

| Mechanism / arm | Requested native mode | M / S2 | Requested key |
|---|---|---|---|
| Legal baseline | coupled_base512 | 512 / 512 | 109000000000000000 |
| Legal fault | coupled_m128 | 128 / 128 | 109000000000400000 |
| Replay baseline | coupled_m128 | 128 / 128 | 109000000000400000 |
| Replay fault | coupled_replay_m128 | 128 / 128 | 119000000000400000 |
| OFF, either | off | 512 / 512 | 103000000000000000 |

These are requests from the already tested standalone contract, not claims
that a natural Worker selected a key. `strategy_request` never enters native
scope or sets environment variables.

## Concrete interface

Inside the initialized standard Worker, after collector startup if one exists:

```python
audit = FiaPrefillAudit(worker, max_calls=224)
audit.install()
audit.expect(actual_internal_request_id, actual_2048_token_prompt)
# The driver runs the ordinary enqueued request through the unchanged engine.
report = audit.report()
audit.close()  # before collector teardown; restores only its own wrappers
```

The installed hooks use `ForwardBinding`, `require_serial_eager` and
`snapshot_fresh_prefill`, with the independently reviewed actual attention
backend/layer source identities. They corroborate driver IDs against the real
input batch and request, real module/context ownership, full fresh prefill
lengths and actual API arguments. They retain tensor metadata without reading
Q/K/V/mask values. Unknown operators, TND/page16, page tables, prefix/quant/rope
inputs, changed lengths, masks replaced by equal-shaped objects and
noncontiguous QKV do not become BNSD candidates. Original positional/keyword
arguments and the return object pass through once. A later API owner is never
overwritten during cleanup. Observation-budget overflow remains explicit and
does not skip the original operator.

The current scheduler reader supports one fresh request, so this new boundary
accepts only B=1 as a candidate. The standalone B=8 result is not treated as
natural batched Worker evidence. Matching BNSD metadata still leaves mask
values, actual QKV bytes, device core budget, native route, launch, full tensor
bridge/recovery and raw B/F statistics unproven. All reports retain
`worker_qualified=false`, `native_launch_verified=false`, C2=false and C3=false.

`verify_sdk_binding(completion, expected_sha, contract_path, actual_sdk_root)`
reads the actual five SDK interface files and checks sizes/digests. The initial
legal/replay registered completions predate `sdk_interface_files`, but both pin
the original source-build contract SHA
`4b709903ea4ba1a8dc60a0608c4f443df7cc6e05669167391bec43b27077a91d`.
Only that exact contract may supply absent legacy interface rows. New explicit
rows must equal its pinned CANN 9.1 set. The helper rejects the separate TND33
variant and alternate SDK probe manifests. SDK identity alone still sets
`loading_allowed=false`; full Worker initialization and real stock/private
library coexistence remain required. No 8.5.1 Norm/FIA loader pin is imported
as a 9.1 loading permission.

## Prepared root queue probe; not executed by this agent

After integrating and building a new wheel, root can use its normal resource
queue to execute, in the **stock** vLLM 0.18.0 / CANN 8.5.1 image:

```sh
python3 -B /source/experiments/c2c3_closure_20261117/attention28_prefill_worker_probe.py \
  --model /model --out /evidence/run --memory .35
```

Mount `/home/gyl/models/Qwen3-0.6B` as `/model`, the fresh source snapshot as
`/source`, and the installed new wheel as `/installed` using the existing root
queue conventions. No private vendor, observer preload or native policy should
be present before startup. The script starts no container. It runs standard
eager TP1, max_seqs=1, page16, one 2048-token legitimate text prefill, max_tokens=1,
with chunking/prefix-cache/async/speculation disabled. It records healthy
before, observed, and healthy after generations, the real enqueue mapping and
per-layer API/layout/input metadata. Generated-token equality is explicitly
not full tensor equality. It does not collect statistics or select a first
difference. Source/metadata mismatches, cleanup failures and overflow remain
failures with original evidence.

The expected current outcome is a natural API/layout census with no BNSD
eligibility, because the already observed Qwen path is TND. No assertion in the
probe forces that outcome: it records whatever the actual original API uses.
This is useful evidence for deciding whether a new native prefill route is
needed. It is not a substitute for the pending 33 stock/private bridge or the
fresh FA build and launch work.

The new boundary tests exercise metadata-only candidates, TND refusal,
freshness/optional/mask/storage refusals, real SDK file mismatch and separate
33 artifact rejection, the original one-call/return-object contract, bounded
observation, and refusal to overwrite another API wrapper owner. CPU tests
grant no natural scheduling or NPU qualification.
