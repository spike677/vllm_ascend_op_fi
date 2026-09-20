# Fault17 — paired msprobe B/F evidence

Only baseline (B) and fault (F) are included. These are compressed original
statistics dumps, not full tensor dumps. Archive sizes and SHA256 checksums are
in `archives.json`. No experiment source or compiled runtime is delivered here.

## Experiment and observed response

- Model: Qwen3-VL-30B-A3B-Instruct-w8a8; A2/910B4, physical devices 2 and 5,
  TP=2, eager, CANN 8.5.1.
- Frozen VQAv2 validation question 42000, image 42:
  `What color are the gym shoes?` / `Respond with only one color word.`
- B: native spatial padding `[0,0,0]`; response `white`;
  token IDs `[5782,151645]`.
- F: visual.patch_embed.proj spatial padding `[0,7,7]`; response `red`;
  token IDs `[1151,151645]`.
- Both requests: HTTP 200, finish_reason=stop, 2 completion tokens.
  Both match their respective previously captured no-msprobe responses.
- Real Conv3d output remains finite BF16, shape `[1200,1152,1,1,1]`.
  This is a geometry-parameter deviation effect proxy, not proof of the
  original historical kernel defect.

## Capture configuration and contents

`task=statistics`, `summary_mode=statistics`, `level=mix`, ranks 0/1,
configured steps 0–7. The requests finish after two tokens, so actual data
contains **step0 and step1 only**, not eight populated steps.

Each archive contains 32 files under its own B/ or F/ root:

```text
probe_phase/dump_config.json
probe_phase/dumps/step{0,1}/rank{0,1}/
    dump.json
    stack.json
    construct.json
evidence/
    request, response, HTTP, install/remove and observation records
    probe acknowledgements/final status, service/startup records
```

Each group has 12 raw dump/stack/construct JSON files. First-forward evidence
includes Functional.conv3d.0.forward and the visual patch embedding module.
Input/weight/bias statistics agree across B/F; padding and output statistics
differ. Rank0 output norm: B=450.0, F=446.0.

The existing debugger's first start is advanced to the visual encoder entry,
because this pinned runner executes vision before its normal debugger start.
The later start in that same forward is coalesced; native stop/step and decode
starts remain unchanged. This matters when interpreting the dump window.

## Provenance and boundaries

- Successful host evidence: `/home/haoran/fault17_msprobe_bf_20260920_v2`.
- vLLM: `bcf2be96120005e9aea171927f85055a6a5c0cf6`.
- vLLM-Ascend: `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f`.
- torch 2.9.0+cpu; torch_npu 2.9.0.post1+gitee7ba04;
  transformers 4.57.6.
- Image SHA256: `d3fdefd1394f0af4453bbee1c095138148ee8e6e22ab7c6c594235da8432a0c3`.

No R/Z group is included. Recovery was validated in earlier independent
B/F/R experiments; this two-group capture does not independently repeat it.
The failed initial v1 observation attempt is not mixed into these archives.
Capture services were stopped and the resource lock released after collection.

## Verify and unpack

```powershell
Get-FileHash B_dump.tar.gz,F_dump.tar.gz -Algorithm SHA256
tar -xzf B_dump.tar.gz
tar -xzf F_dump.tar.gz
```

Compare the hashes with `archives.json` before unpacking. Keep extracted B/F
directories separate; do not merge their dump files.
