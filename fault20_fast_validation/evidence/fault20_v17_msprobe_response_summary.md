# Fault20 v17 msprobe response summary

This file records the model-level response texts for the canonical Fault20 v17
MindStudio Probe validation runs committed under
`fault20_fast_validation/msprobe_dump/`.

The raw response JSONL files are intentionally not committed. They remain in the
remote forensic evidence directory:

```text
/home/haoran/fault20_msprobe_validation_20260901/runs/
```

## Request shape

Both responses below used the same completion setup:

```text
model: qwen3-32b-w8a8-msprobe
prompt_tokens: 12
completion_tokens: 16
total_tokens: 28
http_status: 200
finish_reason: length
```

## No-hit control

Source:

```text
/home/haoran/fault20_msprobe_validation_20260901/runs/dump_nohit_control/responses.jsonl
```

Response text:

```text
 Paris. The capital of France is Paris. Paris. The capital of France is
```

Text SHA-256:

```text
46c7dc050ee650788ed800f030c7fd7ca18e2d123a6ae8e8159324163344fb89
```

Generated tokens:

```text
[" Paris", ".", " The", " capital", " of", " France", " is", " Paris", ".", " Paris", ".", " The", " capital", " of", " France", " is"]
```

## All-hit INT4-equivalent QMM fault

Source:

```text
/home/haoran/fault20_msprobe_validation_20260901/runs/dump_allhit/responses.jsonl
```

Response text:

```text
 Paris的城市琮 견s情节精选ATUSمين圩ALES皆inks ië
```

Text SHA-256:

```text
e6759c97835742c29edd535279de6c1a32c882437c4ac7cbf7b1af8ad72b9505
```

Generated tokens:

```text
[" Paris", "的城市", "琮", " ", "견", "s", "情节", "精选", "ATUS", "مين", "圩", "ALES", "皆", "inks", " ", "ië"]
```

## Interpretation boundary

This summary records only the final model responses for the canonical no-hit and
all-hit validation groups. Native fault-injection traces such as `run_trace.jsonl`
and `metadata.jsonl` are excluded from Git by design and remain available only in
the remote forensic evidence package.
