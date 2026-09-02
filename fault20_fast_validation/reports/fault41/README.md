# Fault41 Reports

Fault41 is the KV-cache / tail-column poison exploration line.

Current status: BLOCKED / EXPERIMENTAL.

Known report:

- `fault41_tail_column_poison_smoke_20260901.md`

Boundary:

- Fault41 is not marked PASS.
- The known LD_PRELOAD experiment loaded its `.so`, but did not interpose the
  torch_npu/ATB dispatcher path.
