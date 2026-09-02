# Fault41 Reports

Fault41 is the `state/cache tail corruption` fault family.

Current status: DESIGN FEASIBLE / IMPLEMENTATION BLOCKED / EXPERIMENTAL.

Reports:

- `fault41_tail_column_poison_smoke_20260901.md`
- `fault41_state_cache_tail_corruption_roadmap.md`

Current boundary:

- The old same-name LD_PRELOAD experiment loaded its `.so`, but did not
  interpose the torch_npu/ATB dispatcher path.
- The next valid milestone is not all-hit corruption. It is read-only closure of
  the real cache writer, cache storage/slot metadata, and later attention
  consumer relationship.
- Fault41 is not marked PASS.
