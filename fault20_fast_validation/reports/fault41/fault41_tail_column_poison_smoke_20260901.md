# Fault41 Tail-Column Poison Smoke — 2026-09-01

## Status

`FAULT41_LD_PRELOAD_SYMBOL_LOADED_BUT_NOT_INTERPOSED`

This is a blocked exploratory implementation, not a PASS.

## Target hypothesis

Use the Qwen/vLLM equivalent of the original compressor/cache-state fault:

```text
_npu_reshape_and_cache
→ key/value cache write
→ next fused attention consumes cache
```

## Prototype route attempted

The exploratory route attempted a same-name C++ LD_PRELOAD wrapper around the
ATB/torch_npu cache-write path:

```text
atb::_npu_reshape_and_cache(...)
```

The intended fault primitive was to poison the tail column of key/value before
or during cache write, then observe downstream fused-attention consumption.

## Observed result

- The Fault41 preload `.so` was loaded into the worker process.
- `libop_plugin_atb.so` was loaded into the same process.
- Worker-side arm/disarm/query RPC wiring was reachable.
- No `fault41_tail.jsonl` trace was generated.
- Fault response matched baseline.
- No visible output degradation was observed.

## Root cause

The observed `_npu_reshape_and_cache` route appears to be registered through
torch_npu / ATB dispatcher machinery using a compile-time function pointer path.
It is not called through a PLT/JUMP_SLOT dynamic symbol lookup that ordinary
same-name `LD_PRELOAD` interposition can capture.

## Next direction

Do not continue with the same-name LD_PRELOAD route as the primary plan.

Potential next routes:

1. native torch dispatcher / extension-level wrapper;
2. ATB registration-level wrapper;
3. direct KV cache pre-read/post-write corruption using block table / slot
   mapping;
4. msprobe-guided cache tensor address discovery.

## Source status

No Fault41 source directory is present in this GitHub worktree at the time of
this cleanup commit. If a local or remote experimental source tree exists, it
should be reviewed separately before being added, and runtime `.so` artifacts
must remain excluded from Git.
