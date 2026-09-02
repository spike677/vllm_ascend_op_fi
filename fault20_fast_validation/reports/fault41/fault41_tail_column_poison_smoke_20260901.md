# Fault41 Tail-Column Poison Smoke — 2026-09-01

## Status

`FAULT41_LD_PRELOAD_SYMBOL_LOADED_BUT_NOT_INTERPOSED`

This is a blocked exploratory implementation, not a PASS.

## Fault family

Fault41 is now classified under the internal family name:

```text
state/cache tail corruption
```

The first implementation should not claim natural cache OOB. The accurate
prototype name is:

```text
OP_TRIGGERED_STATE_CACHE_TAIL_CORRUPTION_PROXY
```

If the victim is later proven to be a concrete KV-cache slot, the name can be
narrowed to:

```text
KV_CACHE_POST_WRITE_TAIL_CORRUPTION_PROXY
```

## Target hypothesis

Use the Qwen/vLLM equivalent of the original compressor/cache-state fault:

```text
_npu_reshape_and_cache
→ key/value cache write
→ next fused attention consumes cache
```

The intended semantic chain is:

```text
real cache writer executes normally
→ controlled mutation of a proven cache slot tail region
→ later attention consumes corrupted cache state
→ token/model behavior changes
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

## Consequence

Do not continue by repeating the same-name LD_PRELOAD dispatcher interposer. The
next useful work is to locate the real cache writer/exported API/typed GWS or
executor path used by the current model and prove the writer-consumer relation
in read-only mode.

## Source status

No Fault41 source directory is present in this GitHub worktree at the time of
this cleanup. If a local or remote experimental source tree exists, it should be
reviewed separately before being added, and runtime `.so` artifacts must remain
excluded from Git.
