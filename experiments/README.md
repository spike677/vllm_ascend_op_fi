# Experiment and evidence boundary

Fault numbers identify historical validation cases. They are retained for
reproducibility and evidence review, while reusable runtime concepts move
toward `src/ascend_fi/`.

No existing experiment tree is moved in the first restructuring step. Moving
hundreds of dump files or native sources now would obscure history and risk
breaking validated build paths.

## Workspace ownership

The canonical development clone for this repository is:

```text
D:\project\demoncatpfi
```

Its starting `HEAD` for this architecture task is `31d8e365`; the architecture
and contracts commits are built locally on top of that commit. The older
`C:\Users\admin\Documents\demoncatpfi` copy is a legacy/local investigation
clone and is not an alternate canonical workspace.

The nested `fault26_push_wt_20260902/` worktree found in the legacy clone is
external experiment state. It is not part of this repository's product tree,
source inventory, or capability manifests.

## Transitional index

| Case | Current canonical location | Future reusable roles |
|---|---|---|
| Fault15 | `fault15_scale_granularity/` | discovery, tensor statistics, evaluation |
| Fault20 | `fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/` and `l4_metadata_lifecycle_proxy/` | `backends/op_api`, precision/tensor strategies, non-finite observer |
| Fault26 | `fault20_fast_validation/source/fault26_readonly_trace_v1/`, `fault26_fia_writer_proxy_v1/`, reports and evidence | operator trace, topology discovery; specialized stomp remains experimental until generalized |
| Fault39 | shared Fault20 native runtime plus `fault20_fast_validation/reports/fault39_*` | output-corruption strategies, tensor/non-finite observer, evaluation |
| Fault41 | canonical committed source not yet curated; current scripts/runtime remain external investigation assets | cache observer/state backend only after source and evidence are frozen |

## Freeze rules

- Existing experiment paths remain valid and are the canonical build source
  until an explicit migration passes build, ABI, baseline, fault, and recovery
  regression.
- New product-facing modules do not use Fault numbers in package names.
- Experiment reports may name a historical Fault, but capability manifests use
  backend/strategy/observer vocabulary.
- Runtime binaries, service logs, transient traces, and local work directories
  are not promoted into `src/`.
- Root-level or external `fault41_*`, `run_fault41_*`, and `tmp_*` investigation
  scripts are not part of the product commit until they are curated under
  `experiments/fault41_state_cache_corruption/` with hashes and evidence.
- Evidence claims remain limited to their recorded model, hardware, CANN,
  vLLM, topology, and request conditions.

## Fault41 external status

```text
FAULT41_FUNCTIONALLY_COMPLETE
WITH_KNOWN_MSPROBE_OBSERVABILITY_LIMITATION
```

This status describes the current external/remote experiment only. It does not
create a validated capability manifest because canonical committed source has
not yet been curated. No temporary probe is copied into product source in this
phase.
