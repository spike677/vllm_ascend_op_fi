# Source Directory Map

This directory contains source code for multiple local vLLM Ascend
fault-validation efforts. Directory names are preserved for compatibility with
existing build and launch scripts.

## Fault20

- `l4_buffer_substitute_allhit_v1/`
  - `OpExecCache::Run` interception.
  - Temporary device-buffer substitution.
  - INT4/INT2-equivalent transforms and all-hit controls.

- `l4_metadata_lifecycle_proxy/`
  - libopapi metadata proxy.
  - QMM/FIA typed metadata capture.
  - Semantic-slot proof support for msprobe-active execution.

## Fault26

- `fault26_fia_writer_proxy_v1/`
  - libopapi/libcust_opapi proxy wiring.
  - FIA writer observation.
  - Wrong-destination residual stomp proxy.
  - Apply/restore/pass proof and sustained all-hit support.

- `fault26_readonly_trace_v1/`
  - Read-only AddRms/FIA/QMM topology tracing.
  - Worker extension and collective RPC controls.
  - Typed probe and analysis helpers.

## Fault39

Fault39 currently reuses the lower-level L4 proxy framework and is documented
mainly through reports.

## Fault41

No Fault41 source directory is present in this GitHub worktree at the time of
this cleanup. The known exploratory status is documented under
`../reports/fault41/`.

## Exclusions

Do not commit runtime build outputs from this tree:

- `*.so`
- `*.o`
- `__pycache__/`
- temporary launch outputs
- raw JSONL runtime traces
