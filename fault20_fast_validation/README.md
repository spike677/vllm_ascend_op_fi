# Fault Validation Workspace

`fault20_fast_validation` is a historical directory name. The tree now carries
multiple developer-mode fault validation efforts for local vLLM Ascend model
testing:

- Fault20: QMM buffer substitution and semantic-slot msprobe validation.
- Fault26: FIA wrong-destination residual stomp proxy validation.
- Fault39: MatMul/QMM numerical non-finite exploration and reports.
- Fault41: KV-cache/tail-column poison exploration, currently blocked.

The directory is intentionally not renamed yet because existing build and launch
scripts still refer to the historical path.

## Directory contract

- `source/` contains reproducible source code and build scripts.
- `harness/` contains reusable launch, validation, and packaging helpers.
- `evidence/` contains curated frozen evidence such as selected raw msprobe
  dumps, compact summaries, and restoration archives.
- `reports/` contains human-readable validation reports and status notes.
- `runtime/` is for locally generated binaries and must not be committed.

Do not commit compiled `.so`/`.o` artifacts, service logs, container logs,
temporary runtime directories, or large native JSONL traces unless a task
explicitly designates them as curated evidence.

See `MANIFEST.md` for the current fault-to-source/evidence mapping.
