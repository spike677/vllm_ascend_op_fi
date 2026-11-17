# Four fresh FIA CPU-FP64 B/F cases: launch review

Read-only offline review of the new remote root
`/home/17-fault-c2c3-20261117/reviewed-next8-a2` verifies all eight calls in:

| Case | Actual dynamic key, twice | Actual M / S2 | Complete output SHA prefix |
|---|---:|---:|---|
| `fia-legal-cpu-fp64-baseline` | 109000000000000000 | 512 / 512 | `63d153f8892833d3` |
| `fia-legal-cpu-fp64-fault` | 109000000000400000 | 128 / 128 | `69d51d91aa24b472` |
| `fia-replay-cpu-fp64-baseline` | 109000000000400000 | 128 / 128 | `69d51d91aa24b472` |
| `fia-replay-cpu-fp64-fault` | 119000000000400000 | 128 / 128 | `90a1776554397373` |

All actual launches have block dimension 16 and schedule mode 1. The source and
compiled metadata determine the 8-byte MIX910B hardware-sync prefix and argument
slot 280. The verifier chooses the launch from independently captured registered
ELF bytes and function binding, then decodes the actual runtime inline argument
slice through the public schema. It requires complete raw equality with the
same host callback and exact typed/public-schema agreement. It never searches
for expected host bytes or chooses a launch by expected key/profile. Other
launches can be excluded only by their own independently observed binary binding.

The verifier source SHA is
`71c42dce81d83d29d4f6c09ed4d3ae3f7cda924f74aba4099fd6957eaaa5aa57`.
Both separately built vendor inventories were fully rehashed against their
exact completion manifests. The schema SHA is
`20dd6a361142aa9acd73f0787673e24660daef46846e5b362eb9306b15fe035c`.
The JSON receipt alongside this note includes complete original evidence hashes,
actual runtime keys, raw hashes, typed geometry, registered binary identities,
argument ABI and excluded launch records for each call. Every original file
read was rehashed afterward; no remote file was written and no NPU call was made
by this review. `attention_reverify.py` can reproduce each two-call native check
with the same runtime/host directory, exact registered vendor and public schema.

The original `independent_launch.json` already reports
`FINAL_ARGUMENTS_AND_LAUNCHES_VERIFIED` in all four cases. The separate legacy
`result.json` execution-layer status still says
`EXPERIMENTAL_EXECUTION_COMPLETED_NOT_LAUNCH_QUALIFIED`; it does not override the
independent final-argument verifier. Neither original receipt was edited.

Each case has two identical complete-output hashes, including the call collected
by the new CPU-FP64 backend. Both mechanisms have a different B/F complete-output
hash. Each new collector receipt reports 201326592 original bytes copied,
`NEW_COLLECTION_COMPLETE`, no errors and restoration of its original method.
These are fresh B/F observations. The prior separate same-process OFF evidence
is not relabeled as a new CPU-FP64 OFF run.

The raw statistics dump, stack and construct were not read by this launch
review. No E5 first difference, model effect, historical percentage, public
Worker admission or C2/C3 qualification follows. Independent fixed-tolerance
raw E5 comparison remains a separate review.
