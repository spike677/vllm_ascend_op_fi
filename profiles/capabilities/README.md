# Capability manifests

These manifests are the handoff protocol between the validated low-level
implementation in this repository and DemonCAT's `RuntimeBinding` layer. They
describe what exists; they are not a second runtime registry. The v0 manifests
are JSON documents so validation and loading require only the Python standard
library.

Required fields:

- `schema_version`: manifest format version;
- `id`: stable implementation capability id, without a Fault number;
- `kind`: `injector`, `observer`, `discovery`, or `proxy`;
- `maturity`: `validated`, `experimental`, or `probe`;
- `semantic_mapping.candidate_positions`: positions the backend can support;
- `execution_plane` and `mechanism`;
- `supports`: observe/lossless_validate/inject/restore flags;
- `canonical`: canonical source/build/evidence references;
- `compatibility`: validated environment metadata where known;
- `limitations`: explicit unsupported or unproven claims.

`semantic_mapping.candidate_positions` and `execution_plane` are deliberately
independent. For example, a `kv_cache` semantic position can be implemented in
the `vllm_backend`, `torch_npu`, or `op_api` execution plane.

The manifests do not define `PrimitiveSpec`, normalize user targets, assign
request policy, or expose a production CLI. Those remain DemonCAT concerns.
The manifest is descriptive metadata, not a second position or primitive
registry. It does not imply that the referenced implementation has already
moved into `src/ascend_fi/`.
