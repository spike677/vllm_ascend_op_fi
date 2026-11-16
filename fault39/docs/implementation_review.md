# Fault39 review record

The implementation is isolated from previous unpublished fault work. Review
found and corrected these issues before attempting corresponding acceptance:

| Finding | Correction / evidence boundary |
|---|---|
| A successful old observation could survive a later hit | Reset every per-hit field before admitting a new Run; bind proof to generation and hit; require terminal/zero-lease evidence in Inspector |
| Formal parameter order alone does not independently prove live slots | Observer-only P2 compares all three native addresses with the live PyTorch storage addresses before any input substitution |
| Worker snapshot or collector exceptions could poison thread scope | Cleanup uses nested finally blocks; instrumentation failures remain explicit and do not mask model exceptions |
| Two adjacent inventory reads could falsely imply stability | Require an additional real Run observation across repeated workloads |
| Null could be reported as equality | Null is unobservable, including one-sided null; empty evidence is insufficient |
| `math.isclose` used a different tolerance rule | Explicit baseline-referenced additive tolerance, with recorded rtol/atol |
| Physical card ID used as a logical visible-device index | Physical mapping is controlled by container device cgroups; in-container logical index is separately observed |
| Device mappings incorrectly treated as active reservations | Record Docker/OCI access grants separately from actual processes/handles |
| `torch.mm` selected aclnnMm, not aclnnMatmul | Preserve negative finding; use torch.matmul and verify actual Run API |
| Split libopapi_nn bypassed the main metadata proxy | Reuse historical 8.5.1 split-proxy loading method in owned containers only |
| Model warmup metadata predates the first Worker RPC | Bind same-process unbound metadata to the actual immutable Worker rank; no inference from target config |
| Qwen3 has several legitimate target geometries | Strict mode refuses; optional max-K/min-M rule is explicit and keeps tie rejection |
| A short EOS request left collector steps for the following OFF request | Explicit owner-checked finish_capture stops and clears the collector at the selected request boundary; affected Llama data are superseded and recollected |

CPU/mock checks do not establish vendor execution, NPU stream recovery under
device errors, actual inner-framework owner composition, or arbitrary-model
support. Real results must remain separated by stage and source/build identity.
