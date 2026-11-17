# Strict MLA fresh-prefill context adapter

The first real DeepSeek Fault26 residual attempt refused before injection
because its `AscendMLAMetadata` has no dense `seq_lens_list` field. Fault20's
intermediate QMM qualification shares this reader and would encounter the same
boundary. The reader now has an explicit adapter for the inspected MLA path.

`snapshot_fresh_prefill` still requires one real fresh request, synchronous
TP1/PP1/DP1, no prefix cache, chunking, speculative tokens, graph capture,
context parallelism or padding. Scheduler request IDs, prompt/position CPU
mirrors, actual device input buffers, block ownership and slot indices retain
their existing checks. No worker execution method is changed.

MLA selection requires the actual loaded class
`vllm_ascend.attention.mla_v1.AscendMLAMetadata` and reviewed live source
fingerprints for `AscendMLAMetadataBuilder.build`,
`AscendMLAMetadataBuilder.build_prefill_metadata` and
`AscendMLAImpl._forward_prefill`. Matching an architecture name or merely
providing similarly named fields does not open this adapter.

The observed CPU fields are read directly: top-level `seq_lens_cpu`,
`seq_lens`, `query_lens` and prefill `query_lens`, `seq_lens`, `context_lens`.
All must equal the actual scheduled prompt length. In this implementation,
`prefill.context_lens` contains full sequence lengths; the adapter explicitly
does **not** interpret it as a previously computed cache prefix. Freshness is
proved by the real request and scheduler counters.

MLA has no metadata `causal` flag. The causal contract is the reviewed actual
prefill consumer's `mask_type="mask_type_triu"`; the shared 512×512 mask's
actual alias, dtype, device and geometry are checked. Its device values are
not read or certified by this context reader. The derived prefill query-start
buffer is also marked as unread rather than assigned invented values. The
root-level query-start and position buffers and both block-table views retain
actual storage checks.

CPU validation passed: 25 context tests, including 6 new MLA tests covering
real-field preservation, each inconsistent length, foreign buffers, unsupported
decode/chunk/CP/padding, forbidden device-value reads and unreviewed class
identity. The 31 Fault20 tests still pass. A separate read-only check against
the downloaded actual `mla_v1.py` verified all three source fingerprints
without importing the NPU backend or calling a device. These results establish
reader implementation checks; real subsequent Worker acceptance remains
separate.
