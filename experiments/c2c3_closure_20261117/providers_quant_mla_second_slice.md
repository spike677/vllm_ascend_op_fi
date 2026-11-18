# Source61 MLA metadata: bind both actual slices

The source61 healthy quant census retained all 270 decode calls and cleaned all
owned hooks, API wrappers, modes, profiles and live metadata records. It did not
qualify a Native target. Besides the separately addressed ND storage rank, its
builder records refused `NO_OBSERVED_PREFIX_SLICE_TO_METADATA_POSITION_RETURN`.

The source-checked `AscendMLAMetadataBuilder.build_decode_metadata` in
`vllm_ascend/attention/mla_v1.py` (SHA256
`17892014ecd431a1804fb81f5380509275e1440b41e1f9e0870ba8440c365516`)
actually executes two slices:

```python
input_positions = common_attn_metadata.positions[:self.num_actual_tokens].long()
input_positions = input_positions[:self.num_decode_tokens]
```

It passes the second object to the actual cached rotary producer and returns
that same object in `AscendMLADecodeMetadata.input_positions`. The rotary source
SHA remains `74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7`.
Both files were checked against root's existing source61 source cache without
importing the runtime or performing any device operation.

The actual source61 events were a first slice at sequence 1, identity int64
`aten.to.dtype` at sequence 2, then the second slice at sequence 10. The returned
second slice was the rotary input object. A later source slice at sequence 31
had the same storage but a different object and must not substitute for it.
Both decode requests showed this structure. The complete inventory remains on
124 with SHA256
`a98019062d5b474fd07fa9d464dbee0fc794c288bf94357b97dd878e533b0c59`.

The adapter now walks exactly two real `aten.slice.Tensor` edges backward from
the returned object: final output → intermediate object → actual common source.
Each edge requires the actual input/output identities, parameters `(dim=0,
start=0, end=8, step=1)`, one completed original call, strict event order, exact
storage/offset/dtype/extent and an unchanged write epoch. The optional actual
int64 no-copy cast is restricted to the intermediate object between those two
events. The final rotary producer must belong to this same builder and use this
same returned positions object. There is no value/shape-only inheritance and no
generalization to graph-padding concatenation, speculative batches or unknown
operators.

PyTorch's transient raw view version 0 is retained as observed. At the completed
builder boundary both view objects must have the unchanged source's actual
version (6 in source61). No other metadata field may drift. The source,
intermediate and final Tensor objects are strongly retained until the consumer
join and checked again there. Their references are removed by the existing
metadata release path. A rejected builder now also retains a detached snapshot
of its actual returned positions, improving diagnosis without granting proof.

CPU contracts use actual PyTorch dispatch and source-pinned fixture functions.
They cover two slices, version 0→6 attachment, an unrelated later same-storage
slice, cloned middle/final objects, one or three slices, changed second-slice
indices, changed metadata on the retained intermediate, source writes, report
isolation, cleanup retries and two successive inference-mode requests. These
contracts establish only the adapter behavior, not the natural NPU row map.

This Python increment can use the fresh ND flat-storage SDK bundle because it
does not change its Native sources. It still requires a new installed package
and source snapshot. After root reviews both commits and the fresh SDK/device-free
load, the existing public Core C1 probe may attempt the seven bounded B8 phases.
Use `providers_quant_prepare_inventory_case.py --group-abi --single-site-c1`
with `providers_quant_c1_probe.py` and the newly reviewed bundle, not source61's
diagnostic-only bundle. Host collection stays explicit through `collect_replay()`.

Actual C1 remains false until that run proves two healthy censuses, current row
proof at admission and consumer, genuine reference/lossless/F/OFF and complete
same-Worker cleanup. C2/C3 remain false, all 135 structural candidates stay in
the census and unknown Attention/MoE edges remain unverified. E5 remains a
separate assessment. No SDK, container, SSH or NPU operation was run by this agent.
