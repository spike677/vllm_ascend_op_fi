# API and configuration reference

`import demoncat`; `demoncat.configure(mechanism, config)` returns an owner-scoped session.
`python -m demoncat catalog|example|validate|inspect|generate --check` exports and checks definitions.
All session operations return `demoncat.operation_result.v1` with ok/code/state/owner/generation.
See ../docs/api.md for typed Request, lifecycle and Worker usage.

| Mechanism | Parameters | Schema |
| --- | --- | --- |
| 8.restore_offset | offset, mapping_policy | schemas/mechanisms/f008.restore_offset.schema.json |
| 15.alias_detach | storage_format, view_shape, stride, storage_offset | schemas/mechanisms/f015.alias_detach.schema.json |
| 17.spatial_padding | padding, kernel_spatial | schemas/mechanisms/f017.spatial_padding.schema.json |
| 18.api_half | gain | schemas/mechanisms/f018.api_half.schema.json |
| 20.buffer_substitute | source_slot, replacement_transform | schemas/mechanisms/f020.buffer_substitute.schema.json |
| 22.wrong_row | source_row, target_row | schemas/mechanisms/f022.wrong_row.schema.json |
| 25.reduce_scatter | split, replay_delta | schemas/mechanisms/f025.reduce_scatter.schema.json |
| 26.residual_stomp | writer, destination, consumer | schemas/mechanisms/f026.residual_stomp.schema.json |
| 26.fia_destination | writer, destination, consumer | schemas/mechanisms/f026.fia_destination.schema.json |
| quant.batch_shared_scale | quant_scope, scale_policy, projection_scope | schemas/mechanisms/fquant.batch_shared_scale.schema.json |
| 28.fa.legal_tiling | m_tile, s2_tile | schemas/mechanisms/f028.fa.legal_tiling.schema.json |
| 28.fia.legal_tiling | m_tile, s2_tile | schemas/mechanisms/f028.fia.legal_tiling.schema.json |
| 28.fia.denominator_replay | s2_block, replay_previous_mass | schemas/mechanisms/f028.fia.denominator_replay.schema.json |
| 33.paged_decode | page_size, s_inner, cache_geometry | schemas/mechanisms/f033.paged_decode.schema.json |
| 39.matmul_range | input_range, substitution | schemas/mechanisms/f039.matmul_range.schema.json |
| 39.output_proxy | output_transform, max_live_leases | schemas/mechanisms/f039.output_proxy.schema.json |
| 41.tail_column | cache_kind, tail_elements, slot | schemas/mechanisms/f041.tail_column.schema.json |
