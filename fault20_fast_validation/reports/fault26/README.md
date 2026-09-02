# Fault26 Reports

Fault26 is the residual stomp validation line.

Current status: PASS for the controlled proxy-induced FIA wrong-destination
residual stomp path.

Relevant top-level reports currently preserved for compatibility:

- `../fault26_workspace_stomp_readonly_probe_plan.md`
- `../fault26_qmm_triggered_residual_stomp_20260817.md`
- `../fault26_qmm_residual_physical_adjacency_audit_20260818.md`
- `../fault26_residual_writer_timeline_audit_20260818.md`
- `../fault26_residual_lifetime_visible_api_audit_20260818.md`
- `../fault26_fia_writer_lossless_phase_c_20260818.md`
- `../fault26_fia_wrong_destination_residual_stomp_20260818.md`
- `../fault26_writer_allocation_topology_and_qmm_underflow_20260818.md`
- `../fault26_qmm_underflow_bounded_multihit_20260818.md`
- `../fault26_qmm_underflow_all_eligible_20260818.md`

Relevant source:

- `../../source/fault26_fia_writer_proxy_v1/`
- `../../source/fault26_readonly_trace_v1/`

Relevant evidence:

- `../../evidence/fault26_evidence_20260818.tar.gz`
- `../../evidence/fault26_sustained_dump_20260902/`

Boundary:

- The current PASS is for a gated proxy-induced wrong-destination residual
  stomp used to validate diagnosis and propagation behavior.
- It is not proof of a spontaneous natural OOB write in a healthy vendor kernel.
