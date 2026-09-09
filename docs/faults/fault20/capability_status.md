# Fault20 能力与验证状态

canonical capability 以原始 capability matrix 与 inventory 为准：

- [`fault20_capability_matrix_final.md`](../../../fault20_fast_validation/reports/fault20_capability_matrix_final.md)
- [`fault20_injection_capability_inventory_20260807.md`](../../../fault20_fast_validation/reports/fault20_injection_capability_inventory_20260807.md)
- [`fault20_v17_msprobe_response_summary.md`](../../../fault20_fast_validation/evidence/fault20_v17_msprobe_response_summary.md)

可由这些 source 支持的结论包括：真实 QMM target/metadata 路径、lossless temporary substitution、受控 INT4/INT2-equivalent 输入退化、single/multi/all-hit 的受限选择与多 rank 证据。具体版本、模型、shape、rank、CANN/vLLM 与 proof policy 是能力的一部分；未知 binding、ABI/metadata/stream 失败或 developer-mode 不满足时必须 fail closed。

本轮文档整理不重跑模型、不重新生成 evidence，也不改变任何 Fault20 native algorithm、控制 ABI 或 bundle。
