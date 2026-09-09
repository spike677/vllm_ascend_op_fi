# Fault20：QMM 输入数值退化与 L4 executor-cache 路线

Fault20 的已验证工作聚焦在开发者模式下、对真实量化矩阵乘执行路径进行受控输入数值退化/temporary substitution，并用 rank-aware、baseline/fault/recovery 证据支持诊断工具开发。

- [架构与故障语义](architecture.md)
- [能力与验证状态](capability_status.md)
- [历史探查目录](../../archive/fault20/README.md)

原始 runtime、构建脚本、reports 和 evidence 仍在 [`fault20_fast_validation/`](../../../fault20_fast_validation/)。本目录不复制 `.so`、trace、dump 或 response。
