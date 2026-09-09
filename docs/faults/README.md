# Fault 文档索引

每个 Fault 目录只保留三类 canonical 阅读文档：`README`（范围和状态）、`design/architecture`（语义与实现边界）、`validation/capability status`（已验证结论和限制）。阶段探查、handoff 与一次性审计由 [archive](../archive/README.md) 的 catalog 指向原始 reports 路径。

| Fault | 当前入口 |
| --- | --- |
| Fault15 | [scale granularity](fault15/README.md) |
| Fault20 | [QMM 输入数值退化与 executor-cache 路线](fault20/README.md) |
| Fault26 | [受控 residual stomp proxy](fault26/README.md) |
| Fault39 | [浮点 MatMul non-finite 路线与 output proxy](fault39/README.md) |
| Fault41 | [state/cache tail corruption 的文档状态](fault41/README.md) |

这些页面均不替代原始 evidence；能力 status 必须以各页列出的 source report、manifest 和环境边界为准。
