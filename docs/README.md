# DemonCATPFI 文档入口

本目录是项目的 canonical 文档入口。实现、native runtime、原始 trace、msprobe dump、模型响应和压缩归档不迁入本目录；它们仍由各 capability、experiment 或 evidence 目录保管。

## 推荐阅读顺序

1. [项目概览](overview/project_overview.md)：仓库定位、边界和当前状态。
2. [术语](overview/terminology.md)：统一区分语义故障、执行平面、能力、证据与代理。
3. [故障效果评估核心指标](evaluation/fault_effect_evaluation_core_metrics_zh.md)：如何解释一个已验证 Fault 的实际效果。
4. [故障效果评估设计 V1](evaluation/fault_effect_evaluation_design_v1.md)：baseline/fault/recovery 的工程协议和证据要求。
5. [Fault 索引](faults/README.md)：Fault15、20、26、39、41 的 canonical 文档入口。
6. [模型索引](models/README.md)：当前模型的注入面与效果报告。
7. [历史归档](archive/README.md)：阶段审计、探查与失败/替代路线的定位目录。

## 证据与文档的边界

文档用于说明事实、范围、状态和证据位置。以下内容不应放入 `docs/`：

```text
JSONL、msprobe dump、raw response、trace、gzip/binary、模型权重、容器产物
```

这些数据仍位于诸如 `fault20_fast_validation/evidence/`、`fault20_fast_validation/reports/`、`fault20_fast_validation/msprobe_dump/` 或受控的远端结果目录。每个 topic README 会给出 canonical evidence pointer；本轮不改变任何 evidence 文件或历史实验结论。

## 文档来源与迁移

- [文档盘点 V1](document_inventory_v1.md)：四个来源的只读 Markdown 清单与 SHA-256 去重信息。
- [迁移映射](document_migration_manifest.md)：本轮新增/重定位 canonical 文档的旧路径、来源和理由。

`C:\Users\admin\Documents\demoncatpfi` 仅是 `LEGACY_READONLY_SOURCE`。它可用于历史查阅，不是 canonical 文档依赖或开发入口。
