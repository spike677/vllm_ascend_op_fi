# 故障效果评估

本目录描述“一个已通过 lossless validation 的 Fault 如何被评估”，而不是实现某个 native mutation。

- [核心指标说明](fault_effect_evaluation_core_metrics_zh.md)：面向汇报、评审和新开发者的简明解释。
- [效果评估设计 V1](fault_effect_evaluation_design_v1.md)：输入、状态机、指标公式、multi-rank、trace policy 与交付物。

模型/故障专属的真实结果在 `docs/models/` 和 `docs/faults/`；原始 evidence 保持在各自 reports/evidence 路径。
