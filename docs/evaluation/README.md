# 故障效果评估

本目录描述“一个已通过 lossless validation 的 Fault 如何被评估”，而不是实现某个 native mutation。

- [Distortion Evaluation V1](distortion_evaluation_v1.md)：主模式，固定 Probe/Detector、同题稳定性、恢复与矩阵界面。
- [正常输入用例库与双卡 live](normal_diagnostic_cases_live_v1_zh.md)：V2 对话协议、clean 校准、真实 R1 请求级证据与恢复。
- [评分解释修正与完整原文审计](distortion_scoring_audit_v2_zh.md)：区分预算未完成、实际重复和恢复范围，含版本化原始数据重放。

- [核心指标说明](fault_effect_evaluation_core_metrics_zh.md)：面向汇报、评审和新开发者的简明解释。
- [效果评估设计 V1](fault_effect_evaluation_design_v1.md)：输入、状态机、指标公式、multi-rank、trace policy 与交付物。

Distortion 是可见退化验收；Analysis 是补充解释。PPL/NLL、RDR/VDR、L0–L6 保留，但不决定 Robust Visible Distortion PASS。

模型/故障专属的真实结果在 `docs/models/` 和 `docs/faults/`；原始 evidence 保持在各自 reports/evidence 路径。
