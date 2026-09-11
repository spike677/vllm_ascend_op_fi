# 文档阅读地图

当前默认演示是Qwen3-0.6B单卡。本页只导航，不复制公式、字段定义或完整报告。

| 目的 | 入口 | 类型 |
| --- | --- | --- |
| 项目与边界 | [根目录](../README.md)、[概览](overview/project_overview.md) | 总览 |
| 启动到结果 | [单卡手册](inspect/qwen06b_single_npu_user_walkthrough_zh.md) | 操作手册 |
| 手选／显式全选／一次有损后停止 | [新操作入口](usage/fake_requant_manual_all_targets_v1_zh.md)、[选择合同](reference/target_selection_contract_v1_zh.md) | 操作与参考分开 |
| 识别／绑定／扰动／清理 | [算法](architecture/algorithms_zh.md)、[源码](../src/README.md) | 实现参考 |
| 文件给谁、何时重建 | [产物合同](reference/artifact_io_contracts_zh.md) | 接口参考 |
| 怎样解释效果 | [核心评估逻辑](evaluation/fault_effect_evaluation_core_metrics_zh.md) | 汇报与评审 |
| 分母、阈值、空值 | [指标字典](evaluation/fake_requant_tool_v1_metrics_zh.md) | 唯一指标定义 |
| 采样 NMSE/SQNR 与检测审计 | [A/B 离线增强](evaluation/usability_numerical_detector_audit_zh.md) | 派生定义与验证边界 |
| 概率参考序列 | [D 离线适配](evaluation/reference_probability_offline_v1_zh.md) | 不代表现场概率事务 |
| 新旧 hook 与恢复命名 | [源码核对](architecture/hook_mechanism_compatibility_v1_zh.md) | 工程风险不冒充已复现 bug |
| 完整工程规范 | [Evaluation设计](evaluation/fault_effect_evaluation_design_v1.md) | 工程设计 |
| 命令与依赖 | [开发指南](development/workspace_and_dependencies_zh.md)、[CLI](../tools/README.md) | 开发手册 |
| 源码与实测绑定 | [本轮离线候选](releases/fake_requant_usability_v1/README.md)、[历史基线](releases/source_baseline_v1/README.md) | 发布记录 |

专题：[Inspect](inspect/README.md)、[Evaluation](evaluation/README.md)、[模型](models/README.md)、[Fault15/20/26/39/41](faults/README.md)。

历史：[归档](archive/README.md)、[迁移映射](document_migration_manifest.md)、[旧盘点](document_inventory_v1.md)、[早期工具V1](releases/fake_requant_tool_v1/README.md)。历史快照不改为当前统计。

证据：[保存规则](../artifacts/README.md)。`fault20_fast_validation/reports/`、`evidence/`、`msprobe_dump/`未迁移改写。原始response/trace、部署配置和密钥不混入docs。手册讲如何做，参考定义合同，报告讲某次发生什么，原始证据支持报告，四者不能替代。
