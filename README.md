# DemonCATPFI：故障识别、注入与效果评估

正式源码入口：`D:\project\demoncatpfi`，打开 [demoncatpfi.code-workspace](demoncatpfi.code-workspace)。**Windows 用于源码阅读维护；真实推理、注入和服务管理仍运行于 Linux／Ascend。**

当前整合分支 `codex/fake-requant-usability-v1`：手选／显式全选、参与 rank 展开、单次有损后结束实例与离线评价增强。见 [本轮收束记录](docs/releases/fake_requant_usability_v1/README.md) 和 [变更说明](CHANGELOG.md)。旧标签 `source-baseline-v1-20260911` 保留；这不是一键安装包或 Windows NPU 运行版。

## 从哪里开始

1. [手选／全选操作手册](docs/usage/fake_requant_manual_all_targets_v1_zh.md)：查看目标 → 明确选择 → 预览 → IDENTITY → 一次有损 → 报告 → 完整停止实例。启动准备与历史同进程 B/F/R 见 [Qwen3-0.6B 单卡手册](docs/inspect/qwen06b_single_npu_user_walkthrough_zh.md)。
2. [产物合同](docs/reference/artifact_io_contracts_zh.md)：输入什么文件、哪些旧计划不能复用。
3. [算法与生命周期](docs/architecture/algorithms_zh.md)：实际怎样识别、绑定和扰动。
4. [评估核心逻辑](docs/evaluation/fault_effect_evaluation_core_metrics_zh.md)、[指标字段字典](docs/evaluation/fake_requant_tool_v1_metrics_zh.md)。
5. [源码地图](src/README.md)、[真实命令目录](tools/README.md)、[依赖与开发](docs/development/workspace_and_dependencies_zh.md)。
6. [全部文档](docs/README.md)、[历史故障](docs/faults/README.md)、[证据保存](artifacts/README.md)。

## 当前认证边界

**本轮新增功能仅离线验收，未启动模型或占用 NPU。** 多 worker 选择／控制面测试不等于 Ascend 多 rank 实测；异常热卸载仍后置，停止实例不冒充热恢复。数值 NMSE/SQNR 来自有界采样；检测回归与独立人工标签质量验证分开。D 概率适配为离线能力，TargetPlan 概率事务仍 `LIVE_UNVERIFIED`。

此前源码基线的历史实测：Qwen3-0.6B，28层、112个真实浮点输入目标、单NPU/TP=1：同epoch IDENTITY及三道官方MT-Bench首轮原题一个B/F/R周期已完成，三题数值变化、输出分叉并精确恢复。配置为参考FakeRequant `tensor_absmax ×20`，不改模型权重，不是模型转换W8A8。

这是三个开放问题的功能验收；没有官方MT-Bench分数、该配置PPL、准确率损伤或N≥3稳定性结论。`target_call_window=PASS`，独立scheduler级 `full_request_window=UNPROVEN`，统一结果UNKNOWN/PARTIAL不提升为全功能PASS。

[历史Qwen32B/R1](docs/models/qwen/README.md) 与本次结果分开。[DeepSeek/DPSK](docs/models/deepseek/README.md) 仅有只读规则，下一步迁移，未实机认证。

## 使用边界

私有开发服务、显式资源锁、独占请求、故障默认关闭。新服务重新绑定和IDENTITY，异常／清理失败／请求完成无法证明时停止。health 200不等于恢复成功，现有异常处理限制见算法页。

必需[参考依赖](experiments/fake_requant_reference/README.md)已固定纳入本仓库，不再隐式依赖研究工作树。其他故障源码仍保留原目录，不在本轮迁移或升级成熟度。
