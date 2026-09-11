# DemonCATPFI：故障识别、注入与效果评估

正式源码入口：`D:\project\demoncatpfi`，打开 [demoncatpfi.code-workspace](demoncatpfi.code-workspace)。**Windows 用于源码阅读维护；真实推理、注入和服务管理仍运行于 Linux／Ascend。**

当前整合分支 `codex/source-baseline-v1`，发布标签 `source-baseline-v1-20260911`。见 [发布记录](docs/releases/source_baseline_v1/README.md) 和 [变更说明](CHANGELOG.md)。这是源码基线，不是一键安装包或 Windows NPU 运行版。

## 从哪里开始

1. [Qwen3-0.6B 单卡新用户手册](docs/inspect/qwen06b_single_npu_user_walkthrough_zh.md)：启动 → Inspect → 真实输入绑定 → TargetPlan → IDENTITY → B/F/R → Dashboard → 清理。
2. [产物合同](docs/reference/artifact_io_contracts_zh.md)：输入什么文件、哪些旧计划不能复用。
3. [算法与生命周期](docs/architecture/algorithms_zh.md)：实际怎样识别、绑定和扰动。
4. [评估核心逻辑](docs/evaluation/fault_effect_evaluation_core_metrics_zh.md)、[指标字段字典](docs/evaluation/fake_requant_tool_v1_metrics_zh.md)。
5. [源码地图](src/README.md)、[真实命令目录](tools/README.md)、[依赖与开发](docs/development/workspace_and_dependencies_zh.md)。
6. [全部文档](docs/README.md)、[历史故障](docs/faults/README.md)、[证据保存](artifacts/README.md)。

## 当前认证边界

Qwen3-0.6B，28层、112个真实浮点输入目标、单NPU/TP=1：同epoch IDENTITY及三道官方MT-Bench首轮原题一个B/F/R周期已完成，三题数值变化、输出分叉并精确恢复。配置为参考FakeRequant `tensor_absmax ×20`，不改模型权重，不是模型转换W8A8。

这是三个开放问题的功能验收；没有官方MT-Bench分数、该配置PPL、准确率损伤或N≥3稳定性结论。`target_call_window=PASS`，独立scheduler级 `full_request_window=UNPROVEN`，统一结果UNKNOWN/PARTIAL不提升为全功能PASS。

[历史Qwen32B/R1](docs/models/qwen/README.md) 与本次结果分开。[DeepSeek/DPSK](docs/models/deepseek/README.md) 仅有只读规则，下一步迁移，未实机认证。

## 使用边界

私有开发服务、显式资源锁、独占请求、故障默认关闭。新服务重新绑定和IDENTITY，异常／清理失败／请求完成无法证明时停止。health 200不等于恢复成功，现有异常处理限制见算法页。

必需[参考依赖](experiments/fake_requant_reference/README.md)已固定纳入本仓库，不再隐式依赖研究工作树。其他故障源码仍保留原目录，不在本轮迁移或升级成熟度。
