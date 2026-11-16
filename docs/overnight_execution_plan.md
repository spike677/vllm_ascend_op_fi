# 17 条机制工程整理与执行记录

本轮已按用户计划执行，起点是实际 fetch 核对的
`bfdd6abc1adb5d08b034db7a96ec97ffbec0b9bf`，工作分支为
`codex/overnight-17-fault-closure-20261116`。原 Worker 分支及其他可写工作区不作重置。

## 执行约束

- 复用既有模型发现、`attach`、`NativeSession`、资格门禁和同一个 Worker Extension；不建第二套生产 Registry/Controller/CLI。
- 用户另行允许通过 vLLM 标准启动流程创建独立临时验收 Worker。扩展不管理 Worker 生命周期，也未修改 vLLM/vLLM-Ascend 源码或替换 `execute_model`。
- 每次 NPU case 使用既有资源锁和空闲门禁，一批不超过 8 个 case；本轮未终止外部任务、修改 Docker 入口或恢复旧隔断。
- 比较固定 `abs(F-B) > 1e-8 + 1e-5*abs(B)`；先只读统计定位，再关联运行时证据。结构不齐、null、NaN 不改成通过。
- 新采集、完整运行回执、临时数组及 vendor 构建留在 124；Git 只放实现、总结、路径和哈希。

## 实际推进

1. 基线与工作区盘点，三条并行工作线分别负责 Native/Worker、模型环境适配、17 条能力和证据审计。
2. 六种实际 Transformers 小模型完成 CPU 结构、逻辑角色、prefill/cache decode、透明路由/OFF 验证；四个镜像的 CANN 安装与 ELF 只读复核。
3. 修复 Worker 恢复未完成时跨机制/跨租户继续操作的缺口，保留中断后的 owner 与隔离状态。
4. 15、17、18 在卡5完成构造 NPU 推理模块 B/F/OFF 和各三个统计 step；17 修复采集器与包装器的安装顺序后重试。
5. 真实 Qwen3 TP1 Worker 完成20/39拒绝、隔离、恢复；WeightNz 只读发现未命中。32B W8A8 TP2 观测到 QMM，但跨轮稳定目标未成立。
6. 对其余机制开展源码、历史及接口探索；不把缺少适配的机制强行注入。每项实际探索与未执行部分见[台账](../experiments/overnight_20261116/exploration_ledger.json)。
7. 445 项 Python 回归、124 无设备 C++/ctypes 检查、三包安装验证及最终报告。

三次探索是结构性问题的上限，不要求每条机制人为启动三轮 NPU。
资源等待和配置失败分别保留，健康门禁失败的用例没有继续进入故障组。

## 结果入口

[最终报告](fault_closure_report.md) · [17 条矩阵](../experiments/overnight_20261116/fault_matrix.json) ·
[覆盖资格](../experiments/overnight_20261116/coverage_matrix.json) ·
[未闭合事项](../experiments/overnight_20261116/blockers.md)
