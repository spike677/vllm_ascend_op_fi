# Inspect → FakeRequant 配置探索 V1（预运行契约）

本实验不替换 R1，不改变量化公式，不更改模型原生 scale 或 Fault20 native 算法。
采用已冻结的官方 MT-Bench 首轮八题（149、82、83、86、88、141、143、146），原输入、模板、seed=42、temperature=0、max_tokens=1024 均保持不变。
这不是完整 MT-Bench；开放题不计算 Accuracy，也不使用 LLM judge。

## 窗口审计

参考 runtime 的 `max_hits` 在一次 arm 内累计。arm 清空 candidate/hit/history；达到上限后 enabled 仍为 true，但 selector 拒绝后续目标调用。`request_generation` 是配置与证据标签，不在 hook 中用于识别请求。旧 149 题的 4096/65536=6.25% 仅是目标调用覆盖率；不能除以 64 冒充 token 位置。

新实验复用 VerifiedOwner 与既有服务会话管理器，只增加显式配置参数。请求隔离依赖项目独占租约、独立的临时生成 API key、串行请求，以及引擎 running/waiting=0 和成功请求计数恰好增加 1。它不声称隔离同机其他 root 管理员的主动操作。超时后不能当作请求结束；无法证明结束即停止矩阵并恢复独立实验服务。

完整窗口仍使用正整数硬上限：`模块数 × (1024 + 最大 prompt tokens + 16) × 2`。当前 down 集合预期为 141952；它不是剂量目标，更不是无界 all-hit。只有各目标模块 candidate=hit=hook_seen、无 budget/其他拒绝、两个 rank 符合策略且请求结束得到证明时，才标记 `REQUEST_WINDOW_COVERAGE=PASS`。达到硬上限即 `WINDOW_INCOMPLETE`，不自动重置或扩大。

## 固定阶段

1. A0 保留前一轮八题 R1 原始证据；A1 为 down/0.25/全请求。
2. A1 覆盖与恢复通过后，B 串行测试 down 的倍率 1、2、5、10、20。
3. B 按“上下文支持的异常题数最多；并列取较小倍率”选值。没有异常则按预先指定的 20 做覆盖压力探索，不能称其为有效病态候选。
4. C 按 live Inspect 发现的 down → down+gate_up → 再加 attention output → 再加 QKV 扩展。每个新集合必须重新采样、绑定、精确核对模块及 disabled/IDENTITY/recovery；禁止父子重复匹配。
5. 最终集合补 0.25 全窗口对照。
6. 完成矩阵后，如有明确规则支持的候选，按支持题数最多、并列取先执行配置选定；追加两个完整八题 B/F/R 周期，与搜索周期合计 N=3。没有候选不重复填数，不无限升级。

任一服务不健康、机制失败或全文/token pieces 恢复不精确均停止升级，保留失败结果。每请求 finally disarm，每轮 remove。单请求超时 3600 秒。

## 有界观测，不改变数值路径

新的 opt-in `ExplorationWorkerExtension` 直接调用 SHA 固定的原 `_requant_tensor`，原样返回其结果；原 combined entrypoint 不变。只将 history sink 换为有界聚合：所有调用的命中、changed、total、clamp 真正改变码值的 clip、误差按模块累计，另保留全局首末各 16 个事件。

归零和相对 L2 额外诊断在每模块第 1、65、129…次已接受调用上采样。差与平方和用 FP32，分母明确为采样区域，不能写成全量精确值。`changed_elements` 是累计 activation 元素事件，不是独立权重数。未采样有效码值个数时直接记为 NOT_SAMPLED。

验收器显式识别 `BOUNDED_EFFECT_SUMMARY_V1`、交叉核对 runtime per-module 计数，不伪造逐 hit 历史；旧 R1 逐事件验收分支保留。新的统计源文件和验收源文件 hash 与旧评分源码分别记录。

## 输出分类和候选标准

原 detector 阈值保持冻结。分别保存生成轨迹差异、原规则告警、格式上下文复核结果、尚未评估的语义质量。只有 baseline-relative repetition、lexical collapse、punctuation 或 garbage 规则支持的模式可参与候选选择；Markdown 表格/水平分隔线不确认刷屏，length-only 不确认病态。此“确认”仍局限于可计算输出模式，不是语义正确性判决。

报告保留 EvaluationResult、全部原文、token pieces（token IDs=null）、首次分叉位置、逐模块剂量和恢复记录。未通过完整矩阵或 N=3，不宣称矩阵完成或重复性已证明。
