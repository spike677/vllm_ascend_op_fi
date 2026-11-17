# P3 / P4：独立诊断与推理链路闭环

本阶段接续 `d1c8a20c8d3f832b911011628e0901766be36cea` 的 Fault20/39 真实 Worker 单点验收。
只发布实现、总结和哈希；原始 statistics/mix、完整张量、权重和 vendor 构建留在124。

## P3 已完成：保留严格失败的第二层诊断

对原 Fault15/17/18 的 `step0–2/rank0` 重新计算，三组原严格报告与历史文件完全一致：
均为 `INCOMPLETE_COMPARISON`，各 step 的调用结构不齐。严格全局首差通过仍为 **0/3**。
新诊断不覆盖这些结果，也不供应全局首差。没有新增 NPU 调用或预训练模型验收。

规则在读取数据前冻结于 [p3_rules.json](p3_rules.json)。仅使用原始 dump、stack、construct、配置与采集器身份；
不读取 Native 命中、注入日志或完整张量。固定 `rtol=1e-5, atol=1e-8`，baseline 为参考。

| 机制 | 每 step 的 B/F / 配对记录数 | 诊断所得 | 不能推出的结论 |
|---|---|---|---|
| 15 alias_detach | 11 / 12 / 11 | F 新增 format_cast；equal.3 输出 true→false，equal.4 false→true；equal.3 输入 Min −32→0 | 原 base 的 add 消费者四项统计均为 null；统计不能独立证明 as_strided 别名断裂 |
| 17 spatial_padding | 3 / 3 / 3 | conv3d padding 两维 0→7；其输出 Max 0.5635518431663513→0.24107059836387634 | 调用位置相同但参数改变，不能声称同参卷积精度异常 |
| 18 api_half | 3 / 5 / 3 | 两次新增 ×0.5 均保留；add 输出 Max 3.410391092300415→1.7051955461502075 | 原 RMSNorm hook 在返回代理之前；原 RMSNorm Max/Min 未变不表示代理未发生，Mean/Norm 的 null 也不是相等 |

调用身份要求算子族、原始解析调用栈、construct 项一致且两侧各唯一；共享锚点顺序必须一致。
重复来源、缺失 stack/construct、顺序冲突均拒绝配对。null construct 被披露，不能证明存储或消费者身份。
非张量输入变化与标量输出变化分别报告；相同参数描述不证明完整输入张量相等。
每条未配对记录保留前后锚点、原因及自身统计，不过滤新增调用。
完整私有 ledger 覆盖每条 B/F 记录，公开摘要保存其哈希。

| 报告 | 原严格报告复算一致 | 新报告用途 |
|---|---|---|
| [Fault15](summaries/p3_fault15.json) | 是 | 观察边界和不可观测项 |
| [Fault17](summaries/p3_fault17.json) | 是 | 同调用身份与参数变化分离 |
| [Fault18](summaries/p3_fault18.json) | 是 | 新增调用及返回代理窗口 |

实现：[诊断库](../../fault39/src/demoncat_fault39/diagnostics.py)、[CLI](../../fault39/tools/diagnose_call_sequence.py)。
原 `compare_msprobe.py` / `compare_model_capture.py` 未改。15项新诊断测试与9项历史诊断测试通过；
全套571项 CPU 测试通过、无跳过，见 [回执](summaries/cpu_p3.json)。CPU 结果不是 NPU 验收。

124 的冻结源码及私有详细报告在 `/home/p3p4-closure-20261117/p3-source-a1` 和 `p3-{15,17,18}-a1`。
[冻结清单](summaries/p3_source_manifest.json) 与各报告包含源码及原始三件套哈希。
原始历史失败、三组原件和旧比较器均保留。

## P4 实施顺序与当前边界

1. 26 FIA：真实推理 producer→writer→consumer→恢复，接入现有 Worker 生命周期，不替换核心执行方法。
2. 26 residual：先证明写前 NaN 的分配和初始化来源，保留不可比较结果。
3. batch_shared_scale：真实请求到 row 的映射，以及逐行 scale/Q 证据。
4. 33：分页/KV 对应、连续生成、实际命中和 OFF 恢复，独立报告首差。

当前仍在准备 P4；历史 Provider 的通过结果不自动升级为本次公共 Worker 验收。
GLM-4-9B-0414 和真实 DeepSeek-V2-Lite 的正常预训练 NPU 推理已在上一阶段通过；尚无本阶段故障验收。
全部17条仍未取得公共 `all_eligible` 资格。使用一张空闲卡、共享锁及每 case 门禁，不停止外部任务。
