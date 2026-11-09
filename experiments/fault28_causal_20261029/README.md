# 故障 28：固定 Q/K/V 的原生 FA 因果验证

**2026-10-29 已在 124 物理卡 5（910B4、TP1）完成 32 次正式采集，受控 Batch→真实 host 策略→FA 数值差异及独立 msprobe 首差验证通过。** 本轮是原生算子 replay，输入来自真实 Qwen3 layer1；没有执行新的模型 forward，也没有证明历史 FIA 缺陷或唯一 Reduce 指令根因。

[原始 statistics/mix 证据与验收](../../artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029.md)。此前[真实 HF 模型传播结果](../fault28_native_tiling_20261029/README.md)对应 M/nRatio 联合策略，不能自动变成此次 M-only 的模型传播结论。

## 这次确认了什么

固定原生策略时，B1/B8 的完整目标输出逐 bit 一致；原生 stock 没有在这个严格等值输入上暴露默认 batch 不一致。注入模式在 C++ 中读取实际 batch：B1 保持 base，B8 选择 M-only、ratio-only 或联合策略。B8 同 batch 的 B/F 对照确认差异来自实际策略选择，而不是上游 Q/K/V 改变。

完整目标为 `[16,2048,128]`，4,194,304 个 FP16 元素。下表 B1、B8 的结果相同，两次独立进程重复稳定；差值用精确 `2^-24` 整数单位计算。

| 干预 | 变化元素 | 受影响 query | 最大绝对差 | query2047 变化元素 |
| --- | ---: | ---: | ---: | ---: |
| M128→64，固定 R8 | 27 | 23 | 0.0001220703125 | 0 |
| R8→4，固定 M128 | 83,176 | 1,536 | 0.00048828125 | 34 |
| 两者联合 vs base | 83,181 | 1,539 | 0.00048828125 | 34 |
| M128→64，固定 R4 | 25 | 23 | 0.0001220703125 | 0 |
| R8→4，固定 M64 | 83,174 | 1,536 | 0.00048828125 | 34 |
| `joint−m_only−r_only+base` 交互 | 42 | 33 | 0.0001220703125 | 0 |

M-only 最早受影响的 query 是 413；R-only 从 512 开始。**只看最后一个 query 会漏掉真实 M 效应。** 逻辑坐标的先后不等于设备指令执行顺序。

预注册的十个 FP64 参考 query 覆盖主要边界，但没有覆盖 M 的 27 处变化；M 与 base 的这十点参考误差指标相同。R/joint 的 RMS 在部分 query 更小、部分更大，MaxAbs 不变。因此不能把数值不一致统称为精度恶化或业务质量下降。

## 实际注入边界和调用链

```text
controller 的空闲门禁与源码冻结
→ vendor_exec 在导入框架前加载并校验任务私有 vendor
→ replay：真实捕获的 Q/K/V，B8 为同序列八份副本
→ torch_npu.npu_fusion_attention_v3
→ aclnnFlashAttentionScore
→ S1S2 host 原生选择点：SelectM / SelectNRatio
→ 原生生成 BMM、Softmax、core、workspace 和完整 tiling
→ 未改动的真实 kernel 执行
→ 原始返回值、独立完整输出检查及 msprobe statistics/mix
```

同一私有 host 二进制执行所有策略。B/F 不改输入、mask、scale、精度或输出，不在 Python 重算 Attention。M/R 改动发生在原生 selector，后续依赖 tiling 由真实 host 生成。所有实际 launch 的 raw 与公共 context raw 完整一致，205 个 POD 字段经 C++/Python 解码一致；没有用请求标签替代实际策略。所有合法 causal keys 恰好参与一次，没有跨核 SplitKV。

| 位置 | 作用 |
| --- | --- |
| [fault28_causal_policy.h](source_policy/fault28_causal_policy.h#L38) | C++ 根据实际 batch 解析三个受控策略 |
| [Eligible / scope](source_policy/fault28_causal_policy.h#L96) | PID、TID、phase、batch、token、形状和原生属性守卫 |
| [SelectM / SelectNRatio](source_policy/fault28_causal_policy.h#L133) | 选择合法候选值；未匹配时保留原生值 |
| [apply_policy.py](source_policy/apply_policy.py#L41) | 插入真实 host selector 边界和公共 raw 证据 |
| [replay.py](replay.py#L296) | 真实 ReplayModule 子模块、一次原生 API 调用和完整输入/输出观察 |
| [verify_causal_matrix.py](analysis/verify_causal_matrix.py#L327) | 完整序列因素/交互及重复稳定性分析 |
| [compare_replay_statistics.py](analysis/compare_replay_statistics.py#L228) | 独立原生统计首差分析，不读 host 命中或私有 tensor 指定位置 |

selector 每次匹配都选择策略，没有全局 once 限制。本轮每个 case 是独立进程中的一次调用，不能称为模型 token 步数或长期持续运行验收。

M-only 同时改变 BMM 的 M 几何，R8 时 BMM2 的 depthA1 由 4→8、stepKa 由 2→8；部分 query 的 computed masked 范围、动态 Vec/Softmax 几何也变化，合法 key 分组不变。R-only 改变 BMM 的 N/Ka 和在线分组（末 query 为 2×1024→4×512）。这证明 M/R 干预及其合法依赖执行路径能各自产生差异，**不能唯一归因某条 Reduce 指令**。源码派生计划不是硬件浮点指令轨迹。

## 预注册与真实采集

| 阶段 | 组合 | 调用数 |
| --- | --- | ---: |
| stock bridge | B1/B8，各重复两次 | 4 |
| 主矩阵 | B1/B8 × base(128,8)、m_only(64,8)、r_only(128,4)、joint(64,4)，各重复两次 | 16 |
| 受控 batch 策略 | B1/B8 × batch_m、batch_r、batch_joint，各重复两次 | 12 |

先证明 stock B8 完整目标输出与旧模型捕获一致，再证明同 batch 的 private base 与 stock 完整输出/实际 raw 相同，最后运行预注册对照；没有按数值结果选择输入或删掉单元格。三种 batch 模式的 B1 均为实际 M128/R8，B8 分别实际为 64/8、128/4、64/4，且逐字节匹配主矩阵对应输出。

软件版本为 torch 2.10.0、torch-npu 2.10.0.post4、CANN 9.1.0；官方 ops-transformer v9.1.0 commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`。同一 vendor 的 670 文件完整核 SHA，kernel 源保持不变。当前支持固定 FP16/BNSD、Q16/KV8/D128、S2048、causal、keep_prob1、inner_precise0、B1/B8 的特定 S1S2 分支，尚非泛用 FA/FIA 能力。构建与精确 scope 见 [BUILD.md](source_policy/BUILD.md)。

## msprobe 能独立推断到哪里

两侧均为 `task=statistics`、`level=mix`、`summary_mode=statistics`、`precision=high`。**high 使用已披露的任务内 FP32 统计 collector 修正，并非未修改官方实现。** 原 SHA `e981aa6f…`，修改后 `3aa7ede0…`，stage manifest `743354fe…`；模型和张量源 dtype/shape 不改。此前默认半精度统计漏掉 FA 首差的失败结论继续保留。

38 对独立统计比较均完成对齐。自动策略 B8 六对均在 `NPU.npu_fusion_attention_v3.0.forward` 输出观察到主 Attention Mean 差异，输入可观测统计未差；Max/Min/Norm 未分辨这些变化。B1 六对未观测到主输出差异。候选主 B/F 是 B8 的 base r0/r1 与 batch_m r0/r1，保留原生单调用 step0/rank0 三件套、配置、collector 披露和完成回执。

每对 20 项不可观测统计原样保留；null 不等于相等。seed/offset 在相同主输出的重复对照中也会变化，不能把辅助返回的首差称为 Attention 主输出故障。M-only 的 softmax_sum 完整字节有差而四统计未分辨，也说明摘要匹配不证明张量相等。

这与“真实策略变化先影响 FA 输出”的预期一致。msprobe 只定位到 API 数值边界，不能单独证明输入逐元素相同、目标请求具体坐标、物理故障点或 tiling 根因；这些由独立完整哈希、真实 raw 和私有逐元素因果检查支撑。详见[统计分析说明](analysis/compare_replay_statistics.README.md)。

## 保留的失败与交付范围

- `stock_bridge_a1` 在占卡门禁退出，零 NPU 调用；外部服务始终保留。
- `stock_bridge_a2` 的一个真实 FA 已返回，但 msprobe 跳过根模块，缺少 Module 记录，采集失败。v2 将原 ReplayModule 注册为子模块后，正式 32 例全部同版重采。旧代码、原始失败记录和静态注册检查保留，未补造节点或改写失败回执。
- 首次主分析误将跨 batch 的六个 `InputParamsPH` 对齐字节当作语义变化。独立源码核查证明该字段无计算读取且 reset 不初始化；仅修正跨 batch 语义比较，同 batch 严格比较和完整 raw 绑定不变。原失败报告、旧分析器、源码依据和九项针对性 CPU 回归全部保留。

主交付保留原生统计 JSON、完整性回执、无损压缩的原生日志、源码/构建/collector 指纹和独立报告；不包含完整 Q/K/V、输出 `.bin/.pt`、模型权重或编译库。完整逐元素重算需要 124 上的私有输出，不能只凭公开四统计重算计数。

本轮达到的是**固定场景下受控 batch 策略的原生算子因果及统计现象验证**。该日期实验未覆盖的 M-only 模型/logits 传播，已由[2026-11-09 的独立完整模型补验](../fault28_m_only_model_20261109/README.md)闭合；该补验没有把 joint 结果转算为 M 的作用。历史 FIA 默认缺陷、唯一 Reduce 根因、更多模型形状和业务影响仍需分别验收。
