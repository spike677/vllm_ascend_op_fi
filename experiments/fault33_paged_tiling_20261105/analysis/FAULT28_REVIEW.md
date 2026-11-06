# 故障 28 的证据范围与故障 33 的对照设计

**故障 28 的实测结论是，固定完整逻辑 Q/K/V，在相同 B8 上改变合法的原生 FA host 策略，可以产生重复稳定的数值差异。** 原生算子调用、宿主最终 tiling、运行日志、完整输出和独立 msprobe 首差各有证据。历史 FIA 图模式中的具体触发因素，以及某条 Reduce 指令的独立贡献，仍需针对相应路径取证。[算子因果验收](../../../artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029/reports/causal_reports/execution_acceptance.json)

| 问题 | 已有对照和结果 | 证据能支持的结论 |
| --- | --- | --- |
| 同一 B8、同一份 Q/K/V，切换 M 是否影响输出？ | B 实测 M128/R8，F 实测 M64/R8。目标 4,194,304 个 FP16 元素中有 27 个变化，分布于 23 个 query，最大绝对差 0.0001220703125。两次独立进程重复一致 | M 选择及其合法派生的整套执行计划影响结果。最早变化的逻辑 query 为 413，末 query 2047 在该对照中逐 bit 相同 |
| 只改变 batch，固定策略，是否也发生同样差异？ | 将同一目标请求复制为 B8，各个固定 M/R 策略下 B1 与 B8 的完整目标输出都逐 bit 一致 | 此 replay 中，差异需要策略干预。自动模式由 C++ 读取实际 B 来选策略，证明了受控的 batch→策略链 |
| M、R 和实际在线分块各有什么关系？ | R 固定时，M 改变 BMM/Vec/Softmax 的部分几何。M 固定于 128 时，R8→4 令末 query 的 key 分组从 2×1,024 变为 4×512，有 83,176 个元素变化 | 两种受控因素均会影响计算路径。在这两个实验中，M-only 与 R-only 对合法 key 分组的作用不同 |

原生路径为 `npu_fusion_attention_v3 → aclnnFlashAttentionScore → S1S2 host → kernel`，replay 输入来自 Qwen3 第 1 层。B8/R8 的 M-only 还伴随 BMM2 的 depthA1 从 4 变为 8、stepKa 从 2 变为 8，部分 query 的 masked 计算范围及向量行几何也变。源码可重建的执行计划和输出差分，支持对 M 选择及这些派生参数整体所作的归因。把 27 个差异唯一归到某条 Reduce 或某次特定的舍入，还需要指令或更细粒度隔离证据。[主因素摘要](../../../artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029/reports/causal_reports/primary_causal_a3_summary.json) · [受控 batch 策略记录](../../../artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029/reports/causal_reports/secondary_batch_policy_a3.json) · [实现与依赖字段说明](../../fault28_causal_20261029/README.md)

**算子级 msprobe 定位。** 在相同 B8、固定 R8 的 B/F 中，原生统计分析在 `NPU.npu_fusion_attention_v3.0.forward` 的主 Attention 输出 Mean 观察到差异。输入可观测统计相同，输入逐元素一致性另由完整字节校验。该实验使用任务私有的 FP32 high 统计修正。报告保留 null 和辅助返回值，主输出未见差异的对照仍可能有 seed/offset 变化。[原生 B/F 路径索引](../../../artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029/reports/independent_statistics/automatic_a3/candidate_batch_m_b8_bf_index.json) · [统计方法及限制](../../../artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029.md)

**全模型验证了另一个干预。** HF Qwen3-0.6B 在 28 层使用 `every_match`，策略为 M128/R8 → M64/R4 的联合切换。B1/B8 五轮共 560 次 SDPA，280 次 prefill 中有 112 次符合受控条件，实际切换 112/112 次，覆盖 504 个层/请求映射对。另 280 次 Q1/KV2049 decode 的几何在证书范围外。两次 F 的原生统计首差都出现在第 0 层 prefill 的框架 SDPA 输出 Mean，两次 fault 重复稳定，回到 OFF 后输出与 logits 恢复。这份报告验证 joint 策略的模型传播；M-only 的数值量级和末 query 行为应引用上面的算子级实测。[全模型联合报告](../../../artifacts/native_execution_strategy/fa_model_coverage_statistics_20261031/reports/acceptance.json) · [逐层和逐请求结果](../../fault28_model_coverage_20261030/README.md)

**历史 FIA 的对应证据。** 已归档的 vLLM stock 校准在 FP16、TND、eager 的 1,024-token prefill 比较 B1/B8，每组两个重复。原日志关联到实际 FIA 选择和 launch，四组均为同一 key，但完整 raw 缺失，且第 0 层跨 batch 的整段目标 Q、K 在进入 FIA 前已不同。历史图模式的服务调度、完整同输入和微观归约路径尚未由这份校准重现。[原始 FIA 校准与缺失字段](../../../analysis/fault28_tiling_check_20261029/native_fia_calibration/README.md)

**对故障 33 的直接要求。** 算子输入还多了一层页表语义。先按各自的 block table 反查全部有效 K/V，再判断目标请求的逻辑输入是否相同。跨物理页大小的输出差分报告分页条件效应；固定物理缓存、页表与请求后再改变内部 SInner，才可把效应收窄到受控计算分块及其原生依赖。两阶段分别记录原生选择/launch、每个固定计划的重复结果以及原 API 输出。要验证线上调度链，还需补同请求的真实调度轨迹。[故障 33 试验和验收条件](../README.md)
