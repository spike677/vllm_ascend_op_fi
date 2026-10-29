# 故障 28：固定 Q/K/V 的原生 FA 因果矩阵与统计证据

2026-10-29 在 124 物理卡5（910B4，TP1）完成32次正式原生 FA replay：stock桥接4、主因素矩阵16、受控 batch策略12。输入来自真实Qwen3 layer1；没有新的模型forward。验收状态为 `PASS_CONTROLLED_BATCH_NATIVE_FA_CAUSAL_AND_STATISTICS`。

[实现、调用链、源码位置与限制](../../experiments/fault28_causal_20261029/README.md) · [验收清单](fault28_causal_matrix_statistics_20261029/reports/causal_reports/execution_acceptance.json) · [完整主分析](fault28_causal_matrix_statistics_20261029/reports/causal_reports/primary_causal_a3_review2.json) · [精简数值摘要](fault28_causal_matrix_statistics_20261029/reports/causal_reports/primary_causal_a3_summary.json) · [实际自动策略链](fault28_causal_matrix_statistics_20261029/reports/causal_reports/secondary_batch_policy_a3.json)

| 相对base的变化 | 完整b0变化元素 | query数 | 最大绝对差 |
| --- | ---: | ---: | ---: |
| M128→64，R8不变 | 27 | 23 | 0.0001220703125 |
| R8→4，M128不变 | 83,176 | 1,536 | 0.00048828125 |
| M/R联合 | 83,181 | 1,539 | 0.00048828125 |

每个目标为4,194,304个FP16元素，B1/B8结果相同且重复稳定；各固定profile跨batch完整b0逐bit相同。M差异最早query413，最后query2047无M差异，仅看末token会漏检。FP64参考误差有升有降，未证明质量下降。

## 主 B/F 与独立统计

主B/F选择**相同B8、只切换M的条件策略**，各保留两次独立进程；B1保护和其余M/R对照完整保留。原生每case只有step0/rank0，不能称为模型token步。

| arm | 实际策略 | 原生统计repeat0 / repeat1 |
| --- | --- | --- |
| B | M128/R8，base | [r0 dump](fault28_causal_matrix_statistics_20261029/attempts/private_base_a3/base_b8_r0/observation/step0/rank0/dump.json) / [r1 dump](fault28_causal_matrix_statistics_20261029/attempts/private_base_a3/base_b8_r1/observation/step0/rank0/dump.json) |
| F | M64/R8，C++实际batch选择 | [r0 dump](fault28_causal_matrix_statistics_20261029/attempts/automatic_a_a3/batch_m_b8_r0/observation/step0/rank0/dump.json) / [r1 dump](fault28_causal_matrix_statistics_20261029/attempts/automatic_a_a3/batch_m_b8_r1/observation/step0/rank0/dump.json) |

每个dump同目录有原生stack/construct，上级case有原配置、collector披露、receipt/result；attempt有完成回执、资源/源码绑定。[B/F来源索引](fault28_causal_matrix_statistics_20261029/reports/independent_statistics/automatic_a3/candidate_batch_m_b8_bf_index.json)及[12对自动策略统计汇总](fault28_causal_matrix_statistics_20261029/reports/independent_statistics/automatic_a3/summary.json)均保留原始SHA。

配置为 `task=statistics, level=mix, summary_mode=statistics, precision=high`。**high依赖披露的任务内FP32统计修正，不是未修改官方collector。** [补丁与清单](fault28_causal_matrix_statistics_20261029/collector/)保留实际源码；collector原SHA `e981aa6f…`、修改后 `3aa7ede0…`、stage manifest `743354fe…`。此前官方默认半精度统计漏检FA首差的失败结果仍在旧证据包。

全部38对独立比较完成。自动策略B8六对输入可观测统计未差，主输出Mean有差，首个API统计差异为 `NPU.npu_fusion_attention_v3.0.forward` 输出；B1六对主输出未观测差异。每对20项不可观测统计保留。seed/offset在相同主输出的对照中也会变，辅助返回变化不能当作主Attention故障。

统计分析只读原生dump/stack/construct和配置，以回执做完整性门槛；不读取host命中或私有完整tensor指定首差。它定位API边界，不能单独证明逐元素相等、具体请求坐标、物理故障点或tiling根因。精确输入/完整输出和实际native raw由另一路证据校验。

## 原始性、失败与可复核范围

manifest覆盖538个文件、82,484,729字节（不含manifest本身）。32份原始采集及回执保持字节不变；完整原生日志仅无损gzip压缩，并同时绑定压缩前SHA/大小。包中无完整Q/K/V、输出.bin/.pt、权重或编译库。

`diagnostics/`保留占卡门禁失败和首次缺Module采集的原始失败JSON；`revisions/`保留v1与v2注册修复。主分析首次误判未使用的六个POD对齐字节的失败报告、旧分析器、源码依据及针对回归也保留；只修比较器语义分类，未改任何raw数据。

私有完整目标输出仍在124，因此**不能仅从公开四统计重新计算逐元素计数**。公开包可重验原生统计推断、raw/源码/回执绑定及完整性；全量因果重算另需私有输入/输出和匹配源码。受控策略不等于SDK默认batch策略；M还改变BMM/Vec/Softmax依赖路径，未唯一识别Reduce指令，也未复现历史FIA或新M-only模型传播。

```sh
python experiments/fault28_causal_20261029/analysis/package_causal_evidence.py verify \
  --root artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029
```

Windows的深目录可用绝对长路径前缀，Git可配置 `core.longpaths=true`。打包器 `mechanism_accepted=false` 表示它只做完整性检查；科学范围以独立报告和execution_acceptance为准。

manifest SHA256：`db24066e8265bf44b457030e935b2c9dc8a8787dee92312601a0bb52159df7c4`。验收清单 SHA256：`bff19faaedfe384604023f57d2ac3f4729040c8f66188ccd97e068de1c4e4846`。
