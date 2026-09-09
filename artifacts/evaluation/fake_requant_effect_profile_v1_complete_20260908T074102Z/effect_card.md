# Fault Effect Card

- 评估：`fake_requant.effect_evaluation_v1_complete_20260908T074102Z`
- 能力：`python.fake_requant.activation`
- 当前分级：`L4_REPEATABLE_TASK_DAMAGE`

| 维度 | 指标 | 结果 |
| --- | --- | --- |
| Mechanism | 注入机制通过 | `True` |
| Task | Paired Damage | `3` / `28` (`0.10714285714285714`) |
| Probability | PPL Ratio / NLL Delta | `1.0655228418807177` / `11.995000297072238` |
| Output | Response Divergence Rate (RDR) | `0.9473684210526315` |
| Visible | Visible Degradation Rate (VDR) | `0.05263157894736842` (3) |
| Phenotype | pathological count | `0` |

## 证据边界

- RDR 表示生成轨迹敏感度；只有 baseline-relative 的可解释退化才计入 VDR。
- `DIVERGED_BUT_VALID` 不计入 VDR，也不等同于任务能力下降。
- Recovery 是实验有效性门禁，不作为故障强度指标。

## 分级依据

- 完整 cycle：`3`；可重复任务损伤 cycle：`3`。
- 恢复精确匹配 cycle：`2`；恢复率：`0.6666666666666666`。
- L4 由跨 cycle 的 Paired Damage 证据触发；它与总体 VDR 阈值是不同的、可并存的判据。
