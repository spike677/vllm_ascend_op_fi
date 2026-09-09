# Qwen 双卡 FakeRequant 效果报告 V1

> 适用范围：私有 `Qwen3-32B-w8a8-nopdmix`、TP=2、冻结的 FakeRequant R1 配置。本文不重新运行模型，不将结论推广到其他模型、环境或配置。

## 结论摘要

当前配置已经在三轮完整 baseline → fault → recovery cycle 中证明：机制命中稳定、teacher-forced 概率质量稳定退化、短输出和长输出生成轨迹广泛分叉。固定 GSM8K-10 冒烟集尚未重复证明稳定任务准确率损伤；故障也没有表现为重复刷屏、早停、失控生成或服务崩溃。

## Mechanism

| 项目 | 结果 |
| --- | --- |
| 有效 fault transaction | 66 / 66 |
| rank 0 | 达到每请求 4096 次命中预算，且记录元素变化 |
| rank 1 | 每次均为 0 hit，拒绝原因均为预期的 `rank_filter` |
| 结论 | 这是实际数值修改的稳定证据，不只是 hook 已安装 |

## Probability effect

三个 teacher-forced 周期均对相同 189 个 reference token 评分，结果一致：

| 指标 | Baseline | Fault | 变化 |
| --- | ---: | ---: | ---: |
| NLL | 544.7059 | 556.7009 | +11.9950 |
| PPL | 17.8507 | 19.0203 | 1.0655× |

该结果支持当前冻结配置下的可重复概率质量下降。PPL 来自 reference-token log probability，不是 hidden-state 或自由生成文本统计。

## Output effect 与 phenotype

| 指标 | Fault 相对 Baseline | Recovery 相对 Baseline |
| --- | --- | --- |
| 短输出 probes | 每周期 3 / 4 response 分叉 | 精确一致 |
| 长输出 probes | 每周期 5 / 5 response 分叉 | 精确一致 |
| 重复、标点刷屏、早停、runaway、empty、明显乱码 | 未见 fault 诱发 | — |
| worker/service crash | 未观察到 | — |

支持的表述是“生成轨迹分叉”，不是“病态输出崩溃”。

## Task effect 与 recovery 边界

| Cycle | Baseline | Fault | Recovery |
| --- | ---: | ---: | ---: |
| 1 | 10 / 10 | 9 / 10 | 9 / 10 |
| 2 | 9 / 10 | 9 / 10 | 9 / 10 |
| 3 | 9 / 10 | 9 / 10 | 9 / 10 |

同一 clean 样本存在自然波动，因此不能把一次 `10/10 → 9/10` 写成“FakeRequant 导致 GSM8K 下降 10%”。短/长输出和概率测量均精确恢复；完整 recovery gate 因 GSM8K baseline 波动保留 `PARTIAL`。

## Evidence pointer

```text
/home/haoran/demoncat_overnight_20260907/evaluation/
fake_requant_effect_evaluation_v1_complete_20260908T074102Z/live_validation
```

该路径包含三个 cycle 的真实结果。本文仅总结其支持的证据边界；不将 raw trace、response 或压缩归档复制进 `docs/`。
