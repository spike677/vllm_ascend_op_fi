# 真实 Worker 正向注入闭环（2026-11-17）

从 `dcbef7a027bbc39e7796f12880305c3fa516c1f8` 的未通过项继续，本轮已将
**Fault20 WeightNz、Fault39 MatMul**推进到标准 vLLM Worker 的限定单点注入、
完整 Qwen3-0.6B 模型输出变化和 OFF 精确恢复。两条都取得原始 statistics/mix
的独立首差。17条没有全部通过，也未开放公共 `all_eligible`。

| 机制 | 独立首个可观测差异 | B → F | 恢复 |
|---|---|---|---|
| Fault20 | `Functional.linear.112.forward/output/0` 的 Max | 17.625 → 16.875（约4.2553%） | 完整logits精确恢复，首token未变 |
| Fault39 | `Functional.linear.112.forward/output/0` 的 Max | 17.625 → +Inf | 完整logits精确恢复，token 576→0→576 |

固定比较公式为 `abs(F-B) > 1e-8 + 1e-5*abs(B)`。比较器仅读取原始三件套、
配置和采集器身份；不读取命中或完整张量指定首差。Fault20首差前6620个、
Fault39首差前6128个不可观测字段保留，不能把null当成相同。

两个实际部署均为124卡5、910B4/A2、CANN8.5.1、vLLM-Ascend0.18.0，标准流程
创建临时Worker，使用安装后的wheel。Extension不创建Worker、不替换核心执行方法。
Fault20使用明确披露的controlled PTA/NZ部署，不能称为默认ND/PTA路径已通过。
完整CPU回归556项通过；CPU与Native/NPU资格分开记账。

GLM-4-9B-0414公开预训练权重已按固定revision完整校验并实测；真正DeepSeek-V2-Lite
量化预训练模型也已实测。两者均为2次各4token，完整logits有限且重复一致；
这些是正常推理进展，**不是它们的Native故障验收**。SmolLM2与QMM的本轮范围见最终报告。

- [完整验收、机制和范围](FINAL_REPORT.md)
- [真实Serving Replay接口与生命周期](../../docs/worker_extension/host_driven_replay.md)
- [Fault20独立首差](summaries/fault20-independent-statistics.json)
- [Fault39独立首差](summaries/fault39-independent-statistics.json)
- [失败账](FAILURE_LEDGER.md) · [全部尝试摘要](summaries/attempts.json)
- [17条机制增量状态](mechanism_delta.json)
- [CPU回执](summaries/cpu-validation.json) · [Native构建摘要](summaries/builds.json)
- [校验清单](checksums.json) · [后续接续](RESUME.md)

按用户要求，GitHub只发布实现、总结和哈希；**不提交新原始statistics/mix、
完整张量、模型权重或vendor库**。原件在 `/home/worker-positive-20261117`。
公开摘要可审查结论与方法，逐条重算仍需访问124原件。
