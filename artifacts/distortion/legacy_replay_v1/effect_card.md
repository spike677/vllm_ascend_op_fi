# 故障可见效果卡

## 注入了多少、在哪里注入

- Capability：`python.fake_requant.activation`；策略：`REFERENCE_REQUANT`。
- 模块族：`MLP_DOWN`；层：64/64；rank：[0]/TP2。
- Strength：`{'scale_multiplier': 0.25, 'scale_provider': 'tensor_absmax', 'eps': 1e-12, 'fixed_scale': None}`。
- Hit policy：`{'max_hits': 4096, 'interval': 1, 'scope': 'PER_REQUEST_REARM', 'short_request_padding': "When a metric request does not naturally exhaust R1's immutable 4096-hit budget, issue one labelled fixed padding completion under the same arm and record measurement versus transaction hit counts separately."}`。
- Mechanism：**PASS**，66/66 个原始请求通过。

## 用户实际看到了什么

- 数据集：`LEGACY_FAKE_REQUANT_EVAL_SET`，不是伪装的新 16 条探针。
- 完整周期：3。
- 稳定失常：**4/19**；样本：`['gsm8k-test-0002', 'long_explanation', 'long_reasoning', 'punctuation_sensitive']`。
- 稳定 phenotype：`{'UNRESOLVED': 1, 'REPETITION': 3}`。
- Crash：0。
- Recovery：12/12 个故障引起的失常 pair 恢复；gate=PASS。
- **ROBUST_VISIBLE_DISTORTION_FAIL**；未通过条件：`['frozen_16_probe_suite']`。

输出轨迹变化不等于病态退化。N 个周期必须是同一道题稳定失常；不将不同题的数量相同视为可重复。
本结果是固定 smoke 集与规则证据，不是统计显著性或总体模型性能结论。只检测字符级乱码；不推断语义不连贯。

## 文件

- [逐项矩阵](effect_dashboard.html)
- [完整评测结果](evaluation_result.json)
- [只包含有效失常 pair 的日志](distortion_hits.jsonl)
- [原始来源与校验信息](replay_manifest.json)

PPL/NLL、旧 RDR/VDR/L0–L6 保留在 Dashboard 的 Analysis Details；不决定本页 PASS。
