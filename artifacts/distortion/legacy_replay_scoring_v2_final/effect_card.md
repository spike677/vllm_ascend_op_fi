# 故障可见效果卡

## 注入了多少、在哪里注入

- Capability：`python.fake_requant.activation`；策略：`REFERENCE_REQUANT`。
- 模块族：`MLP_DOWN`；层：64/64；rank：[0]/TP2。
- Strength：`{'scale_multiplier': 0.25, 'scale_provider': 'tensor_absmax', 'eps': 1e-12, 'fixed_scale': None}`。
- Hit policy：`{'max_hits': 4096, 'interval': 1, 'scope': 'PER_REQUEST_REARM', 'short_request_padding': "When a metric request does not naturally exhaust R1's immutable 4096-hit budget, issue one labelled fixed padding completion under the same arm and record measurement versus transaction hit counts separately."}`。
- Mechanism：**PASS**，66/66 个原始请求通过。

## 用户实际看到了什么

- 数据集：`LEGACY_FAKE_REQUANT_EVAL_SET`；版本/hash 见 manifest。
- 完整周期：3。
- 稳定失常：**5/19**；样本：`['gsm8k-test-0001', 'gsm8k-test-0004', 'long_explanation', 'long_reasoning', 'punctuation_sensitive']`。
- 稳定 phenotype：`{'LEXICAL_COLLAPSE': 1, 'PUNCTUATION_SPAM': 1, 'REPETITION': 5}`。
- Crash：0。
- 失常表型恢复：15/15；该分母不是全测试集。
- 全量恢复 Gate：FAIL，同时核对全用例任务/validator 状态及 runtime。
- **ROBUST_VISIBLE_DISTORTION_FAIL**；未通过条件：`['frozen_16_probe_suite', 'recovery']`。

## 补充任务的独立结论

`{'status': 'NOT_APPLICABLE_LEGACY_REPLAY'}`

旧 robust breadth gate 仍保留；本轮可交付用例不要求产生两种病态。未命中、无数值变化和证据未知保留在完整矩阵，不伪装为“无效果”。

## 各周期恢复范围

```json
{
  "cycle_01": {
    "fault_induced_phenotype": {
      "recovered": 5,
      "denominator": 5
    },
    "all_text_exact": {
      "matched": 17,
      "denominator": 19
    },
    "all_task_validator": {
      "matched": 13,
      "denominator": 14,
      "mismatch_ids": [
        "gsm8k-test-0008"
      ],
      "not_applicable_ids": [
        "long_explanation",
        "long_reasoning",
        "numbered_long_list",
        "punctuation_sensitive",
        "repetition_sensitive"
      ]
    },
    "runtime_clean_health": {
      "status": "PASS",
      "reason": "raw worker query after remove + recorded health",
      "pids": [
        541466,
        541467
      ]
    },
    "same_process": true
  },
  "cycle_02": {
    "fault_induced_phenotype": {
      "recovered": 5,
      "denominator": 5
    },
    "all_text_exact": {
      "matched": 19,
      "denominator": 19
    },
    "all_task_validator": {
      "matched": 14,
      "denominator": 14,
      "mismatch_ids": [],
      "not_applicable_ids": [
        "long_explanation",
        "long_reasoning",
        "numbered_long_list",
        "punctuation_sensitive",
        "repetition_sensitive"
      ]
    },
    "runtime_clean_health": {
      "status": "PASS",
      "reason": "raw worker query after remove + recorded health",
      "pids": [
        541466,
        541467
      ]
    },
    "same_process": true
  },
  "cycle_03": {
    "fault_induced_phenotype": {
      "recovered": 5,
      "denominator": 5
    },
    "all_text_exact": {
      "matched": 19,
      "denominator": 19
    },
    "all_task_validator": {
      "matched": 14,
      "denominator": 14,
      "mismatch_ids": [],
      "not_applicable_ids": [
        "long_explanation",
        "long_reasoning",
        "numbered_long_list",
        "punctuation_sensitive",
        "repetition_sensitive"
      ]
    },
    "runtime_clean_health": {
      "status": "PASS",
      "reason": "raw worker query after remove + recorded health",
      "pids": [
        541466,
        541467
      ]
    },
    "same_process": true
  }
}
```

Scorer：`DISTORTION_SCORER_V2_TASK_TERMINATION_PATHOLOGY`。任务结果、生成终止、输出病态和机制归因分别记录；仅 unresolved 不计为明显病态。

输出轨迹变化不等于病态退化。N 个周期必须是同一道题稳定失常；不将不同题的数量相同视为可重复。
本结果是固定 smoke 集与规则证据，不是统计显著性或总体模型性能结论。只检测字符级乱码；不推断语义不连贯。

## 文件

- [逐项矩阵](effect_dashboard.html)
- [完整评测结果](evaluation_result.json)
- [只包含有效失常 pair 的日志](distortion_hits.jsonl)
- [原始来源与校验信息](replay_manifest.json)

PPL/NLL、旧 RDR/VDR/L0–L6 保留在 Dashboard 的 Analysis Details；不决定本页 PASS。
