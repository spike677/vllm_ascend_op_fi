# 正式统计和逐请求哈希的本地复查

`review_formal_summary.py` 是额外只读复查工具，不替换已冻结的独立统计分析器或原生执行验收器。
它先从原始 `dump/stack/construct` 重新计算全部 20 个 phase 对比，逐字段核对已生成的
独立报告；之后才打开 call 元数据，核对逐请求输出哈希、重复运行和关闭后的恢复。
它不会用哈希、策略命中日志或 native profile 决定 msprobe 首差。

## 下载范围

每个成功 attempt 保留 `completion.json`。建议同时下载 `source_manifest.json`、
`selected_cases.json` 和完整源码快照，供外部来源复核；本脚本不替代执行源码及 vendor 验收。
每个 B1/B8 case 保留以下文件及原相对路径：

- `result.json`、`config.json`、`receipt.json`、`collector_provenance.json`。
- `model_config.json`、`module_inventory.json`、`module_events.jsonl`、`phase_manifest.json`。
- 10 份 `phases/stepN/request_mapping.json`。
- 30 份 `observation/stepN/rank0/{dump,stack,construct}.json`。
- `result.calls` 引用的 280 份 `native_calls/call_NNNN/call.json`。
- framework 报告、statistics 的 `summary.json` 和 10 份 phase 报告。

两个 case 共 560 份 call 元数据。可额外下载 census 与联合 acceptance 报告。
无需下载私有 logits/activation 二进制、模型权重、vendor `.so` 或全部大体量日志来运行此工具。
未下载完整原生日志时，不能声称在本地重新验过全日志 launch 覆盖。

## 命令

准备本地 suite，沿用联合验收器的 schema，每个路径指向本地文件；相对路径相对于 suite：

```json
{
  "schema": "nes.model_coverage.acceptance_inputs.v1",
  "cases": [
    {
      "case_dir": "outputs/formal_a1/main_b1",
      "framework_report": "reports/main_b1/framework.json",
      "statistics_report": "reports/main_b1/statistics/summary.json"
    },
    {
      "case_dir": "outputs/formal_a1/main_b8",
      "framework_report": "reports/main_b8/framework.json",
      "statistics_report": "reports/main_b8/statistics/summary.json"
    }
  ]
}
```

`completion` 默认是 case 的上级目录下 `completion.json`，也可在 suite 中显式指定。
本地 suite 不需要填写不存在的远端 census/log 路径。

```bash
python -B analysis/review_formal_summary.py --inputs local_suite.json \
  --acceptance reports/acceptance.json --output reports/local_formal_review.json
```

`--acceptance` 可省略。提供时仅验证联合报告引用同一批 completion、framework 和统计报告，
不把报告本身当作重新核验原生日志的替代。输出必须是全新文件；Windows 长路径自动加前缀。

## 结果口径

本地复查 PASS 要求完整 B1/B8 矩阵、原始 SHA 与报告一致、Session 比较源栈可对齐，
并要求 calibration/native、两次受控运行、native/off 的输入、输出和 logits 哈希恢复或重复一致。
校准与 Session 的栈不同可以保留不可对齐，不强制制造统计对照。

数值变化与模型影响没有预设成功要求。报告分别保留原始统计首差、null、后层输入传播、
逐请求哈希变化、统计碰撞，以及统计变化但输出哈希相等的情况。整批统计不能推导每个请求
各自的统计差异。完整 logits 二进制未读取，因此这里只核对封存哈希，不重新做逐字节比较。

本工具永远不宣称 native 执行重新验收、真实事故复现或模型准确率下降。
CPU fixture 报告 `review_formal_summary_cpu_a2.json` 包含 10 项检查，不属于 NPU 实机证据。
