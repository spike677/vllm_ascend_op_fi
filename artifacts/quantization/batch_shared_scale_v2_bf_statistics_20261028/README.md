# Batch 共享尺度 v2：已完成的 baseline / fault 统计证据

本目录公开此前已完成的主动故障注入实验的 **JSON 证据子集**，用于审阅 baseline / fault 的 msprobe 记录与首差异定位。数据来自 2026-10-27 的正式采集，没有重新运行 NPU，也没有重新采集另一套纯统计摘要数据。

原采集配置为 `task=statistics`、`level=mix`、`summary_mode=md5`、`tensor_list=["."]`。原数据同时保存统计值、校验摘要和可保存的完整张量。本目录保留原生 JSON 的全部字节，但不包含 `.pt`、其他张量二进制或模型权重；完整原始包私有保留，未随本次证据提交公开。原 JSON 中的 `data_name`、`dump_data_dir` 仍指向历史张量位置，这些引用不是本目录提供的下载链接。

`statistics` 是采集任务，`mix` 表示模块与 API 混合层级；它们不等于 `summary_mode=statistics`。本次公开不能描述成“已重新采集并验收仅有 Max/Min/Mean/Norm、无张量的纯 statistics 版本”。原生 dump 确实含统计字段；部分 dtype 的 Mean/Norm 为 null，应按原值解读。此安装版本中名为 `md5` 的字段实际使用 CRC32，不能作为密码学摘要；本目录文件完整性另用 SHA-256。

## 运行映射

| 目录 | 原始 run | 原阶段 | 真实量化输入行数 | 条件名 |
|---|---|---|---:|---|
| [baseline/rows2_run002](baseline/rows2_run002) | run002 | H，健康私有后端 | 2 | letter_counts_B2 |
| [fault/rows2_run004](fault/rows2_run004) | run004 | F，双 rank 注入 | 2 | letter_counts_B2 |
| [baseline/rows8_run012](baseline/rows8_run012) | run012 | H，健康私有后端 | 8 | even_count_B4 |
| [fault/rows8_run014](fault/rows8_run014) | run014 | F，双 rank 注入 | 8 | even_count_B4 |

这里将健康 H 组作为 baseline；不将它改称原矩阵中的 B 阶段。`even_count_B4` 是历史条件名，实际选中调用为 8 行，不能从名字推断成 4 行。每组均含 rank0 / rank1 的 `dump.json`、`stack.json`、`construct.json`，以及实际 `config_rank0.json` / `config_rank1.json`。每 rank 113 个观测节点，窗口为模型第 30–32 层。

原始 [completed.json](records/completed.json) 和 [execution.json](records/execution.json) 记录整个历史矩阵的 17 次运行完成、退出码为 0；本目录只提供上述 4 次运行的 B/F 观测。单次完成信息另见 [derived_completion_receipts.json](records/derived_completion_receipts.json)：它是明确标记的派生回执，保留源 result.json 的 SHA-256、完成状态、形状与后端命中计数，省略 prompts、token IDs、生成文本和 logprobs。它不是改写后的“原生 result.json”。

## 原始报告和结论

- [2 行官方比较](reports/official/pair002_004)与[8 行官方比较](reports/official/pair012_014)：安装版本的 `msprobe compare -da` 原始输出，两 rank 均报告 `NPU.npu_dynamic_quant.127.forward` 为候选。
- [独立分析报告](reports/independent)：四份原始报告在各自 113 个节点中发现量化输出为首差异，输入未发现差异，前面 72 个节点未发现差异；父模块为 `model.layers.31.mlp.down_proj`。
- [独立验收摘要](reports/independent/summary.json)和[官方验收摘要](reports/official/acceptance_summary.json)记录历史完整矩阵 PASS。摘要及其报告哈希清单还引用本目录未选择公开的重复、恢复和其他 rank 对照；不能据本目录声称那些原始观测也已公开。

故障方式是让私有 ACLNN 后端把一次 DynamicQuant 的内部归约描述符从 `[T,H]` 改为 `[1,T*H]`，使当前 rank 的各行共享 batch 极值尺度。原生计算写出 Q，单个 scale 再发布到外部各行，原消费者继续使用这对 Q/scale。它模拟主动引入的归约范围错误，不是证明健康量化后端会被任意大 batch 或异常 prompt 自动触发。

这些报告的逐元素差异数量、数值公式拟合和消费者 pair 检查来自当时的完整张量证据。本目录可以审阅原生统计/摘要、调用栈与既有报告，但缺少 `.pt`，不能独立重算报告内全部逐元素指标，也不能把摘要相等直接当作严格逐元素相等。首差异定位限于观测窗口和 API 计算边界，不能据此断言某条设备指令或生产事故的唯一根因。

## 来源与本地校验

[selection.json](selection.json) 列出每个公开原文件的私有包路径、公开路径、长度和源 SHA-256；这些原生文件和报告均逐字节恢复，未重写字段。[SHA256.json](SHA256.json) 覆盖本目录的其他文件。公开内容经过检查，无模型权重、张量二进制、远端登录入口或凭证；调用栈保留通用容器源码路径。

[source/](source) 是采集时冻结的 7 个核心实现文件及原始 SHA 清单，保留历史字节以便审查故障实现。它依赖未随本目录公开的运行输入、环境和模型，不是开箱即用的部署包。

无需 torch / NPU，可在本目录执行：

```bash
python restore_subset.py --verify
python verify_publication.py
```

`verify_publication.py` 还核对 4 组运行映射、8 份采集配置、24 个原生 JSON、派生完成回执，以及官方/独立报告中的首差异边界。它的 PASS 表示公开子集完整且既有记录相互一致，不是重新执行原始张量验收。官方原始报告包含 NaN 字面量，按原字节保留，验证器使用 Python 标准库的兼容解析。

持有私有原包者可将同一白名单恢复到新目录；工具核验归档、分块和原文件 SHA，不导出白名单以外的内容：

```bash
python restore_subset.py --package PRIVATE_MODEL_PACKAGE --output NEW_DIRECTORY
```
