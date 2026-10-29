# 故障 28：stock FIA 路径校准 a5

**四组 stock 采集均已完成；没有进行故障注入，四份原生 profile 仍为 `INCOMPLETE_PROFILE`。** 这些结果不能证明“相同注意力输入下 FIA 首先出错”。目标序列在单请求与八请求合批时，进入第 0 层 FIA 之前已经存在 Q/K 差异。[计算摘要](summary.json)、[controller 完成记录](raw/completion.json)。

2026-10-29 在 124 物理卡 5 上使用 Qwen3-0.6B、TP1、vLLM 0.23.0 / vLLM-Ascend 0.23.0、Torch 2.10.0 / torch-npu 2.10.0.post4。原 checkpoint 为 BF16，原生 `dtype=half` 加载为 FP16。四组均使用 eager、关闭 prefix caching / chunked prefill / async scheduling，并设置 `ACLNN_CACHE_LIMIT=0`；没有 `LD_PRELOAD` 或私有 OPP。目标为相同 1024-token prompt 的最后一个 prefill 位置 1023；B8 的七个同行都是目标 token 序列的重复副本。每组均由新模型进程完成一次完整 prefill，调度记录包含实际 request/row/token/logits 映射。[B1 配置](raw/source/setup/cases_1024/stock_b1.json)、[B8 配置](raw/source/setup/cases_1024/stock_b8.json)、[B8 调度记录](raw/stock_b8/scheduler.json)。

| 校准条件 | 原生 FIA 调用 | 已记录 key | blocks / schedule | profile |
|---|---:|---|---|---|
| stock_b1 | 28 | 5000000000000200103 | 20 / 1 | INCOMPLETE_PROFILE |
| stock_b1_repeat | 28 | 5000000000000200103 | 20 / 1 | INCOMPLETE_PROFILE |
| stock_b8 | 28 | 5000000000000200103 | 20 / 1 | INCOMPLETE_PROFILE |
| stock_b8_repeat | 28 | 5000000000000200103 | 20 / 1 | INCOMPLETE_PROFILE |

每个 API 窗口都记录到一次 `NnopbaseExecutorDoTiling` 选择和匹配的 `NnopbaseExecutorLaunchKernel` launch；op name、key、进程、blocks、schedule 相符。选择线程匹配调用线程，launch 可能来自另一个 dispatch 线程。这里沿用 parser 的窗口关联边界，未将其升级为完整的原生线程因果证明。[B1 profile](raw/stock_b1.native_profile.v4.json)、[B1 重复](raw/stock_b1_repeat.native_profile.v4.json)、[B8 profile](raw/stock_b8.native_profile.v4.json)、[B8 重复](raw/stock_b8_repeat.native_profile.v4.json)。

**相同 key / blocks / schedule 不能证明 M、S1、S2 等内部字段完全相同。** 本次缺少完整 raw tiling 字节、匹配长度及严格 native-thread correlation，不能据此确认内部 PFA/FIA 细分实现、reduction 次序或排除 SplitKV。静态注册日志不作为实际选路证据。

## 重复与跨 batch 比较

下表比较的是目标请求的逻辑切片，而不是整张 B1/B8 张量。Q 全序列、K、V 都对齐目标 prompt 的 1024 个 token；Q 末位置对齐位置 1023。每项比较同时核对 shape、dtype、字节数、字节序和采集时逻辑 SHA-256。跨进程的对象地址不参与比较。

| B1 与 B8 的目标数据 | 第 0 层 | 第 1 层 |
|---|---|---|
| Q 全序列 | 不同 | 不同 |
| K 全序列 | 不同 | 不同 |
| V 全序列 | 相同 | 不同 |
| Q 末位置 | 相同 | 不同 |
| FIA 输出全序列 | 不同 | 不同 |
| FIA 输出末位置 | 不同 | 不同 |
| 实际传入 mask | 相同 | 相同 |

B1 与 B1_repeat、B8 与 B8_repeat 的第 0/1 层目标 Q/K/V、Q 末位置、输出全序列、输出末位置以及目标 logits 哈希均相同。跨 B1/B8 的目标 logits 哈希不同，但本次 next-token argmax 相同。重复结果只说明这两次有限观测稳定，不能推广为所有输入下确定性。

第 0 层虽然 Q 末位置与 V 相同，K 全序列已经不同，因此目标末位置注意力也不满足“相同输入”的归因前提。第 1 层所有目标输入切片都不同。当前只能确认差异最迟已存在于第 0 层 FIA 输入处；更早的具体产生位置尚未定位。这不是 FIA 同输入首错、故障 28 注入成功或历史事故复现的证据。[第 0 层 B1](raw/stock_b1/native_calls/call_000.json)、[第 0 层 B8](raw/stock_b8/native_calls/call_000.json)、[第 1 层 B1](raw/stock_b1/native_calls/call_001.json)、[第 1 层 B8](raw/stock_b8/native_calls/call_001.json)。

所附第 0/1 层记录还可重新核对每次原生调用前后的全输入逻辑哈希、目标切片哈希和对象/存储元数据未改变。`inner_precise` 未显式传入，记录中的 null 仅表示参数省略；不能从 FP16 输入或该 null 推断实际累加精度。该精度路径仍需匹配源码/profile 证明。

## 证据范围与校验

这是约 6 MB 的本地校准证据子集。`raw/` 保留选中的原始 JSON/源码字节；包括四组 result、receipt、scheduler、logits 元数据，第 0/1 层调用记录，四份 v4 profile，controller completion 和相关冻结配置/源码。[来源映射](source_mapping.json) 区分 capture 时冻结源码与之后生成四份 profile 所用的 [v4 parser](analysis_source/calibration_log_parser_v4.py)；不能把两者混同。

`log_excerpts/` 含每组 56 条实际 selected/launch 行，共 224 条。提取时只读完整原日志以核对 SHA-256，随后按 v4 parser 的行号核验文本；base64 保留原始字节和换行，文本对比仅采用 parser 自身的 ANSI 去除规则。[B1 原文片段](log_excerpts/stock_b1.json)、[B8 原文片段](log_excerpts/stock_b8.json)。四份原始完整日志及 `private_replay/*.bin` 均未下载或包含。原 JSON 中对这些文件的路径引用保留原样，不代表文件已随包交付。读者可验证记录的哈希关系，不能从此子集重新计算原张量数值误差或完整重跑日志解析。

本次 `capture_msprobe=false`，receipt 的 `native_files=[]`；此包没有 msprobe dump/stack/construct，也不冒充 statistics/mix B/F 证据。[B1 回执](raw/stock_b1/receipt.json)、[B8 回执](raw/stock_b8/receipt.json)。controller 记录自有容器已删除、实验锁已释放；这不表示整机其他服务或另一张卡空闲。

```bash
python -B verify_evidence.py
```

校验涵盖包内文件 SHA/大小、原字节来源、冻结源码、调用记录与 result ledger、profile 输入和 parser SHA、精确日志片段以及摘要重算。`manifest.json` 覆盖除自身外的全部交付文件；本文件和脚本也在清单中。`summarize.py` 可从本包 JSON 重算摘要，生成输出要求不存在同名文件，避免覆盖既有结果。校验通过表示证据子集内部一致，**不表示故障机制验收通过**。
