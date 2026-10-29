# 故障 28：stock HF 原生 FA 校准 a3

**四组 stock HF 模型运行均完成，实际日志各包含 28 次 FlashAttentionScore launch；未进行故障注入或 msprobe 采集。** 本包的 tiling 状态保持 `CAPTURED_UNDECODED_FRAMING_CANDIDATE`，没有提前标记为完整 profile 或故障验收通过。[摘要](summary.json)、[controller 完成记录](raw/completion.json)。

2026-10-29 使用 124 物理卡 5、Qwen3-0.6B、Torch 2.10.0 / torch-npu 2.10.0.post4 / Transformers 5.5.4。原 checkpoint 为 BF16，通过原生 HF 加载为 FP16，`attn_implementation=sdpa`，无 cache、dropout=0、未编译模型，使用官方 `SDPBackend.FLASH_ATTENTION` 上下文。由于 NPU overrideable 路径本身不能仅凭该上下文保证没有 fallback，本包另保留实际 FA launch 日志。它是独立的 HF/SDPA/FA 路径，不能与 vLLM/FIA 的 a5 校准混为一次运行。[B1 原始结果](raw/stock_b1/result.json)、[B8 原始结果](raw/stock_b8/result.json)。

每次单个完整 prefill 使用相同的 2048-token 目标序列；B8 的其余七项是同一 token 序列的重复副本。目标是 batch slot 0，最后位置 2047。每个条件重复一次，并使用新模型进程。配置和实际调用的 Q/K/V、mask、scale、GQA 参数记录均保留。[冻结 B1 配置](raw/source/setup/cases_2048/stock_b1.json)、[冻结 B8 配置](raw/source/setup/cases_2048/stock_b8.json)。

| 条件 | SDPA / 实际 FA launch | key | blocks | 声明长度 / 对齐日志长度 | API workspace |
|---|---:|---|---:|---|---:|
| stock_b1 | 28 / 28 | 22619644976 | 20 | 1464 / 1472 字节 | 73,401,856 字节 |
| stock_b1_repeat | 28 / 28 | 22619644976 | 20 | 1464 / 1472 字节 | 73,401,856 字节 |
| stock_b8 | 28 / 28 | 22619644976 | 20 | 1464 / 1472 字节 | 102,761,984 字节 |
| stock_b8_repeat | 28 / 28 | 22619644976 | 20 | 1464 / 1472 字节 | 102,761,984 字节 |

profile 里的 kernel workspace **数量**为 1；上表 **API workspace 字节数包含 helper 算子**，不能等同于 kernel workspace-size 向量。该提取结果没有 schedule mode，因此本包将它记为未知，不从 key 推算。[B1 profile](raw/stock_b1.fa_log_profile/profile.json)、[B1 重复](raw/stock_b1_repeat.fa_log_profile/profile.json)、[B8 profile](raw/stock_b8.fa_log_profile/profile.json)、[B8 重复](raw/stock_b8_repeat.fa_log_profile/profile.json)。

## 目标输入与输出

HF 记录的 `shape` 是源张量的完整 batch 形状，`hashed_slice` 明确为 `batch[0:1], all heads/positions`。摘要先将逻辑切片形状重建为 `[1, ...]`，再核对 FP16 字节数与 SHA，避免将 B1/B8 源形状不同错误判成目标内容不同，也不直接比较不同大小的整张量。

| B1 与 B8 的目标切片 | 第 0 层 | 第 1 层 |
|---|---|---|
| Q 全序列 | 不同 | 不同 |
| K 全序列 | 不同 | 不同 |
| V 全序列 | 不同 | 不同 |
| FA 输出全序列 | 不同 | 不同 |
| scalar 参数 / causal 条件 | 相同 | 相同 |

B1 与 B1_repeat、B8 与 B8_repeat 的上述目标输入/输出以及目标 logits 哈希均相同。B1 与 B8 的目标 logits 哈希不同，next-token argmax 相同。本版本没有独立记录 Q 末位置或 FA 末位置的单独哈希，本包不从全序列哈希推断具体哪个 token 发生变化。[第 0 层 B1](raw/stock_b1/native_calls/call_000.json)、[第 0 层 B8](raw/stock_b8/native_calls/call_000.json)、[第 1 层 B1](raw/stock_b1/native_calls/call_001.json)、[第 1 层 B8](raw/stock_b8/native_calls/call_001.json)。

第 0 层调用前目标 Q/K/V 已全部不同。因此本次跨 batch 输出变化不满足“相同注意力输入”的归因前提，不能指认为原生 FA 同输入首错，也尚未定位更早的差异产生位置。有限两次重复只支持本次观测的重复稳定性。

**Q/K 的原生格式 0→2 规范化与数值差异分开记录。** 所附第 0/1 层中，调用前后 Q/K 的 `actual_format` 从 0 变为 2；目标切片逻辑 SHA 及其他已记录对象/几何/地址元数据保持一致，V 格式未发生该变化。摘要逐字段复核后才接受这项单向规范化，并保留原始 `input_format_normalizations`。它是本次 stock 路径观察到的格式行为，不是故障注入或目标值变更。此旧校准版本核查的是目标切片完整性，不应宣称已完成后续正式方案要求的全 B8 Q/K/V before/after 哈希门槛。

## tiling 元数据及范围

`raw/<case>.fa_log_profile/` 保留每次调用的两个文件：1464 字节 `declared_prefix.bin` 与全部 1472 字节 `aligned_log.bin`。共 112 次调用、224 个小型二进制文件，**均是日志中的 uint32 tiling 调度元数据，不是张量数据**。`verify_evidence.py` 依据 profile 的完整原始 word 序列重新打包，逐字节核对 aligned 文件及其声明长度前缀；没有猜测字段偏移或抛弃多出的字节。

每个 batch 条件内 28 次调用的声明长度前缀相同，重复条件的对应前缀也全部相同；B1/B8 前缀不同。完整 aligned 文件在同一组内有 28 个不同哈希，两次重复之间仅 1/28 对应 aligned 哈希相同。这些额外变化局限于前缀以外的最后 8 字节，但**本包不擅自将它们命名为 padding、校验码或某个结构字段**。前缀边界仍须与匹配版本的公共 `TilingContext` raw 字节比较，并结合 typed POD 解码验证；本包没有纳入这些后续证据。不能仅凭 key 或 raw 字节有差异推断 M/S1/S2、reduction 合法性、精度模式或 non-SplitKV。

`log_excerpts/` 保留每组的 launch、两段 PrintTilingData 和 API workspace 精确行，共 448 条。提取时只读核对整日志 SHA，base64 保存原行字节和换行；未下载完整约 400 MB 日志。[B1 原始行](log_excerpts/stock_b1.json)、[B8 原始行](log_excerpts/stock_b8.json)、[提取器](analysis_source/extract_fa_log_profile.py)。所有 `private_replay` 张量二进制均未包含，原 JSON 对它们的引用保留。因此此包可以校验记录和 tiling 字节来源，不能重算模型数值误差。

`capture_msprobe=false`；这些运行没有统计三件套，也没有 msprobe receipt。没有以 runtime/native log 代替 msprobe B/F 首差证据。controller 记录自有容器删除、锁释放；没有声称整机或其他服务空闲。

```bash
python -B verify_evidence.py
```

[清单](manifest.json) 覆盖除自身外的全部文件，包括本说明、脚本、原始字节和派生摘要。校验会重算摘要、核对 source manifest 与调用 ledger、raw word 重建、日志原行和全部 SHA。[来源映射](source_mapping.json) 标明原始选取路径。校验 PASS 仅表示本包内部一致，不是完整 profile 或故障机制验收 PASS。
