# Fault26 FIA：真实预训练推理与公共 Worker

在124的单张空闲卡5、910B4/A2、CANN8.5.1、vLLM-Ascend0.18.0 上完成了 SmolLM2-135M 和公开 GLM-4-9B-0414 的限定验收。真实模型分别是 `LlamaForCausalLM`、`Glm4ForCausalLM`，不是构造模型。TP1、BF16、eager，每阶段一个新请求的短文本 prefill；没有训练调用。

**实际 writer→consumer→恢复通过；统计验收须按下面各阶段分别读取。** GLM 的原始 NaN 和两种模型的 OFF 缓存统计差异均保留，不能把本次结论写成所有统计相等或整个缓存池恢复。

## 接线与实际作用

标准 vLLM 创建临时 Worker，安装后的三个 wheel 提供共用 Extension 和 Provider。Host 通过实际 enqueue ID 推进 `reference → lossless → comparison → off`，Worker 读取当前 scheduler、输入存储及输入字节并交叉核对。Extension 不创建 Worker，不替换 `execute_model` / `_prepare_inputs`，用户不填写层名。

自动发现实际 Attention 模块；当前 forward 的 AddRmsNormBias 产生仍存活的 residual。故障将真实 FIA kernel 的输出目的地改到该 residual，复制正确 FIA 输出到原目的地，让后续真实 AddRmsNormBias 消费被覆盖的 residual，再恢复 residual。没有在消费者之后修改 logits。

| 模型 | 实际 Attention 候选 / 适用调用 | F 实际写入、消费、恢复 | 首 token：reference → F → OFF |
|---|---:|---:|---|
| SmolLM2-135M | 30 / 29 | 29 / 29 / 29 | 2（空文本）→14423（` affili`）→2 |
| GLM-4-9B-0414 | 40 / 39 | 39 / 39 / 39 | 576（` The`）→104276（`”（`）→576 |

首层没有本轮产生的 residual，因此正常透传。全部观察到的适用调用都已路由，但这只是一请求的已观测范围，**公共 `all_eligible` 资格仍为 false**。无损和 OFF 的完整 hidden states、完整 logits 及生成结果精确等于 reference；F 的完整 hidden/logits 有限且变化。每次原生写入均核对实际消费者、正确 FIA 输出复制和 residual 写后恢复哈希。

实现入口：[Serving 会话](../../src/demoncat/runtime/vllm_ascend/worker_extension/serving26.py)、[Provider](../../src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault26.py)、[部署适配器](../../fault_injection/deployments/residual26.py)、[原生桥接及状态 ABI](../../fault_injection/native_adapters/legacy26_fia_bridge.inc)。新状态 ABI 读取真实注册、prepared、executor、pending consumer 和当前线程，不用 Python flag 推断原生空闲。

## 独立 statistics/mix 复核

B/F 均采完整模型 API/module，原始配置为 `statistics/mix`、`summary_mode=statistics`，无算子过滤。采集器一致。`step0/1/2/3` 分别为 reference/lossless/comparison/OFF，不是 decode token 步数。独立比较器未改：仅读取三件套、配置和采集器身份，固定 `abs(F-B) > 1e-8 + 1e-5*abs(B)`，不读原生命中或私有张量。

| 模型 / step | 原始严格结果 | 保留的解释边界 |
|---|---|---|
| SmolLM / 0、1 | 无可观测统计差异；B 自比相同结论 | 各有5722项不可观测，不等于全部张量相同 |
| SmolLM / 2 | 首个可观测差异：`_C_ascend.npu_add_rms_norm_bias.2.forward/input_args/1`，Max **29.5→2.203125**；之前392项不可观测 | 与 residual 消费输入变化一致；统计不能唯一证明写错地址 |
| SmolLM / 3 | `NPU._npu_reshape_and_cache.2.forward/input_kwargs/key_cache`，Min **−16→−16.375**；之前523项不可观测 | 新请求完整结果恢复，整个缓存池没有统计恢复；未证明差异具体位于哪些无效块 |
| GLM / 0 | 无可观测统计差异；B 自比相同结论 | 8862项不可观测 |
| GLM / 1 | `Tensor.__setitem__.74.forward/input_args/0`，Max **NaN→NaN**，`NONFINITE_UNCOMPARABLE`；B 自比也如此 | **严格统计不通过**，不能把 NaN 相等化或删除 |
| GLM / 2 | 首个可观测差异：同一 AddRmsNormBias `input_args/1`，Max **1.265625→6.4375**；之前442项不可观测 | 后续 B 自比仍遇到同一 NaN，不能宣称全段统计自检通过 |
| GLM / 3 | `Torch.index_select.0.forward/input_args/0`，Max **20726→104276** | 同一新请求的真实输入和输出已另行核对，但持久 token 缓冲区统计有差异；未做整块缓冲区字节恢复 |

统计中的 request mapping 保留 `UNPROVEN_REQUEST_MAPPING`；真实 request ID 对应关系来自独立运行时证据，不能反向充当统计比较输入。这里的首差是**首个可观测数值差异**，不是核内写错地址发生位置。

完整摘要见 [SmolLM B](summaries/fia26-smollm-baseline.json)、[SmolLM F](summaries/fia26-smollm-fault.json)、[GLM B](summaries/fia26-glm-baseline.json)、[GLM F](summaries/fia26-glm-fault.json)。逐阶段统计报告为 `summaries/fia26-{smollm,glm}-statistics-step{0..3}.json`，保留 B 自比结果、原始三件套哈希和私有完整报告哈希。

## 失败、控制和范围

早期观察器在 msprobe 包装 producer 之后失去 wrapper 所有权，清理失败，未计为通过；现在先启动采集器，再安装并核对自己的观察器。早期 SmolLM 的 B/F 自动分配了29215/29217块 KV，严格比较因 shape 不齐失败，见[原失败](summaries/fia26-prior-alignment-failures.json)。正式两侧均明确使用64块 KV，未修改统计原件或比较容差。

正式控制批次前四个 FIA case 各 exit0；同批第五个独立只读 allocation case 因未找到真实赋值方法失败，因此控制器整批仍记失败。修正只读观察器后用新 attempt 重测，原失败没有覆盖。

新原生构建 a2 与已测试 a1 的 FIA DSO 字节相同，SHA256 `584c85381dc927fdedef92b4af7f433aa49c6dddbfe5d23500499537b2569802`。[构建回执](summaries/fia26-build-receipt.json)、[包构建](summaries/fia26-package-receipt.json)、[581项CPU测试](summaries/cpu_fia26.json)。CPU成功不等于NPU验收。

原件在124 `/home/p3p4-closure-20261117/fia26-formal-and-allocation-a1`。只公开实现、摘要和哈希，不公开新统计原件、完整张量、权重或 vendor 二进制。未验收本机制的多请求合批、连续生成、TP2、其他版本/硬件、任意并发 owner；26 residual 的旧 rank1 NaN 缺口也没有由本结果自动关闭。
