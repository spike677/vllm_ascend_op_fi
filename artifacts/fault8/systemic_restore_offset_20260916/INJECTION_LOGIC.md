# Fault8 故障注入逻辑说明

## 1. 本次模拟什么

名称：`SYSTEMIC_TOKEN_RESTORE_OFFSET`。

模拟公共 token 还原逻辑把“目的 token 位置”与“应读取的源 token contribution group”错配。
同一规则作用于受控请求中全部兼容调用及两个 TP rank，而不是选几个层修改输出。

本次实现是 **Python API 边界的真实 mapping 参数替换代理**。它不是 LD_PRELOAD/L4 实现，
不是原生 kernel 自发 OOB，也没有证明历史 padding 缺陷在当前版本自然发生。

实际目标是：

```python
torch_npu.npu_moe_token_unpermute(
    permuted_tokens=hidden_states,
    sorted_indices=...,
    probs=...,
)
```

只替换本次调用的 `sorted_indices` 和 `probs` 参数，不直接覆盖 `hidden_states` 或返回值。

## 2. 真实数据路径和分组含义

当前锁定实现中，routing producer 为
`DeviceOperator.npu_moe_init_routing → npu_moe_init_routing_custom`。
`TokenDispatcherWithAllGather.token_combine` 将
`abs(combine_metadata.expanded_row_idx)` 和 `combine_metadata.topk_weights`
传给上述 unpermute consumer。

实测第一次完整 prefill：

| 对象 | 本次尺寸/类型 | 含义 |
|---|---|---|
| logical tokens | T=38 | 整段请求 prompt，含模板 token |
| top-k | K=8 | 每个 token 的 expert contribution 数 |
| permuted_tokens | [304, 2048]，BF16 | 已完成 expert 计算的行 |
| sorted_indices | [304]，INT32 | 按 token-major 分组的源行索引 |
| probs | [38, 8]，BF16 | 每组 contribution 的配套权重 |

令 `I = sorted_indices.reshape(T,K)`，`P = probs`，`H = permuted_tokens`。
当前 consumer 的数学解释为：

```text
Y[t,:] ≈ Σ_j H[I[t,j],:] × P[t,j]
```

这是当前路径的语义表达，不保证任意硬件/版本具有相同浮点累加次序。
只读模型观察比较了四种索引解释；token-major gather 与实际输出吻合。
随后用独立 scratch tensors 对原生 API 的正常/重复读取索引行为做了完整输出精确校验。
不能仅凭 producer 中的 GATHER 分支名称，推断任意 consumer 都支持重复索引。

## 3. 唯一正式故障规则

```text
offset = +1
boundary = duplicate_last
source(t) = min(t + 1, T - 1)

I_fault[t,:] = I[source(t),:]
P_fault[t,:] = P[source(t),:]
```

例如四个逻辑组：

```text
原始：A B C D
故障：B C D D
```

A 不再被该还原结果引用，D 被重复引用。这里是 contribution group 的 duplicate/drop，
不是直接删除输入文本 token，也不是改 tokenizer 的 token IDs。

每个组的全部 K 个索引与对应 K 个权重一起移动。不能只做 `index += 1`，也不能只移动
索引而保留目的位置的旧权重；后一种做法会变成另一类 contribution/weight 错配故障。

索引值仍来自原来的合法源行，维度、dtype、top-k 和 collective 尺寸均不变。
修改后的 mapping 不再是 permutation，但本次原生 consumer 的重复读取语义已经过验证。

原生算子按错误 mapping 正常计算，因此输出发生变化，随后进入未修改的模型下游。
不直接干扰 residual、logits、sampler、KV cache，也不注入 NaN/Inf。

## 4. 请求、目标与覆盖范围

请求归属复用既有 `RequestScope`，通过 scheduler 的新请求记录、内部 request ID、
真实 prompt token IDs 和 scheduled tokens 关联 forward，并检查单请求 batch 和执行线程。
不是只凭累计 call index 或某段墙钟时间判断目标请求。

本实验的控制入口要求 `VLLM_SERVER_DEV_MODE=1`；它是实验使用门禁，不是产品级鉴权机制。
服务只绑定 loopback，实验限定单请求、并发1。

当前支持边界：

- 模型动态发现的 `.experts` 调用位置，并非硬编码45层或选定若干层；本次发现48层。
- 两个 TP rank 都应用同一逻辑 selection，各自使用本rank的物理 tensors。
- T>1 时必须是受控请求的第一次完整 prefill，且 scheduled token 数与 T 一致。
- 原始索引必须是完整合法排列，raw index 非负，且其绝对值与实际 consumer 索引一致。
- 当前不支持 expert-map filtering、router weight 已提前作用于输入的分支或 partial prefill。
- 当前要求 INT32 索引、BF16 权重/hidden，以及已验证的尺寸关系和范围上限。
- T=1 的 decode 调用原样执行，计为 `INSUFFICIENT_MAPPING_DOMAIN`，不做 stale-token replay。

超出已验证域时停止本次验证或记录归属失败，不静默裁剪索引、跳过若干层后仍宣称全覆盖。
实现采用研究级断言和严格 harness 校验，不能视作已经完成生产容错工程。

独立分母来自 `token_combine` 边界，不由 mutation hit 数自证覆盖率。
本次每rank：48个适用调用全部替换；38×48=1824个组实例执行规则，37×48=1776个实际改变。
最后一个组按规则保持原值，因此“规则覆盖100%”不等于“所有组内容改变100%”。
这些是层×token实例数，不是1824个不同的 prompt token。

## 5. 执行与 buffer 生命周期

每个适用调用按以下顺序执行：

1. 核对请求、metadata、分组、dtype 和索引域。
2. 同步当前 stream，将原始索引和权重读取到 host 副本。
3. 在 host 构造同 shape 的 paired mapping；Identity 模式取 offset=0。
4. 分配并上传两个新的 device tensors；校验地址与原对象不同，readback内容正确。
5. 将新 tensors 作为本次 consumer 参数，真实 native API **只调用一次**。
6. 同步 consumer stream，确保设备已完成读取，之后才允许临时 tensors 释放。
7. 校验原始索引和权重仍完整不变，并记录消费证据。

没有向原始 mapping storage 写回故障内容，所以不需要修改后再恢复原索引。
但错误结果可能已经传播到该请求的模型状态，不能把“参数副本释放”当作“模型状态恢复”。
完整验证采用新进程无注入请求确认恢复；本目录的8-token dump任务则严格只有B/F，没有新增R。

卸载前核对所有 hook owner。发生碰撞时不覆盖未知 hook；失败实验由 harness 停止自己的服务。

## 6. 如何证明不只是 hook 命中

证据分为几层，不能混用：

| 证据 | 实际检查 |
|---|---|
| 实参替换 | 原地址/新地址不同，新索引和权重内容readback正确 |
| 真实消费 | 原生调用收到替换参数、执行完成；输出与按新mapping重建的结果吻合 |
| 原输入保留 | 完整原索引/权重未变；hidden前16列的hash未变 |
| 系统性覆盖 | 独立combine分母、API调用数、替换调用数一致，双rank逻辑hash一致 |
| 模型效果 | 比较真实整数token IDs及完整正文 |
| 恢复 | 完整实验的新进程R与B精确一致 |

模型运行中的输出重建只取前16个 feature columns，relative L2门限0.01，
同时检查 `allclose(rtol=0.02, atol=0.02)`。这不是全量输出逐字节证明。
hidden保留检查也只是前16列；不能把日志字段 `original_inputs_preserved` 解读为全部hidden字节审计。

## 7. msprobe 如何共存

调用组合是 **Fault wrapper → msprobe wrapper → native API**。
Fault wrapper 先替换 mapping 参数，msprobe 因而观测到传入真实 consumer 的故障参数及其真实输出。
这与“原生算子返回后才改输出”的 Fault18 代理观测位置不同。

兼容处理包括：

- 在 Fault owner 安装前完成 msprobe 的一次性模块hook注册，不推进dump计数。
- 将该 API 的 wrapped registry 项组合为 Fault wrapper，保留 original/native registry 项。
- msprobe 采样阶段 restore/register 时沿用组合后的引用，不每步强行重挂hook。
- 每次绑定调用核对 registry/owner；卸载时还原缓存条目与属性。

本次采集：statistics/mix、summary_mode=statistics、step0–7、rank0/1。
F在step0、probe active时发生48次替换/rank；后续7个decode step记录异常的传播。
每个step/rank有48条目标API日志，B/F各768条。

## 8. 已完成的结果与边界

完整B/I/F/新进程R实验：B/I/R均为316 token且精确一致；F达到1024 token上限，
全部为ID198（换行），没有非空白正文。首个token从B的785分叉为198。
另一次单独F复测得到相同1024个换行。未进一步增强到模型级offset=+2。

本目录8-token msprobe实验：

- B：`The marketplace roared to life at dawn`。
- F：`\n\n\n\n\n\n\n\n`。
- 均与对应无msprobe响应的前8个token精确一致。

可以说明当前配置下真实mapping参数故障造成了可观察的模型异常。
不能据此宣称历史padding物理根因已精确复现、所有模型可复用、完整logits已定位，
或进行了三轮完整B/F/R重复性矩阵。PCP/SP/P-D适配和Fault17不在本次结论范围内。

## 9. 实现和证据索引

研究实现中的对应文件/职责：

- `strategy.py::source_groups`：logical source selection。
- `systemic_extension.py::SystemicBinding`：独立计数、实参替换和消费校验。
- `fault8_extension.py::Binding`、`request_scope.py`：请求及调用关系、owner生命周期。
- `probe_extension.py::ProbeBinding`、`composition.py`：msprobe引用组合及恢复。

成功运行的 `systemic_extension.py` SHA256：
`0d3e18b0c3058699be99990d887fdc2afe34b144b255da2c74729ad750fa03e9`。

上述源码目前属于独立研究工作区，**本次文档提交不包含运行实现**。
本目录的ZIP是dump-only证据，不能单独代替native命中、完整响应和恢复证据。

- 完整实验：`/home/haoran/fault8_systemic_mapping_20260915_v3`。
- 单独F复测：`/home/haoran/fault8_f_once_20260916_v1`。
- B/F采集：`/home/haoran/fault8_msprobe_bf8_20260916_v1`。
- ZIP配置、内容清单及SHA256：[README](README.md)。

这些路径用于证据溯源，不表示用户应直接重跑固定目录下的单次harness覆盖原记录。
