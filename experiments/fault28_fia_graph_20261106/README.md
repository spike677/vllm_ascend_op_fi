# 故障 28：真实 FIA 图模式的 Batch/M/causal 归约验证

**124 物理卡 5 已通过受控 FIA 图子图机制验收（2026-11-06）。** 最终 CPU FP64 统计轮完成 23 组、69 次真实 FIA 图执行；原 FP32 轮的 23 组、69 次计算及其首差验收失败结果保留。两轮 69 对调用的完整目标输出、全部请求输出哈希及私有 host raw/key 完全一致，统计增强没有改变算子计算。

[正式验收及原始统计](../../artifacts/attention_tiling/fault28_fia_graph_statistics_20261106/README.md) · [完整验收报告](../../artifacts/attention_tiling/fault28_fia_graph_statistics_20261106/reports/acceptance.json) · [独立分析方法](analysis/README.md)

这里验证的是 CANN 9.1.0 中新增的合法 causal 尾块 kernel 变体，在真实 `FusedInferAttentionScore` 图执行中复现同类数值不一致。输入来自真实 Qwen3-0.6B 第 2 层（index1）的已捕获 Q/K/V，消费者使用该层真实 `o_proj` 权重。它是显式构造的 FIA→o_proj 子图，**不是历史盘古版本、未改模型的自然 FIA 路径或完整 28 层模型图验收**。本轮最后一个 query 没有变化，没有证明历史问题中的第一个生成 token logits 不一致。

## 实测结论

| 对照 | 完整目标 Attention 的差异元素 / 4,194,304 | 最大绝对差 | 同 B8 的真实 o_proj 差异元素 / 2,097,152 |
| --- | ---: | ---: | ---: |
| stock → 私有 OFF；两次独立进程，B1/B8 | 0 | 0 | 0 |
| 原生 M512 → 新 causal 变体 M512 | 0 | 0 | 0 |
| M512 → M128-only / M256-only，无尾块裁剪 | 0 | 0 | 0 |
| 相同 M128，关闭 → 启用 causal 尾块裁剪 | 140 | 0.000244140625 | 6,141 |
| 相同 M256，关闭 → 启用 causal 尾块裁剪 | 81 | 0.0001220703125 | 3,560 |
| 自动 Batch 策略，B8 OFF → F，两个独立进程 | 140 | 0.000244140625 | 6,141 |
| F/F 重复；重新编译 OFF 恢复 | 0 | 0 | 0 |

固定 M 和 kernel 策略时，同一目标 Q/K/V 的 B1/B8 **完整 Attention 输出逐 bit 相同**；自动策略从实际 B1 选择 M512、从实际 B8 选择 M128 后，目标 Attention 相差 140 个元素。M-only 不变，而相同 M 的 causal 变体有差异，补齐了本输入下 `Batch→M→causal 核内宽度→FIA 数值变化` 的控制证据。

消费者本身也存在跨 Batch 数值路径差异：固定 OFF 时，B1/B8 的 FIA 输出相同，`o_proj` 已有 1,750 个元素不同。因此传播归因采用**同 B8 的 B/F 对照**，不能把自动 B1/B8 的 7,789 个 projection 差异全部归因到 FIA。全部目标元素逐项比较；固定十个 query 位置的 CPU FP64 参考最大绝对误差为 0.00023865677269452323。这是预注册的数值 sanity 检查，不替代全算子精度标准或业务质量验收。

## 为什么符合这类故障原理

当前 stock CANN 9.1 和另一套 8.5.1 的发现实验中，B1/B8 的 prefill M 都为 512；默认 20/40 核预算下，9.1 还启用了跨核 FlashDecode。它们不能直接作为历史故障 28 的 M 变化证据。本轮双方统一采用实际接受的 `16|32` Cube/Vector 预算，并通过实际 key、`numOfFdHead=0` 及全部有效分核边界 `s2End=0` 核验 Reduce 轴未切多核。

注入改变的是实际 host 的 M 选择，然后沿用原生分核、workspace、softmax flags 和序列化流程。共享的 causal kernel 规则只去掉该 M 块内**所有 query 都会 mask 的未来列**：

```text
q_end = (gS1_start % actual_S1) + actual_M_rows   # exclusive
kv_begin = S2_index * S2_base
actual_inner_N = min(native_inner_N, align32(q_end - kv_begin))
```

例如同一个 query63，M512 的查询块末端为 512，首个 Reduce 宽度为 512；M128 的查询块末端为 128，宽度变为 128。两者都保留有效 key0..63，原来的 K/V 步进仍为 512，未丢失或重复任何合法 key。M128 产生 128/256/384/512 的动态宽度，M256 产生 256/512，M512 始终为 512。

该宽度进入真实 BMM1 N、Softmax/Reduce 的 columnCount 及 BMM2 K，改变有限精度计算路径，算子自行产生不同的 FP16 输出。**没有修改 QKV、权重、mask、scale、stride 或返回数值，没有在消费者后补写差异。** 采用独立合法 kernel key 和配套实现，未猜测或直接修改 opaque executor/raw 字段。

这里证明了执行策略链和数值现象；N 同时影响 BMM1、Softmax 与 BMM2，没有唯一分离某条 Reduce 指令的贡献。动态 N 宽度来自已核验实际 key/raw 和 kernel 代码的推导，不伪称设备内 trace。历史具体版本的缺陷尚未复现。

## 调用链与关键代码

```text
固定真实 QKV；B8 为目标序列的八份逐字节复制
→ torch.compile(fullgraph=True, dynamic=False)，真实 TorchAir/GE 图
→ GE FusedInferAttentionScore
→ 隔离编译子进程：FiaTilingNonQuant::CalcMBaseSize
→ Select(实际 Batch/shape/版本条件)，选择 M
→ 原生 Split / workspace / flags / public serializer
→ CANN 编译器实际收到 key + 1,576 字节 tiling raw
→ 实际 FIA JIT kernel：CalcParams → M 块 causal 尾宽
→ BMM1 / SoftmaxFlashV2 / BMM2 → FIA output0
→ 原层真实 o_proj 消费该输出
```

- [graph_probe.py](graph_probe.py#L80)：固定输入、原生 FIA OpOverload、真实 o_proj、一次 fullgraph 编译、三次实际图执行、完整输出私有核验。
- [apply_policy.py](source_policy/apply_policy.py#L52)：在原生 M 选择后接入 selector；先重算依赖 tiling，再选择新 key。新增 key 为 `109000000000000000` / `109000000000400000`，保留原生 key。
- [fault28_fia_policy.h](source_policy/fault28_fia_policy.h#L90)：绑定主进程及可核验的编译子进程祖先链、starttime、token、真实几何、dtype、causal、无 paging/quant/rope 等 23 个条件；记录真实公开 context。
- [fault28_causal_tile.h](source_policy/fault28_causal_tile.h#L16)：实际 host/device 共用的尾宽函数；CPU 检查覆盖 12,288 个 query/group/M 组合，NPU 再验证合法执行与数值结果。
- [native_plan.py](analysis/native_plan.py)、[public_plan.py](analysis/public_plan.py)、[audit_case.py](analysis/audit_case.py)：实际任务、独立 public ABI 解码、编译器 raw、二进制 SHA 和输入/版本证据。
- [graph_evidence.py](analysis/graph_evidence.py)：独立统计定位，不读注入命中或期望 profile。[observer_neutrality.py](analysis/observer_neutrality.py) 检查统计方式不改变计算。

每个符合条件的图节点在 host 编译选择边界处理，没有一次命中后关闭的计数限制。图内执行策略随后固化，三次回放持续使用实际编译结果；不能把修改环境变量当作既有图立即恢复。本轮关闭恢复使用新进程、重新编译。当前子图每图一个 FIA，不据此声明所有模型层、形状或请求调度路径均已覆盖。

## msprobe 的首差与精度限制

最终配置为 `task=statistics, level=mix, summary_mode=statistics, precision=high, async_dump=false`，保存 step0/1、rank0 的原始 dump/stack/construct、config、采集器身份及完成回执。step 是相同完整 prefill 子图的两次回放，不是 decode token。

图内保留官方 `set_ge_dump_config` 的原生 GE CSV。它的六位统计在本例首先分辨出下游 BatchMatMul 输出，**没有直接分辨 FIA**。原 FP32 统计轮在 M256 的 FIA 也没有分辨，严格首差验收因此失败；[原失败与原始数据](../../artifacts/attention_tiling/fault28_fia_graph_statistics_20261106/limited_fp32/) 原样保留。

最终额外使用明确披露的任务私有 CPU FP64 四统计采集器，对所有真实浮点张量统一统计 detached 临时副本，不按算子、shape 或注入模式筛选。原生 `GraphBoundary` module 的 output slot0 通过实际 GE `NetOutput → FusedInferAttentionScore:0` 直接别名映射，独立推断首个可观测生产者为 **FIA output0 的 Mean/Norm**。这没有把 module dump 改名为 eager FIA API dump，也不意味着任意未适配模型图的默认 msprobe 都具备相同可观测性。

B8 主对照的 Mean 为 `0.0002203156135607287 → 0.0002203156356586078`，Norm 为 `339.158837192348 → 339.15883705920237`；两步、两个独立 F 均定位一致。输入统计对齐与完整 QKV 字节核验单列，统计相同不等于张量逐元素相同。bool/整数 null、空 LSE 及缺失项保留不可观测，不判相等。

## 构建、复测与交付范围

固定环境为 124/910B4/物理卡5/TP1，CANN9.1.0、torch2.10.0、torch-npu2.10.0.post4；官方 ops-transformer commit 为 `c77c0829f1f0f0b8f882db1d9b051a758b605472`。完整包、源码和依赖指纹见证据。构建及安装只写本任务私有目录，系统 SDK 未修改。原生公开 serializer 的 65 个 SDK 范围覆盖 1,576 字节，63 个逻辑字段与真实 host raw 逐项一致，编译器实际消费的 raw/key 也一致；没有直接读取设备入口参数。

[构建前合同](source_policy/BUILD.md)、`source_policy/build_cpu.py`、`stage_cpu.py` 和 `analysis/repair_stage_cpu.py` 分开记录 CPU 构建、任务内安装与 JIT include 修复。CPU 成功不算 NPU 验收。旧链接、缺失 include 和主 PID 未覆盖编译子进程的失败均未改写成通过。

原数值矩阵为 [formal_matrix_v3.json](setup/formal_matrix_v3.json)，FP64 复采使用 [fp64_matrix_v1.json](setup/fp64_matrix_v1.json)；[逐项等价证明](setup/fp64_numeric_equivalence.json) 确认只改变三个统计采集器配置字段，数值输入和注入矩阵完全一致。新 attempt 必须经过已有 controller 的资源锁及逐次设备门禁，每窗口至多八组，仅用卡5。复测前需审查本机精确容器 allowlist；本次 prelaunch 仅按用户授权停止经核验的 card5-only ATK，没有修改队列或停止 gyl_test。

交付包含实现、原生统计、图和 source mapping、实际 raw/编译/launch 证据、完整比较报告、CPU 测试和 SHA 校验。完整 QKV、整批输出、权重、完整 DEBUG 日志、`.pt/.bin`、库和编译包私有保留。20 项分析/导出测试及实际采集器 CPU 方法检查通过；最终验收以实机 69 次图执行和独立分析为准。
