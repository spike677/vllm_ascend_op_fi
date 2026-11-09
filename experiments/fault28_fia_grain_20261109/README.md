# 故障28：真实 FIA 图的设备端归约粒度补验

**后续完整模型补验：** [完整Qwen3 FIA图](../fault28_fia_fullmodel_20261109/README.md)已在同版本接入28层，完成10组/30次完整模型执行与840次FIA节点执行，证明全部匹配层/请求覆盖、独立FIA首差及8/8请求首token logits传播。本文件的原始子图结果与适用范围保持不变；完整模型结果以新证据单独验收。

**124 物理卡5的受控 FIA 图子图机制验收通过（2026-11-09）。** 固定真实 Qwen3 第2层（index1）的完整目标 Q/K/V，完成23组、69次真实 TorchAir/GE FIA 图调用。实际 Batch1→8 对应 M512→128；在同一 query63 上，设备端实际送入 Softmax 的列数从512变为128。两个独立 F 的 `statistics/mix` 首个可观测生产者均为 FIA output0。

[原始统计、设备标量和验收报告](../../artifacts/attention_tiling/fault28_fia_measured_grain_statistics_20261109/README.md) · [联合验收](../../artifacts/attention_tiling/fault28_fia_measured_grain_statistics_20261109/data/reports/measured_chain/acceptance.json) · [独立分析说明](analysis/README.md)

本轮在[已发布的 FIA 图实验](../fault28_fia_graph_20261106/README.md)上增加只读设备标量，补齐此前动态 N 仅由源码推导的缺口。范围是**真实 FIA→原层 o_proj 的编译子图**；采用当前版本的受控 Batch/M 选择器和合法 causal 尾块变体，尚未复现历史盘古默认策略或完整模型 FIA 图。最后 query 没有变化，不能宣称复现了首个生成 token 的 logits 不一致。

## 固定输入与 B/F

B8 是同一目标序列 Q/K/V 的八份逐字节复制；Q/K/V 来自真实 Qwen3-0.6B layer1，消费者加载该层真实 `o_proj` 权重。每组独立进程、一次 `torch.compile(fullgraph=True, dynamic=False)`，执行 warmup 和两次有统计的 prefill 回放；step0/1不是 decode token。

| 对照 | 实际策略 | 用途 |
| --- | --- | --- |
| stock / 私有 OFF，B1/B8，各两进程 | M512、原生 kernel | 验证私有部署与关闭状态兼容 |
| 固定 M512/128/256，B1/B8 | 分别保持无裁剪或相同 causal 规则 | 分离 Batch、M 与核内宽度 |
| 自动选择，B1/B8，各两进程 | 实际 B1→M512，B8→M128，相同 causal 规则 | 验证受控 Batch→M→宽度 |
| **主 B/F，同 B8** | **B：causal M512；F：causal M128** | **同规则只改变 M，排除合批消费者自身差异** |
| F/F 与重新编译 OFF | 独立重复、关闭恢复 | 检查稳定性与可恢复性 |

主 B 是 `g_causal_base512_b8_r1`，F 是 `g_batch_b8_r1`，独立 F 是 `g_batch_b8_r2`。两边采用同一 causal kernel 规则，不能将“同时启用规则并改变 M”的对照当成唯一归因。完整配置见 [fp64_matrix_v1.json](setup/fp64_matrix_v1.json) 和 [device_trace_equivalence.json](setup/device_trace_equivalence.json)。正式矩阵 SHA-256 为 `e653c6a48d63128d4ce4d84e34c8d4cfb31db1893a9bbbc728ca101248c2fbd9`。

## 原理、实际边界和调用链

```text
固定真实 Q/K/V；B8 为目标序列八份复制
→ TorchAir fullgraph → GE FusedInferAttentionScore
→ FiaTilingNonQuant::CalcMBaseSize 后按实际 Batch 选择 M
→ 原生分核 / workspace / flags / public serializer 重新生成依赖 tiling
→ 实际编译器接收 key 与 1,576 字节 raw → 实际 FIA 静态图任务
→ M 块末端决定 causal 尾块宽度
→ 设备端 SoftmaxFlashV2Compute 的 columnCount/actualColumnCount
→ 原生 BMM1 / SoftmaxFlashV2 / BMM2 → FIA output0
→ 真实 layer1 o_proj 消费输出
```

共同的 causal 规则只裁去该 M 块内所有 query 都会 mask 的未来列：

```text
q_end = (gS1_start % actual_S1) + actual_M_rows
kv_begin = S2_index * S2_base
actual_inner_N = min(native_inner_N, align32(q_end - kv_begin))
```

对于 query63，M512 块末端为512，M128块末端为128；首轮 Softmax 实测列数分别为512和128，两者都保留 key0..63。原生 K/V 步进仍为512。公开 ABI 全字段核验、无 SplitKV 和合法 key 覆盖检查共同确认：没有以遗漏、重复合法 key 来制造差异。

本次只在真实 `SoftmaxFlashV2Compute` 内、未改动的 `SoftMaxShapeInfo` 构造和实际调用之前记录标量。host 算法和二进制不变；Q/K/V、权重、mask、scale、stride、返回值均没有被改写。新记录只采样 request0/query-head0 的七个预注册位置 `[63,255,511,767,1023,1535,2047]`。

实际核验19组私有实现，官方 SDK 解码共798条标量，且与各自进程的原始 runtime printf 行逐项一致。每组42条，唯一记录形态的重数均为3，与三次实际图执行一致。**图模式 printf 异步刷出，本轮只建立进程/图级汇总对应，未证明每条标量与单次 replay 的时间配对。** 保留最初逐次时间窗核验失败报告，没有强行归配。

M128 的七个采样位置实际观察到128/256/512列，M256观察到256/512列。源码还有384列的合法路径，但七个采样位置没有覆盖它，不能称为本轮设备实测。

## 实测数值与验收

| 对照 | 完整目标 Attention 差异 / 4,194,304元素 | 同 B8 o_proj 差异 / 2,097,152元素 |
| --- | ---: | ---: |
| stock→私有 OFF；原生 M512→causal M512 | 0 | 0 |
| M512→M128-only / M256-only，不裁剪尾块 | 0 | 0 |
| 相同 M128，关闭→启用 causal 裁剪 | 140 | 6,141 |
| 相同 M256，关闭→启用 causal 裁剪 | 81 | 3,560 |
| **同 causal 规则，主 B M512→F M128** | **140** | **6,141** |
| 独立 F/F；重新编译 OFF 恢复 | 0 | 0 |

主 B/F 的 Attention 最大绝对差为 `0.000244140625`，最后 query 的差异元素为0。完整目标逐元素比较覆盖16 heads×2048 queries×128 dimensions；其余请求通过同步后的逐请求及整批输出 SHA 核验。两个 F 的三次执行均有8/8请求 Attention 哈希变化；F/F与 OFF 恢复的 Attention、projection 哈希完全一致。这不是所有请求均做了设备标量采样或逐元素差异枚举。

固定策略后，B1/B8 的完整目标 FIA 输出逐 bit 相同；固定 OFF 时跨 Batch 的 `o_proj` 仍有1,750个元素差异。因此主传播证据使用同 B8 对照，不能把自动 B1/B8 的7,789个 projection 差异全部归给 FIA。

只读观测中性检查还将本轮23组/69次调用与上轮无设备 printf 的对应调用逐一比较：完整目标字节、整批/逐请求输出哈希、实际 host raw/key 全部相同。27项 CPU 分析和反例测试全部通过、无跳过；CPU结果不替代本轮实机验收。

本轮通过的是 `Batch→M→causal 核内 Softmax 列粒度→FIA 数值变化` 的**受控机制链**。同一宽度也进入 BMM1 N 与 BMM2 K；尚未唯一分离某条 Reduce 指令的误差贡献，也没有记录归约指令树。合法策略切换后的数值差异本身，不等于不符合算子精度标准或业务质量下降。

## msprobe 独立首差

两边均为 `task=statistics, level=mix, summary_mode=statistics, precision=high, async_dump=false`，保存两步 rank0 原始 `dump.json/stack.json/construct.json`、配置、采集器身份和完成回执，主交付不含完整张量。

统计使用与上轮相同、明确披露的任务私有 CPU FP64 四统计采集器，对所有真实浮点张量统一处理 detached 副本，不按算子或注入状态筛选。原生 `GraphBoundary` 的 output0 通过实际 GE `NetOutput→FusedInferAttentionScore:0` 直接别名映射；这是 module 统计与图生产者的关联，没有把记录改名为 eager FIA API。

独立分析按实际图依赖排序；不读注入命中、期望 profile、设备 trace 或数值验收结果。主 F 与独立 F 的两步首差均为 **FIA output0 Mean/Norm**：

| 字段 | B | F |
| --- | ---: | ---: |
| Mean | 0.0002203156135607287 | 0.0002203156356586078 |
| Norm | 339.158837192348 | 339.15883705920237 |

发布后本地再次构造物理盲验目录，只放配置、采集器身份、原生统计及 FX/GE 图/CSV；不放 case/result、host、策略、设备或完整数值报告。两份 F 均重新得到相同首差，[盲验报告](../../artifacts/attention_tiling/fault28_fia_measured_grain_statistics_20261109/local_blind_verification.json)及独立输出随包交付。

原生 GE CSV 的六位显示精度在此例先分辨下游 MatMul；默认统计不自动具有本轮增强后的可观测性。[此前 FP32 轮的 FIA 首差失败](../../artifacts/attention_tiling/fault28_fia_graph_statistics_20261106/limited_fp32/)仍保留。null、空输出、缺失项均保留不可观测；统计相同也不能当作张量逐元素相等。

## 关键代码和环境

- [prepare_remote.py](prepare_remote.py#L35)：在已验收私有 vendor 的两份匹配 Vec 头文件中插入只读 printf；保存补丁、前后 SHA 和 host 未改证据。
- [graph_probe.py](graph_probe.py#L24)：真实 FIA、真实 o_proj、固定输入、fullgraph 编译、三次调用及私有完整输出；同时收集 `/kernel_meta` 和 `ASCEND_WORK_PATH/kernel_meta`，缺编译 raw/二进制元数据即失败。
- [apply_policy.py](source_policy/apply_policy.py)、[fault28_fia_policy.h](source_policy/fault28_fia_policy.h)、[fault28_causal_tile.h](source_policy/fault28_causal_tile.h)：版本与进程绑定、实际 M 选择边界、合法新 key 和共同 causal 规则。每个匹配的图编译调用持续触发；关闭后须新进程重新编译。
- [public_plan.py](analysis/public_plan.py)、[native_plan.py](analysis/native_plan.py)、[audit_case.py](analysis/audit_case.py)：独立 ABI、实际编译 raw/key、二进制与真正图执行绑定；不以静态注册清单替代 launch。
- [sdk_debug_tool_cpu.py](analysis/sdk_debug_tool_cpu.py)、[device_trace.py](analysis/device_trace.py)、[run_device_audits.py](analysis/run_device_audits.py)：官方标量解码、原始本进程 runtime 行交叉核验、实际 M/N 和无 SplitKV 检查。
- [graph_evidence.py](analysis/graph_evidence.py)、[verify_public_bundle.py](analysis/verify_public_bundle.py)：不依赖控制证据的首差推断与发布字节/物理盲验。
- [join_measured_chain.py](analysis/join_measured_chain.py)、[observer_neutrality.py](analysis/observer_neutrality.py)：在首差分析之后连接 Batch/M/标量/完整输出，以及检查 printf 观测中性。

环境限定为124、910B4、物理卡5、TP1，CANN9.1.0、torch2.10.0、torch-npu2.10.0.post4，FP16/BNSD、Q16/KV8/G2/D128、S1=S2=2048、causal sparse2、inner_precise0。共同核预算16|32；实际 `numOfFdHead=0`、全部有效分核 `s2End=0`，Reduce轴未切多核。SDK公开 serializer 的65个范围覆盖1,576字节、63个逻辑字段，并与实际编译器接收 raw/key 核对。

| 身份 | SHA-256 / 版本 |
| --- | --- |
| 官方 ops-transformer 源码 | `c77c0829f1f0f0b8f882db1d9b051a758b605472` |
| 原始完整 QKV 输入清单 | `509123dc725e351d6b10840919c2640b327a13eecceba2ca4a888c4a094f8e62` |
| 新 vendor 清单 | `205988e230b901421f14eb0844ce435b66647a5846324608d768d313a66f8cd2` |
| CPU FP64 采集器清单 | `06ce36bae3463f44a143047d184f53915dbea4dd04b39d79960518e6878d8010` |

早期 `grain_a1` 因隔离路径改变导致编译元数据快照缺失，严格验收失败；修复双路径搜索并增加缺失门禁后，用 `grain_a2` 新 attempt 完成正式矩阵，原失败与主动停止回执未覆盖。最初设备 printf 的逐 replay 时间窗配对失败也保留，最终只验收汇总对应。

全部实测遵守共享资源锁、逐次设备门禁和每窗口最多八组；只使用空闲卡5，卡2推理服务保留，本轮没有停止外部进程或容器。构建与观测均在任务私有目录，不改系统 SDK。完整 QKV、输出、权重、原始标量缓冲、完整日志及编译二进制留在124私有目录 `/home/fault28-fia-grain-20261109`，GitHub只提交实现、统计与审查后的元数据。
