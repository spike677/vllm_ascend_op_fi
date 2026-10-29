# Fault28 正式 B/F 原生统计独立推断

**两步均观察到差异，但四项统计没有独立定位到 attention 输出。** 冻结分析器只读取每个 arm 的 `config.json` 与原生 `dump.json/stack.json/construct.json`，两次结果均为 `ALIGNED_OBSERVED_API_STATISTICAL_DIFFERENCE`。它没有读取 runtime、selector、命中日志、host profile 或私人张量来决定首差。

每步 B/F 有 1621 个节点完整对齐，28 个 SDPA API 都有对应的 28 层 attention Module 父级，shape/dtype 和调用顺序兼容。模型 API 首个被观测差异都是序号 110 的 `Tensor.pow.7.forward`，父级为 `model.layers.1.post_attention_layernorm`，其**输入** Mean/Norm 已不同。因此不能把 `pow` 认作故障发生算子，也不能用其它证据把本报告的首差回填为 SDPA。

| 观测字段 | B | F |
| --- | --- | --- |
| `pow` 输入 Mean | -0.0007553291507065296 | -0.0007553283940069377 |
| `pow` 输入 Norm | 1081.07177734375 | 1081.071533203125 |
| `pow` 输出 Mean | 0.06966090947389603 | 0.06966088712215424 |
| `pow` 输出 Norm | 1586.7679443359375 | 1586.7664794921875 |

只读抽查原生 `Functional.scaled_dot_product_attention.1.forward`：B/F 输出 shape 均为 `[8,16,2048,128]`、dtype 为 FP16，四项有限统计完全相同：Max=0.640625、Min=-0.67431640625、Mean=0.00022029876708984375、Norm=339.25。四项统计相同不能证明元素相同。首个差异前另有 60 个不可观测统计字段，这也限制了更早位置的排除。每步共 787 个模型 API 有统计差异，196 个统计字段不可观测。

原生 dump 可见的是 PyTorch SDPA API，没有直接的 CANN `FlashAttentionScore` 记录。真实 CANN 调用、合法 tiling 与归约机制由别的证据验证，不能冒称已经在此 dump 中直接出现。实际诊断过滤排除了 0 个节点，因此首差位置并非删除了前序 SDPA 所致。

报告：

- [step0.json](step0.json)、[step1.json](step1.json)：独立原生统计推断。两步三件套字节相同，因此报告字节也相同；这两个路径本身不能额外证明执行次数，实际两 forward 的执行与独立 callback 由其它运行记录佐证。
- [summary.json](summary.json)：以上报告的紧凑摘要与 SHA。
- [formal_runtime_crosscheck.json](formal_runtime_crosscheck.json)：**独立的补充检查**。正式 B/F 各两 forward 的全部 28 层目标 Q/K/V/输出、layer1 完整 native 输入及 logits 均与此前诊断 OFF/ON 对应值一致。该检查不读 msprobe，也不修改 msprobe 的推断；它只说明统计采集未改变已验证的值路径。

分析器 SHA：`4fea04fc41e3fda4c8a526e59ee59e3852b6d3775f1bcead053ce1bdf8e0717b`。运行配置是 `task=statistics, level=mix, summary_mode=statistics, tensor_list=[]`，每 arm 两步。本目录报告不含张量二进制。当前证据支持“已发生且向下游传播的数值差异”，不满足“仅凭这套四项统计把首差准确定位到 attention 输出”的更严格要求。
