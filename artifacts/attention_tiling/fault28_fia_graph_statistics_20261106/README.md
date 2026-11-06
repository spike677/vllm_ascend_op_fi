# 故障 28：FIA 图模式 statistics/mix 证据

最终状态：`PASS_CONTROLLED_FIA_GRAPH_CAUSAL_MECHANISM`。124 物理卡5/910B4、CANN9.1，最终 CPU FP64 统计轮 23组/69次真实 FIA 图执行，19组私有路径的 public raw、63个逻辑字段、编译器输入和实际任务均核验。另保留原 FP32 轮的计算和观测失败；它没有被标为全矩阵通过。

[实现及范围](../../../experiments/fault28_fia_graph_20261106/README.md) · [独立分析方法](../../../experiments/fault28_fia_graph_20261106/analysis/README.md) · [正式验收](reports/acceptance.json) · [统计不改变计算](reports/observer_neutrality.json)

## 主 B/F 与对照

| 数据 | 路径 | 含义 |
| --- | --- | --- |
| B | [g_off_b8_r1](cases/g_off_b8_r1/) | B8，原生 M512，关闭干预 |
| F | [g_batch_b8_r1](cases/g_batch_b8_r1/) | 同 B8/同 QKV，实际 Batch 策略 M128 + causal 尾宽 |
| F 独立重复 | [g_batch_b8_r2](cases/g_batch_b8_r2/) | 新进程、新编译，完整输出与 F 相同 |
| B1 | [g_batch_b1_r1](cases/g_batch_b1_r1/) | 同目标 QKV，实际 B1 选择 M512 |
| OFF 恢复 | [g_restore_b8_r1](cases/g_restore_b8_r1/) | 新进程重新编译 OFF，完整恢复 |
| 原 FP32 受限对照 | [limited_fp32](limited_fp32/) | 同 M256 的 B/F 原始采集，FIA 统计未分辨，严格失败保留 |

[主对照独立首差](reports/automatic_same_batch_r1_msprobe.json)、[第二次 F 首差](reports/automatic_same_batch_r2_msprobe.json)都在两步中由实际图边映射到 `FusedInferAttentionScore:0` 的 Mean/Norm。完整目标 Attention 4,194,304 个元素中有140个变化，最大绝对差0.000244140625；同 B8 的真实 o_proj 有6,141个元素变化。最后 query 不变，不据此声称第一个生成 token logits 改变。

`cases/` 包含全部23组，保留 stock/OFF 独立桥接、M512新变体兼容、M-only、同M causal宽度、固定策略跨Batch、自适应、重复和恢复。每组一次 fullgraph 编译、一次 warmup、两次回放，`observation/step0/1` 对应后两次完整 prefill，不是 token 步数。

## 采集和定位

配置为 `task=statistics, level=mix, summary_mode=statistics, precision=high, async_dump=false`。原生 dump/stack/construct/config/receipt 字节不改写。Python层是原生 GraphBoundary module 记录，slot0 直接暴露图内 FIA output0；通过实际 FX/GE 输出边定位，没有把它伪装成 eager NPU API dump。图内官方 GE CSV 同时保留，它的六位显示精度在本例首先分辨出下游 BatchMatMul。

`collector/` 保存任务私有 CPU FP64 四统计补丁、实际 processor、包指纹和真实 CPU 方法测试。它对所有真实浮点张量统一统计 detached CPU 临时副本，不改原张量、dtype、shape 或计算输出。原 FP32 统计轮 M256 的 FIA 差异被四统计舍入掩盖，`limited_fp32/reports/failure.json` 与原 B/F 均保留。最终统计方式变化前后，全部23组/69对调用的完整目标及所有请求输出哈希均一致。

null、bool/整数缺失统计、空 LSE 均保留不可观测。有限统计相等不证明张量相等。跨 Batch 的聚合统计不直接比较同一 query；跨 Batch 完整 Attention 比较与同 B8 的独立首差分开。固定 OFF 时 o_proj 的 B1/B8 已有1,750个元素变化，不能把所有跨 Batch 下游差异归因到 FIA。

## 文件与校验

- `reports/`：各 case 审核、全部目标逐元素比较、独立统计推断、固定位置 FP64 参考、严格总验收。
- `cases/*/msprobe_graph/`：实际 FX/GE 图及三次图迭代的原生 CSV；没有混入统计采集器自身的 eager 算子来冒充模型节点。
- `cases/*/host_events.jsonl`：最终公开 host raw 与 typed 字段；独立 SDK serializer layout 位于 `build/`。它们不是直接读取的设备入口参数。
- `compiler_binary_metadata.json` 是保留原字节的实际编译元数据；`compiler_tiling_excerpt.json` 明确是审查摘录，保存实际 key/raw/block/schedule 和原文件/生成 header SHA，不冒充完整原始 isolate JSON。
- `runtime_source/`：本轮冻结的实际采集 harness/config。`build/`：源码、包、依赖、ABI、CPU构建及私有阶段指纹。无设备二进制。
- `source_mapping.json`：原路径、缩短后的发布路径、字节数和 SHA。改短文件名时未改写原生内容。
- `SHA256SUMS`：本目录全部交付文件的校验（不包含校验文件自身）。

完整 QKV、整批计算输出、模型权重、完整 DEBUG 日志、`.pt/.bin/.o/.so/.run` 不提交。少量差异数值和索引属于运行时精确核验摘要。实际受控范围是 Qwen3第2层已捕获输入及其真实 o_proj 的 FIA 子图，不是自然服务、完整模型或历史盘古缺陷复现；动态 N 宽度来自实际 key/raw 与执行源码推导，不是设备内部 trace，未唯一分离某条 Reduce 指令。
