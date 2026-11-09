# 故障28：FIA 图模式设备归约粒度与 statistics/mix 证据

**2026-11-09，124卡5受控机制验收通过。** 固定真实 Qwen3 layer1 Q/K/V，在真实 TorchAir fullgraph FIA→o_proj 子图完成23组、69次调用。实际 Batch1→8、M512→128、同一 query63 的设备 Softmax 列数512→128及输出变化相互对应。

[实现、调用链、限制及关键代码](../../../experiments/fault28_fia_grain_20261109/README.md) · [联合验收](data/reports/measured_chain/acceptance.json) · [全矩阵执行验收](data/reports/execution/acceptance.json)

## B/F 与结果

[primary_BF.json](data/primary_BF.json) 标出主对照：

- **B**：[g_causal_base512_b8_r1](data/cases/g_causal_base512_b8_r1/)，B8、M512、causal规则启用。
- **F**：[g_batch_b8_r1](data/cases/g_batch_b8_r1/)，B8、M128、相同causal规则。
- **独立 F**：[g_batch_b8_r2](data/cases/g_batch_b8_r2/)。
- **重新编译 OFF**：[g_restore_b8_r1](data/cases/g_restore_b8_r1/)。

主 B/F 的完整目标 Attention 共4,194,304元素，差异140个、最大绝对差0.000244140625；真实 o_proj 差异6,141/2,097,152。最后 query无差异。两组 F 的全部8个请求 Attention 哈希均变化，独立 F/F及OFF恢复的整批/逐请求 Attention和projection哈希完全一致。每请求哈希与目标逐元素比较分别记录，不能互相冒充。

M-only 不裁剪尾块时无差异；同 M128/M256启用causal裁剪时分别有140/81个目标Attention差异。同causal规则改变M的主对照隔离了M选择。固定策略B1/B8的FIA目标逐bit相同；OFF的跨Batch o_proj本身有1,750个元素差异，故不将跨Batch projection变化全部归因FIA。

## 原始采集与首差

[data/cases](data/cases/) 保存23组正式配置、46套step0/1、rank0的原始 `dump.json/stack.json/construct.json`、完成回执、原生GE CSV、FX/GE图、实际编译 raw/key摘要与来源元数据。配置统一为 `statistics/mix/summary_mode=statistics/precision=high`；step是相同prefill子图回放，不是模型decode token。

采集使用与此前相同、统一处理浮点张量的任务私有CPU FP64增强。[采集器补丁和指纹](data/collector/)公开披露。原生GraphBoundary的output0由实际GE NetOutput直接映射到FIA output0；这是图别名推断，没有伪装成eager FIA API dump。

独立分析不读host命中、策略、设备标量或数值报告。物理盲验只保留统计、配置、采集器身份及实际FX/GE图/CSV，两步×两个F均得到 **FIA output0 Mean/Norm首差**：

- [本地发布字节与盲验总报告](local_blind_verification.json)
- [F的独立首差](local_fault_first.json)
- [独立F的首差](local_repeat_first.json)

原生六位GE CSV首先观察到下游MatMul；不能称默认统计已经直接定位FIA。[此前FP32失败证据](../fault28_fia_graph_statistics_20261106/limited_fp32/)保留。null、空输出、缺失值均保留不可观测；统计一致不等于逐元素相等。

可直接用公开数据重新推断，输出须指向新路径：

```bash
python data/analysis/graph_evidence.py \
  --baseline data/cases/g_causal_base512_b8_r1 \
  --fault data/cases/g_batch_b8_r1 \
  --output /path/to/new_first_difference.json
```

## 实测设备粒度与执行证据

[device/summary.json](data/device/summary.json) 覆盖19组私有实现、798条实际设备标量；另4组stock不含观察器。每组保留：官方SDK未编辑stdout、官方解析回执、原始本进程runtime printf行提取、原日志SHA/字节范围/逐行SHA和独立列数核验报告。

设备观察器在真实SoftmaxFlashV2Compute调用前记录实际columnCount/actualColumnCount，[补丁及前后指纹](data/build/device_grain_trace.patch)随包保存。采样限定request0/query-head0七个query；M128实际采到128/256/512，M256采到256/512。没有将源码推导但未采样的384列写成设备实测。

官方SDK解码与本进程runtime行的多重集合完全一致，每唯一标量形态重数3。printf异步刷出，**仅验收进程/图汇总，不建立逐条记录到单次replay的时间配对**。[原逐次时间窗失败](data/reports/validation_or_limits/pilot_device_evidence_a2.json)保留。

公开SDK serializer的65个范围覆盖1,576字节、63个逻辑字段；实际host raw、编译器接收的raw/key、编译二进制身份及实际静态图执行关联经过核验。共同核预算16|32，真实 `numOfFdHead=0`、有效分核`s2End=0`，Reduce轴未切多核。公开包能重算原字节的标量文本和ABI，不能代替远端完整原始缓冲及二进制核查。

[观察器中性报告](data/reports/observer_neutrality.json)确认旧无printf与本轮23组/69对调用的完整目标、整批/逐请求哈希和实际host raw/key完全一致。[27项分析及反例测试](data/reports/validation_or_limits/analytic_tests_linux_a1.json)通过，无跳过。

## 完整性与边界

`data/`共 **1,514个文件、31,108,919字节**，其中1,492份原始文件通过[source_mapping.json](data/source_mapping.json)映射；[SHA256SUMS](data/SHA256SUMS)覆盖除自身外所有导出文件。Git属性保持原始字节。目录外README和三个本地复核报告是发布说明，不属于远端导出清单。

| 项目 | SHA-256 |
| --- | --- |
| source_mapping.json | `03948b772eb9bf87c110d317c98d55c1797863f93458c80d4f08b3c37e3dbd52` |
| measured_chain/acceptance.json | `9fac19a48089b11ba589c278aefbaa7fc809b29eea53668e288f558661ee4309` |
| execution/acceptance.json | `c46b74e2817813e84f31bbb1d403fa5e7378322405c595a2ec04f811a3225ae2` |
| local_blind_verification.json | `eef92512f196cc409fe9e38974a01212488e40701b027e3e48d64795164f620c` |
| 传输归档（未提交） | `6d9fb31fa2f722c327e0ee39ee6dad103f7a4159b9ed37ff8594177372e149d6` |

早期grain_a1的编译元数据路径缺失、严格验收失败及主动停止回执保留在[superseded_capture_a1](data/superseded_capture_a1/)和[失败报告](data/reports/validation_or_limits/pilot_execution_a1.json)。修复后grain_a2重新执行全部固定矩阵，没有覆盖失败或改动数值输入。

这是当前CANN9.1的**受控FIA图子图机制**，不是历史盘古默认策略、完整模型FIA图或首token logits复现。实测的是Softmax列粒度，不是Reduce指令树；N也进入BMM1/BMM2，尚不能唯一归因某条Reduce指令或断言超过算子精度标准。

完整QKV、权重、完整输出、原始标量缓冲、完整DEBUG日志和编译库均留在124任务私有目录 `/home/fault28-fia-grain-20261109`，未提交到GitHub。采集只用空闲卡5，卡2服务保留，本轮未停止外部进程或容器。
