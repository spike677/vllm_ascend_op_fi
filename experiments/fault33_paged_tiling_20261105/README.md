# 故障 33：分页增量 Attention 的执行策略与数值差异

**124 的 910B4 已通过固定分页布局的算子级故障注入验收。** 在相同 B8、输入和 128-token 物理页下，将一次 KV 计算分块从 8,192 降至 1,024，实际循环从 2 轮增至 16 轮，16,384 个 Attention 输出中有 5,371 个数值改变。两次 B/F 比较中，msprobe 独立观察到的首差都落在原生 IFA 返回值。开启确定性后，固定策略的完整输出重复稳定，同一进程关闭控制后恢复。真实 Qwen3-0.6B 的适配模型也通过独立验收，首差在第 0 层原生 IFA 输出，8 个请求的 logits 均有变化，top-1 全部保持相同。[受控选择点](source_policy/apply_policy.py) · [算子验收程序](analysis/accept_replay.py)

| 验证 | 当前进度 | 可回答的问题 |
| --- | --- | --- |
| stock 发现 | 三组共 18 次真实调用已完成 | 改物理分页或 batch 时，同一请求的算子输出是否变化 |
| stock 的 msprobe 采集校验 | 另一次进程中完成两次同配置调用 | 能否在各 step 看见原生 IFA 的输入、输出和调用栈，重复调用的统计是否稳定 |
| 固定物理布局的受控 B/F | a4 正式验收通过，10 次原生调用 | 同一输入的完整输出变化，msprobe 首差与受控算子边界一致 |
| Qwen3 全模型 | 四个独立作业通过验收，两个 F 合计命中 56 / 56 次符合条件的 decode IFA | 全部 28 层和 8 个请求得到覆盖，输出差异传播至真实 logits |

## Stock 发现

实验使用 124 物理卡 5、CANN 9.1.0、torch 2.10.0、torch-npu 2.10.0.post4。Q 为 FP16 BNSD，每个请求 16 个 query 头、1 个 query token。K/V 各有 8 个头，head dim 为 128。`inner_precise=0`，scale 为 `1/√128`。各请求用独立、固定的 CPU 随机种子产生逻辑输入。PyTorch 确定性开关已启用，18 个原生 IFA 窗口的 NNOP 日志均记录 `deterministic value is 1`。[B1 短序列](setup/stock_pilot_b1.json) · [B1 长序列](setup/stock_long_b1.json) · [B8 长序列](setup/stock_long_b8.json) · [日志与回执](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/stock_discovery/index.json)

| 保持 batch 和上下文长度固定 | 物理页（tokens） | 完整输出变化元素 | 请求 0 变化元素 | 最大绝对差 |
| --- | --- | ---: | ---: | ---: |
| B1，长度 2,048 | 128 → 256、128 → 512 | 每次 0 / 2,048 | 0 | 0 |
| B1，长度 16,384 | 128 → 112 | 333 / 2,048 | 333 | 0.0000152587890625 |
| B1，长度 16,384 | 128 → 80 | 454 / 2,048 | 454 | 0.000030517578125 |
| B8，长度 16,384 | 128 → 112 | 435 / 16,384 | 48 | 0.000030517578125 |
| B8，长度 16,384 | 128 → 80 | 540 / 16,384 | 51 | 0.000030517578125 |

**固定物理页大小，对齐 B1 和 B8 的目标请求 0。** 长度均为 16,384，目标的完整逻辑 Q/K/V 逐 bit 相同。页大小 128、112、80 下，目标的 2,048 个输出元素分别有 **912、910、927** 个发生变化，三组最大绝对差均为 0.000030517578125。实际 launch key 从 B1 的 `10000000000300000` 变为 B8 的 `10000000000200000`。这里 batch 及其带来的原生调度变化共同生效，独立识别 SInner 因果作用要靠下面的固定布局实验。[目标请求逐元素比较](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/stock_discovery/stock_batch_target_comparison.json)

同一 B8 组内，三个页大小的 launch key 相同，均为 20 blocks、schedule mode 1。实际 SDK 几何仍有变化，页大小 128 时 BMM1 `baseN` / BMM2 `baseK` 为 128/128，在 112 和 80 时均为 16/16。短序列则提供了布局变化而完整输出相同的对照。[B8 原生日志摘录](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/stock_discovery/stock_long_b8_native_a2.json) · [短序列比较](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/stock_discovery/stock_pilot_result_a2.json)

## 调用链与两种“块大小”

`block_size` 是 KV 缓存中一页容纳的 token 数。stock 探针先把逻辑 K/V 分页、补齐末页并反转物理页序，用二维 int32 `block_table` 记录位置。逻辑 token `t` 的物理页为 `block_table[request, t // block_size]`，页内偏移为 `t % block_size`。探针在调用前通过该表还原所有有效 K/V，与源张量逐元素核对，同时把真实有效长度传给算子。每次调用前后均校验完整 Q、物理 K/V 和表的字节哈希。[分页和反查](stock_probe.py#L40) · [原生 IFA 调用](stock_probe.py#L119)

```text
真实分页 Q、物理 K/V、block_table、actual_seq_lengths
  → torch_npu.npu_incre_flash_attention（一次调用，返回一个 Tensor）
  → op-plugin 的 npu_incre_flash_attention_symint
  → aclnnIncreFlashAttentionV4
  → CANN IncreFlashAttention host tiling → 该调用选中的原生 kernel
  → 原生 Attention 返回值 → 调用者
```

op-plugin 版本为 `8b9c8534fa41eff367a41c155843daa530ab3a08`，对应固定 CANN 源码 `c77c0829f1f0f0b8f882db1d9b051a758b605472`。通过 ATB 的 `_npu_paged_attention` 则使用 `out` 参数写回，是另一条公开接口。[op-plugin 的 aclnn 转发](https://github.com/Ascend/op-plugin/blob/8b9c8534fa41eff367a41c155843daa530ab3a08/op_plugin/ops/opapi/FlashAttentionKernelNpuOpApi.cpp#L1669-L1709) · [ATB 接口](https://github.com/Ascend/op-plugin/blob/8b9c8534fa41eff367a41c155843daa530ab3a08/op_plugin/ops/atb/PagedAttentionAtb.cpp)

`singleProcessSInnerSize` 是 kernel 一次处理的 KV 序列计算分块，可覆盖多个物理页。匹配 B8 key 的 FP16 PA、BNSD、非 FlashDecode 实现按此大小推进 KV 偏移，跨各轮更新 softmax 的 max、sum 和输出累积，最后归一化。改变轮次及分组可以改变有限精度舍入；具体是否产生输出差异取决于输入和实测。这个 kernel 的 `SoftmaxFlashV2Compute` 根据运行时形状生成 softmax 子 tiling，单凭 host softmax 字段无法确定实际执行了哪条浮点指令。[固定 host 选择](https://gitcode.com/cann/ops-transformer/blob/c77c0829f1f0f0b8f882db1d9b051a758b605472/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp#L2597) · [匹配 kernel 的 softmax 更新](https://gitcode.com/cann/ops-transformer/blob/c77c0829f1f0f0b8f882db1d9b051a758b605472/attention/incre_flash_attention/op_kernel/arch22/incre_flash_attention_split_Bbn2s2_Us2.h#L2707)

Stock 日志中的 `PA FlashDecode is enabled, sInnerLoopSize is ...` 容易误导。匹配源码在 `FillTilingSplitKV` 中仅凭 `pageAttentionFlag_` 打印该行，写入的是 `splitKVParams.sInnerLoopSize`。识别是否选中 FlashDecode 需看实际 key、对应 kernel 及 host 分支；对本次受控分块则读取 `singleProcessSInnerSize` 和 `sInnerLoopTimes`。[日志打印条件与写入字段](https://gitcode.com/cann/ops-transformer/blob/c77c0829f1f0f0b8f882db1d9b051a758b605472/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp#L2979-L3005)

## 受控注入的实现与验收门槛

正式 B/F 固定 **B8、长度 16,384、页大小 128**，保持同一调用的真实 Q、K/V 和映射相同。私有 vendor 用同一个 host 二进制执行 OFF 和 F，保留原生设备 kernel。

1. Python 在真实 IFA 调用前建立仅本次有效的 decode scope，包含 PID、原生线程 ID、唯一 token、调用 ID 和几何。host 还检查真实输入描述、八项有效长度、910B、FP16、GQA 16/8、无量化和非 SplitKV 等条件。**每个符合条件的调用均可触发**，没有“只打一次”的计数；未命中则透传原生选择。[调用 scope](replay.py#L50) · [公共上下文及真实长度检查](source_policy/fault33_pa_policy.h#L47)
2. 注入点位于 `IFATiling::CalcInnerSize` 完成原生分页对齐之后、计算 loop/tail、UB 等依赖之前。OFF 返回原生值，正式 F 请求 `inner1024`，即 1,024 个 KV token、8 个物理页。候选必须为 128 的倍数、小于本轮处理序列且不大于原生值，后续 tiling、矩阵乘、workspace 和 launch 继续由原生代码生成。另一个已编码候选 `inner512` 不在本轮正式 F 范围内。[补丁落点](source_policy/apply_policy.py#L61-L76) · [候选合法性条件](source_policy/fault33_pa_selector.h#L9-L24)
3. 独立验收先把 stock 与私有 OFF 的完整输出、key、blocks、原生日志中的 tiling 大小及 SDK 字段对齐。再检查 B/F 的每个实际 launch 与该次公共 `gert::TilingContext` 记录对应。公开 serializer schema 可从私有 host 最终 3,000 字节 raw 解码已登记的 72 个字段；设备入口完整 kernel argument 字节仍不可直接观察。对固定布局 B/F 比较各请求完整输出、FP64 参考误差与 OFF 恢复，并用原始 msprobe 三件套单独定位首个可比统计差异。[原始日志与 host 回调绑定](analysis/native_evidence.py#L25) · [公开 serializer 解码与验收](analysis/accept_replay.py) · [构建和可观测边界](source_policy/BUILD.md)

## 受控算子实测结果

**a3 在私有 OFF 桥接处被拒，a4 修复守卫后重新验收。** a3 的两次 stock 与两次 OFF 输出均相同，但 OFF 的 `native_scope=false`，因此止于桥接。源码显示，普通非 TND、无 RoPE、非 KV-NZ 的分页路径会设置 `inputLayout_`，而内部 `inputKvLayout_` 可保留默认枚举 `BSH_BSND`。a4 将这一内部条件改成该路径的精确默认值，并逐项记录 24 个原生条件；公开 BNSD 属性、实际 rank-4 K/V 形状及八个请求的有效长度仍单独核对。正式使用同一份 a4 私有库，vendor manifest SHA256 为 `f48a82e89fdf7bbfafba5ac3e05334dbde01b422c55dc726f786152db5874cfe`。[a3 原始拒绝记录](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/rejected_a3_bridge/README.md) · [a4 构建与包指纹](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/build_a4/README.md)

正式报告为 `PASS_CONTROLLED_REPLAY_AND_STATISTICS`，共 10 次真实 IFA 调用。先以两次 stock 和两次私有 OFF 比对完整 Attention、完整合成投影和可见的 launch 与 SDK 字段，再做两次 F 和一个进程内的四次恢复序列。10 次的实际 launch key 均为 `10000000000200000`，NNOP 确定性标记均为 1。[完整算子验收报告](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/reports/replay_a4_acceptance.json)

| 同一私有库与物理输入 | B / OFF | F / inner1024 |
| --- | ---: | ---: |
| 每个 KV 计算块 `singleProcessSInnerSize` | 8,192 | 1,024 |
| 实际循环次数 `sInnerLoopTimes` | 2 | 16 |
| 尾轮有效 token 数 | 8,192 | 1,024 |
| Workspace（字节） | 20,837,376 | 17,396,736 |
| Attention 对 CPU FP64 参考的最大绝对差 | 0.00001860782504081726 | 0.000020470470190048218 |
| Attention 对同一参考的平均绝对差 | 0.0000029160983103793114 | 0.0000029239326977403834 |

B/F 的完整逻辑及物理 Q/K/V、块表和投影权重均相同。**Attention 有 5,371 / 16,384 个元素数值改变**，最大绝对差为 0.000030517578125；按请求 0 至 7，变化数分别为 `703、653、572、593、647、784、746、673`。合成线性投影有 996 / 2,048 个元素变化，最大绝对差为 0.00006103515625。上表的参考值由 CPU FP64 计算，探针将参考和实测输出转到 FP32 后计算误差。本组输入的两项误差均略有增加，这组观测只适用于该输入和精度设置。[参考与误差口径](stock_probe.py#L66) · [真实返回值与合成投影](replay.py)

**独立 msprobe 首差位于原生 IFA 输出。** 两组 B/F 重复都先观察到 `NPU.npu_incre_flash_attention.0.forward` 输出的 Mean、Norm 改变，Mean 从 `-0.00022450658434536308` 变为 `-0.00022453052224591374`，Norm 从 `1.654924988746643` 变为 `1.6549243927001953`。该节点输入的 14 个有限统计均一致，int32 块表 Mean、Norm 为 null。B/B、F/F 两组对照均未观察到有限统计差异，完整原生输出各自逐 bit 重复。比较器只按原始统计、栈和构造关系定位首差；host 的命中记录和私有输出另外用于证明计划及完整字节。[独立比较实现](analysis/msprobe_compare.py#L205) · [算子验收报告](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/reports/replay_a4_acceptance.json)

同一进程按 `OFF → inner1024 → inner1024 → OFF` 调用，四次实际计算分块依次为 `8192 → 1024 → 1024 → 8192`。两次受控调用均命中，末次 OFF 的完整 Attention、合成投影及 3,000 字节公共 host raw 均恢复到首次 OFF。实验证明，在固定物理输入上切换 **SInner 及随之由原生代码重算的整套 tiling 参数** 足以造成可重复的数值偏差，并传到这个下游投影。现场证据能定位到 IFA 返回边界。对设备内部某一条 Reduce 指令的单独归因、真实服务调度以及模型精度影响，需要各自的证据。

## msprobe 与真实模型

Stock 采集校验使用 B8、长度 16,384、128-token 页的同配置重复，在 step 0 和 step 1 各捕获一次真实 `NPU.npu_incre_flash_attention.0.forward`，父节点均为 `Module.replay.ReplayModule.forward.0`。每步保留原生 `dump.json`、`stack.json`、`construct.json`，采用 `statistics/mix` 及已校验的任务本地 high FP32 统计采集器。两次 Attention 和固定投影的完整输出均逐 bit 相同，原生 IFA 的 14 个可比输入统计和 4 个输出统计均一致，int32 块表的 Mean、Norm 为不可观测。这两步仅建立 stock 同配置对照。正式 a4 的 B/F 采用上节独立采集的两对统计。离线比较报告 `stock_statistics_repeat_compare.json` 的 SHA256 为 `53470aad3c935a8917033dbe81bb0f07e6d40f334b63e31113fb9040c0e694be`。[采集配置](setup/stock_statistics.json) · [统计收据](replay.py#L67) · [独立比较逻辑](analysis/msprobe_compare.py#L205)

采集沿用 msprobe 的 `statistics/mix`、原始节点名和 Max/Min/Mean/Norm 四类统计。任务本地 collector 对原始 FP16/BF16 值的分离副本以 FP32 归约，并保留原模型返回张量、dtype 和 shape；四组使用同一实现。补丁 SHA256 为 `8239db6dbb8d54e77d6e7960fd1ff15121c13bfe5d7327557031b3d66d43975a`，实际载入的 processor SHA256 为 `3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141`。四个摘要可能漏掉逐元素差异，null 表示不可观测；因此定位使用原生统计，数值相等与请求影响另用完整字节哈希和私有 logits 验证。[collector 实现说明](../fault28_native_tiling_20261029/analysis/collector_high/README.md) · [实际补丁](../../artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029/collector/collector.patch)

全模型方案使用真实 Qwen3-0.6B 的 8 个固定请求。原生 HF SDPA 在所有 28 层 prefill 16,383 tokens，下一步使用同一已知 token，形成长度 16,384 的真实 KV 缓存。decode 适配器把每层实际收到的 K/V 按 128-token 页重排，校验每个请求的逆映射与 prefill 前缀，调用一次真实 IFA，把返回值交给该层 `o_proj` 和后续模型直至 logits。stock 全模型预检已在一轮 decode 中将全部 28 个 IFA 节点与其真实 Qwen3Attention 父模块逐层对应，覆盖 224 个请求与层的组合。[全模型预检](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/stock_model_pilot/README.md)

### 全模型 B/F、重复与恢复

四个独立作业按 B、F0、F1、OFF 恢复的顺序执行，使用同一个模型、八个已知输入前缀、物理页表和 a4 私有库。四份报告的 prefill KV 和 prefill logits 哈希相同。独立验收结果为 `PASS_ADAPTED_MODEL_AND_STATISTICS`，报告 SHA256 为 `e4c3af499ea9081dba7a44e8dac3f5da0bec2de1335157d6e614325e58af8aaf`。[正式模型验收报告](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/reports/model_a4_acceptance.json) · [四组公开统计、运行记录和源码清单](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/model_a4/INVENTORY.json)

| 四个作业的实际调用 | 数量及结果 |
| --- | --- |
| 原生 HF SDPA prefill | 112 次，每组 28 层，注入仅在后续 decode 建立 scope |
| 分页 IFA decode | 112 次实际 launch，均为 key `10000000000200000`、20 blocks、schedule mode 1，原生确定性标记均为 1 |
| F0、F1 | 每组全部 28 层各命中一次，共 **56 / 56 次**符合条件的 decode 调用；各层计算分块 1,024，16 轮 |
| B、OFF 恢复 | 共 56 次符合条件的 decode 调用执行原生选择；各层计算分块 8,192，2 轮 |
| 请求覆盖 | 每次实际 IFA 处理同一批 8 个请求。四组共 896 个“请求 × 层”组合，其中受控组 448 个 |

**两次 B/F 比较的 msprobe 首差都出现在第 0 层 `model.layers.0.self_attn` 的 `NPU.npu_incre_flash_attention.0.forward` 输出。** 全部原生 API 与调用栈对齐后，保留诊断节点的全 trace 与单独比较模型 API 得出同一首差。第 0 层实际 Q、逻辑及物理 K/V 的完整及逐请求哈希相同。14 项可比输入统计相同，另 2 项为不可观测。输出 Mean 从 `-0.001310580875724554` 变为 `-0.0013106090482324362`，Norm 从 `14.949091911315918` 变为 `14.94908618927002`。Max、Min 相同。每层的 IFA 输出哈希与该层 `o_proj` 实际消费的输入字节相符。[逐层 decode 调用](model_adapter.py#L215) · [独立 msprobe 比较](analysis/compare_model.py#L99)

在每一对 B/F 中，28 层的 Attention 输出和 `o_proj` 输出均在全部 8 个请求上发生字节变化。第 1 至 27 层的输入已经出现上游差异，各层仍验证实际策略命中，后续输出应结合传播解释。最终 8 × 151,936 个 FP16 logits 中有 **1,035,472 / 1,215,488** 个数值变化，8 个请求均受影响，最大绝对差为 **0.1171875**。按请求 0 至 7，变化数分别为 `138093、120883、118007、147270、119311、136197、109528、146183`。每个请求在两侧均只有一个最大值，**8 / 8 个 top-1 token 相同**。这是固定已知前缀下的一步 decode 观测；生成长文本或任务准确率没有在这组实验中测量。[完整 logits、top-1 和逐层比较](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/reports/model_a4_acceptance.json)

F0/F1 对照与 B/恢复对照分别取得相同的真实分层输出、完整 logits 字节及逐层公开 host raw，且各自的 msprobe 比较均未观测到有限统计差异。模型 OFF 恢复使用新进程，前面的算子复验还单独验证了同一进程关闭控制后的恢复。[模型验收程序及复核命令](RESUME.md)

**模型采集的中断与资源操作。** 首次四组作业 `model_a4_a1` 在 B 完成后遭遇外部卡 5 ATK 占用，F 期间资源门禁中止作业，其中的 B 和 F 片段均排除。重新采集的四个作业各有独立清理证明，时间不重叠，冻结源码清单逐字节一致。依用户授权，正式 B 启动前工具核验并停止了一个卡 5 专用 ATK；另外三个正式作业的启动回执未记录停止操作。控制器执行期间继续监控设备占用。[B 启动前 ATK 停止回执](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/resource_prelaunch/model_baseline_a4_a3.json) · [四份作业与资源回执](RESUME.md)

本次验收覆盖显式接入分页 IFA 的真实 HF 模型及其符合条件的 decode 调用。对历史 MindIE/vLLM 事故或自然服务调度进行归因，仍需服务日志将目标请求、同行、物理页和策略选择逐次关联；当前证据也不能确定历史事故中某条 Reduce 指令的贡献。[故障 28 的证据边界](analysis/FAULT28_REVIEW.md)
