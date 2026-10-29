> 本文件保留故障28的前置只读检查结论。后续方案A已完成124实机与高精度统计独立定位，当前结论、实现和限制见[最终实现README](../../experiments/fault28_native_tiling_20261029/README.md)。初始检查记录不追写成事后通过。

# 故障 28：FA/FIA Tiling 切分策略异常检查

2026-10-29。**已完成方案、现有归档和 124 环境的首轮只读检查；尚未实现注入或运行故障验证。** 机理方向有源码候选支持，但当前方案仍缺真实选路、Batch1/8 tiling profile、匹配版本的 host 选择器和精度策略校准。当前状态为 `DESIGN_CHECKED_REQUIRES_RUNTIME_PROFILE_AND_SELECTOR_CALIBRATION`。

本报告的“故障 28”指用户附件中的 `attention_tiling_reduction_mismatch`。原仓库其他编号空间中的 `FM-28` 是另一项 epsilon ABI 案例，不能据此替换本故障。

原方案保存在 [input_plan.txt](input_plan.txt)，输入指纹及本轮环境/源码指纹见 [check.json](check.json)。本轮未导入 torch/NPU、未启动模型或计算、未停止服务、未修改远端文件，也未提交 GitHub。

## 核心结论

若继续要求“msprobe 的原生首差异位置与真实故障原理一致”，正式能力应优先采用 **A：实际 host tiling / kernel 策略路径**。B 的分块重算可以作为独立的传播代理，但不能自动升级为 CANN 原生 FIA 故障复现。

| 检查项 | 本轮结果 | 验收前仍需补齐 |
|---|---|---|
| M/S1 与核内 S2 的关联 | CANN 9.1 arch22 PFA 存在直接源码候选 | 目标模型实际命中该分支，以及目标 query 的归约 trace 确实改变 |
| 历史 Batch1/8 profile | 有界归档检索未找到；附件仍含“实测值 A/B”占位 | tiling key、S1/S2、核映射、workspace 与版本指纹 |
| 124 Attention 调用 | 已找到 FIA/FIA-v2/FA/PA 静态分支 | 一次实际运行中的算子、阶段、dtype、layout、mask、精度和 kernel |
| 真实 host 修改入口 | 找到已编译 FIA host 库，未在检查目录找到匹配 host C++ 源码 | 可构建且与当前 ABI/版本匹配的选择器或受支持的可控入口 |
| 代理 B 的正常对照 | 原方案缺少 | 同一代理的正常 schedule 与异常 schedule 配对 |
| msprobe 首差异 | 尚无本故障运行证据 | 原生 FIA 与代理重算/替换边界必须区分 |

## 124 上实际查到的内容

检查快照为北京时间 10:55:15：卡 2/5 均为 910B4、Health OK、无 NPU 进程。具有设备访问能力的背景容器仍在；这只是空闲快照，不代表设备预留或独占。

已核查版本为 torch 2.10.0、torch-npu 2.10.0.post4、vllm-ascend 0.23.0。原实验冻结镜像中的 CANN 是 9.1.0，内部构建 `V100R001C11SPC001B243`；宿主机目录则是 CANN-7.0，不能把宿主环境当作该实验的编译环境。镜像、Python 源码、host 库和候选 kernel 的 SHA 均列在 check.json。

Python 包根目录是 `/home/tik-constfold-cann91-overnight-20261022/venv/lib/python3.11/site-packages`：

| 代码位置 | 已核实含义 |
|---|---|
| `vllm_ascend/attention/attention_v1.py:1234,1283,1303,1336` | `forward_fused_infer_attention` 存在 FIA-v2、FIA 和 DeviceOperator 分支 |
| 同文件 `:1397` | 存在 `torch_npu.npu_fusion_attention` 的 FA 调用 |
| 同文件 `:1468` | DecodeOnly 在相应条件下走 paged attention，其他条件走 FIA |
| `vllm_ascend/device/device_op.py:92` | A2 路径调用 `torch_npu.npu_fused_infer_attention_score` |
| `torch_npu/op_plugin/meta/_meta_registrations.py:1951` | FIA meta 参数定义；不是 native tiling 实现，也未提供直接 tiling-key override |

归档及当前静态源码的 causal、无 sliding-window 候选会传 TND、sparse_mode=3 和实际 Q/KV 长度；sinks、滑窗、非 causal、graph 等条件仍会选其他分支。因此不能在没有运行轨迹时认定“目标一定经过 FIA-v2 Python 函数”。

### 真正相关的 prefill 源码候选

以下文件相对镜像内 `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/prompt_flash_attention/`：

| 文件及位置 | 观察到的关联 |
|---|---|
| `arch22/prompt_flash_attention_s1s2_bns1_mla.h:417` | S1 基块尺寸从 `PFAcoreParams.s1BaseSize` 读取 |
| `arch22/prompt_flash_attention_var_len_score_sab.h:236–238` | causal S2 上界取 S1 块末端与实际 S2 长度的较小值 |
| 同文件 `:309–330` | core 先分配 B/N/G/S1 任务，再在各任务内部遍历对应的 S2 范围 |
| `arch22/prompt_flash_attention_bnstilling_n_s_no_tail.h:391–396` | outer offset、outer size 和窗口参数参与 inner-loop end_idx 计算 |
| 同文件 `:155–160,209–214,258–309` | outer size 参与 softmax shape、matmul tail 及首次/后续 softmax 更新组织 |

这支持“S1/M 块影响当前任务内部 S2 工作组织”的候选机制，**不证明 Batch1/8 已改变该块，也不证明目标末尾 query 的运算顺序或输出一定不同**。必须记录该 query 实际处理的分块、合并顺序和舍入点；全局 tile 或 kernel key 变化本身不足以证明这一点。

此前审查中的 `incre_flash_attention_allvec_new.h::CombineSplitKVRes` 涉及 SplitKV 合并，应排除出本故障的目标证据。附件历史条件明确 Reduce 轴未进行多核切分；后续需核对每个目标 query 的归约是否仍在单个任务内，不能用跨核 SplitKV 故障替代。

### Host 侧仍缺的内容

FIA 编译来源字符串出现在 `op_host/lib/linux/aarch64/libophost_transformer.so`，包括 nonquant 与 fused-infer tiling 实现名称。检查到的 op_host/op_tiling 目录没有相应 host C++ 源码。通用 `liboptiling.so` 也不能直接认定为 FIA 的修改入口。

在 A 路径中，需要取得匹配当前构建的 host 源码/构建依赖，先通过关闭开关时的正常门槛，再同时验证 tiling key、S1/S2、workspace、core mapping 与 kernel 兼容性。不得只改不透明 buffer 中某个字段，或以切换 inner_precise 代替 tiling 故障。

## 原方案需修订的验收逻辑

### 1. M 分组不是独立数值致错因子

若固定 query 的 Q/K/V、合法 key、运算顺序与舍入位置都相同，只改变 query 行的分组标签不会改变其结果。需要证实 M 策略改变了该行的实际算术路径；不一定必须表现为名义 S2 tile 数字变化，也可能是 QK/PV 的内积归约、online-softmax 合并树或中间舍入变化。

只重算一个 query 时，M 参数若没有导出任何实际计算差异，就是标签。末尾 query 在两种 M 分块下可能仍使用相同完整 key 范围；不能为得到差异而任意增加 FP16 cast 或挑选最容易出差的 tile。应先冻结合法且有依据的 profile，再允许结果为“当前条件未复现差异”。

还要先确定真实受影响的 query/tile。仅替换末尾 query 是缩小的传播实验；若原始变化发生在其他 query 并经后续层传播，该单行代理并不等价于完整 tiling 故障。

### 2. 代理必须有正常 schedule 对照

原生 FIA 与 Python/其他算子重算之间可能本来就有偏差。增加 `P_A8`（同一代理、正常 schedule）与 `P_B8`（同一代理、仅改变已校准 schedule），分开报告：

- `P_A8 − Native8`：代理实现与原生 kernel 的偏差。
- `P_B8 − P_A8`：代理内部的分块变化效应。
- `Fault8 − Native8`：模型消费边界的最终注入效应。

FP16 输入不代表 QK、exp、m/l/u、PV、rescale 全用 FP16。两端必须保持相同的实际 precision policy；高精度参考也必须从同一份已表示的输入值开始，不能把输入转换误差算作 tiling 故障。

### 3. 代理不能冒充原生 FIA 首差异

方案 B 先执行原生 FIA，再重算并替换切片。若原生 FIA 返回值相同，首个真实数值差异应在后续代理运算或替换处，不能将修改后的 wrapper 返回值标成 CANN FIA 原生输出。

采集应保留 `NativeFIA → ProxyRecompute → Splice → O projection` 的实际节点与调用栈。若验收要求“原生 FIA 本体首先产生差异”，B 路径不满足，应采用 A。命中日志只能核对控制状态，不能指定原生统计的首差异结论。

### 4. 输入和目标映射需逐项对齐

绑定目标请求已知前缀的 prefill 末尾 query，而不是输入位置 0 或整个 packed tensor 的最后一行。核对逻辑 Q、全部相关 K/V、绝对位置、RoPE、scale、GQA/MQA head、mask/窗口、padding、实际序列长度与 paged-cache 映射。每个合法 key 必须恰好参与一次，既不能漏算，也不能重复或跨请求。

Batch1/8 在第二层的 QKV 可能已因上游计算不同而出现差异。应先做逐目标逻辑输入比较；不满足等值时，先在捕获的相同 QKV 上做算子级 replay/calibration，不能继续认定 FIA 是第一处故障。真实模型传播仍单独验证，不改写输入来伪造完整模型原生条件。

## 现有 Batch 实验不能代替本故障证据

有界检索未找到附件所述 9 月 19–26 日、38Bv2 的原始 QKV 等值包或 B1/B8 tiling 数值；这不排除它们在其他位置。附件中的历史描述暂按待核实来源记录。

[10 月 28 日 partial11 报告](../batch_input_outliers_20261028/validation_partial11/README.md)属于另一个 32B-W8A8/TP2 实验。其已观察差异区间为 o_proj 量化矩阵乘、all_reduce、残差/归一化到 gate_up 入口；该观测没有确定 FIA 首差异，更不等价于完整 Q/K/V 对齐。不能用它或故障 26 的 FIA writer 记录填补故障 28 的机制证据。

## 修订后的最小实验矩阵

| 条件 | 目的 |
|---|---|
| Native Batch1，至少重复两次 | 单请求基线与重复性 |
| Native Batch8，至少重复两次 | 原有 batch 差异与重复性 |
| Batch1 armed | 验证不匹配透传 |
| Batch8 正常 schedule 代理 P_A8 | 单独测代理固有偏差；仅 B 路径需要 |
| Batch8 已校准异常 schedule 代理 P_B8 | 同一代理仅变分块路径；不冒充原生 A |
| Batch8 恢复正常策略/关闭注入 | 核对恢复及无残留状态，使用与相应基线一致的路径 |
| 同一捕获输入的高精度参考 | 判断数值偏差方向，不能代替机制证明 |

另做离线 M-only 负对照：每 query 的操作 trace 相同，仅改变 M 分组，结果必须一致。A 路径增加真实 tiling/kernel-key 与 profile 的关闭/开启/恢复记录。持久模式需每次满足完整条件均命中，逐调用、逐 TP rank 核验；保留不匹配透传。

模型测试固定 batch 同行、顺序、长度、前缀、调度、执行模式、生成条件与 checkpoint。分别报告 attention、prefill next-token logits、Top-1 和固定前缀文本结果；有 attention 差异不意味着 Top-1 必然变化。`D_fault − D_clean` 的距离差只能作描述，不能当作严格扣除原生噪声的因果估计；优先报告同 Batch8 配对及各自重复分布。

msprobe 沿用 `task=statistics, level=mix, summary_mode=statistics`，保留原生 dump/stack/construct、配置和完成回执，不默认保存全批 .pt。逻辑输入 hash、实际 schedule、必要精确数值/差异索引与命中日志单独归档。摘要相同不证明逐元素相同；null 只表示不可观测。

## 下一阶段的准入条件

1. 在同一冻结环境用正常运行轨迹确认实际 API、native kernel 与目标请求/TP 映射。
2. 对 Batch1/8 捕获目标 query 的完整语义输入等值证据与真实 tiling/profile，并确认未跨核切 Reduce。
3. 固定与当前实现匹配的 arithmetic policy、合法正常/异常 profile 和可构建 host 入口。
4. 根据 A 或 B 的真实边界分别验收，不将传播代理升级为原生故障定位。

当前可以进入上述前置校准；不能按现稿直接宣称注入能力、FIA 首差异或模型 E2E 已验证。

## 公开参考与使用边界

- [CATLASS FAI tiling 示例](https://gitcode.com/cann/catlass/blob/master/examples/40_flash_attention_infer_tla/fai_tiling.cpp)：可审查 Q/S 分块及任务分配，但该例 QS block 固定为 128；不证明安装的 FIA 会因 Batch 改变 S1。
- [Ascend op-plugin FIA-v2 适配器](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/FusedInferAttentionScoreV2KernelNpuOpApi.cpp)：说明 public API 与底层实现存在版本和平台分发，不能代替本机安装版本。
- [TorchAir FIA-v2 文档](https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/ops/npu_fused_infer_attention_score_v2.md)：其中 TorchAir 定制入口仅支持图模式，actual-length 还涉及 tiling 下沉；不能与 eager 接口混用。
- [Ascend CANN ops-adv 目录说明](https://gitee.com/ascend/cann-ops-adv/blob/9c9579a4ea3f69a8c24ed1cc58f4a517119136c8/README.md)：FA、FIA、PFA、IFA 分目录存在，路径名称本身不证明运行时选路。

公开 master 与安装构建不做等同。上述链接只支持其对应的窄范围说明；历史根因是否一致仍以实际版本、输入与运行轨迹为准。
