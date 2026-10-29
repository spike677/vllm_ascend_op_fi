# 故障 28：原生 FA Tiling 切分策略异常

**已在 124 物理卡 5（910B4，TP1）通过受控机理、真实模型传播及高精度统计独立定位验收。** 正式 B/F 使用同一个私有 FA 构建，各执行两次完整 prefill forward。

**定位通过依赖显式的任务内 msprobe FP32 统计修正。** 未修改的 msprobe 26.1.0 默认 FP16 四统计不能分辨本次微小 Attention 输出差异，首个可观测差异是下游 `Tensor.pow.7.forward` 的输入。该失败结果保留，不能把 pow 当作故障源。新 B/F 仍为 `statistics/mix/summary_mode=statistics`，使用公开配置 `precision=high` 和经审查的私有 collector；不是未经修改的官方采集器。

[正式 B/F 证据说明](../../artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029.md) · [最终验收](../../artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029/summaries/execution_acceptance.json) · [采集器修正](analysis/collector_high/README.md)

## 实际路径与版本

模型为本地 Qwen3-0.6B 原 BF16 checkpoint，所有组均由原生 Transformers 加载为 FP16。固定 2048-token 普通文本，Batch8 同行复制目标请求；目标为 batch0、第二层（零起始 layer1），分析其 prefill 末尾 query 2047 和 next-token logits。两个 step 是两次独立、无 KV cache 的完整 forward，不是 decode token 步数。

真实调用链为：

```text
HF Qwen3Attention.forward → 原生 sdpa_attention_forward
→ torch.nn.functional.scaled_dot_product_attention
→ torch-npu C++ npu_fusion_attention_v3
→ aclnnFlashAttentionScoreGetWorkspaceSize
→ FlashAttentionScore 原生 typed host tiling
→ 原生 half/float FA kernel → 原返回 Tensor → O projection / 后续模型 / logits
```

安装环境为 torch 2.10.0、torch-npu 2.10.0.post4、Transformers 5.5.4、CANN 9.1.0（V100R001C11SPC001B243），冻结镜像 `8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9`。匹配源码为官方 ops-transformer v9.1.0 commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`。13 个 FA kernel 文件与安装源码逐字节一致；169 个相关 FIA/PFA/IFA/common 文件也匹配。

已先运行 vLLM Batch1/8：真实 FIA 命中固定 M128 的 SplitFuse，key 为 `5000000000000200103`，不适合只改 host 的 M 字段。本实现使用同一 checkpoint 的真实 HF SDPA→FA 路径；**没有宣称实现了该 vLLM FIA 分支的切分故障**。原始 [FIA 校准](../../analysis/fault28_tiling_check_20261029/native_fia_calibration/README.md) 与 [FA 校准](../../analysis/fault28_tiling_check_20261029/native_hf_fa_calibration/README.md)分别保留。

## 注入实现与关键代码

[apply_policy.py](source_matching/fa_policy/apply_policy.py) 在固定 release 的干净副本上修改 `FlashAttentionScoreTilingS1s2Bn2gs1::CalcS1S2BasicBlock` 和 `CalcNRatio`。ON 时选择已有 kernel 支持的 M64、nRatio4，S2 基块保持128；OFF 完整保留原策略。后续 matmul、softmax、尾块、core mapping、workspace 和序列化均交由原生 typed host 重新生成。kernel 算术源码未改，没有覆盖不透明 buffer 偏移，也没有重算或替换 Attention 输出。

[fault28_fa_policy.h](source_matching/fa_policy/fault28_fa_policy.h) 的 `Eligible` 同时绑定实际 PID/native TID、layer1、完整 prefill、forward token、B8/S2048/Q16/KV8/D128、FP16/BNSD、causal BOOL mask、scale、inner precision0、dropout0和全部可选输入缺省。按匹配版本的 IR 索引及原生 rank0 缺省规则判定；每次满足条件都触发，不使用调用次数选一次。

[hf_runtime_probe.py](hf_runtime_probe.py) 的 `PrivateVendorScope` 仅在真实 layer1 SDPA 同步调用前设置作用域，并在 finally 恢复。`Observer.observe` 原样调用原 API 一次、返回同一个 Tensor，记录真实输入前后完整哈希和目标映射。[vendor_exec.py](vendor_exec.py) 在新进程加载前核对全部670个私有文件；不替换全局 SDK。[controller.py](controller.py) 记录两卡资源，只使用卡5，只清理本轮身份可证明的自建容器。

重建与安装见 [BUILD.md](source_matching/fa_policy/BUILD.md)。最终 a2 policy manifest 为 `e760be8f4b4dd42b4f44898a8d91c68585c9f96f42c47eecb038eb8d37859294`，vendor manifest 为 `63a5171d2031aa1a8ae5b37c8c9b488614779af7759841e224079a7e746d7b07`。B/F、保护和恢复均使用同一最终构建。

## 实机结果与模拟依据

| 条件 | M | S2 group | 实际结果 |
|---|---:|---:|---|
| stock / 私有 OFF，B1 | 128 | 1024 | 逐层输出、logits、1464字节 raw tiling 一致 |
| stock / 私有 OFF，B8 | 128 | 1024 | 两 forward 均一致，资格匹配但不改变策略 |
| ON，B1 | 128 | 1024 | 两 forward 均不触发，输出不变 |
| ON，B8 | 64 | 512 | 两次指定调用均触发；首个运行时值差在 layer1 Attention 输出 |
| 同进程 ON→OFF，B8 | 64→128 | 512→1024 | 后一次所有层、raw tiling、logits 恢复 OFF |

本例两种策略的实际 key 都是 `22619644976`，不能只凭 key 判断是否注入。公开 context raw、独立 POD 解码和真实 launch 的原始 tiling 前缀逐字节对应。OFF 与 stock 的完整1464字节 raw 一致；源码 ABI 结论限定于这些实测形状。

B/F layer1 **全 Batch 的 Q/K/V 逻辑字节哈希相同**，mask/scale/GQA/精度配置相同。固定源码与实际 profile 的独立计划推导检查了262144个 queries：每个 query 独占一个真实任务、合法 causal keys 恰好覆盖一次，没有跨核 SplitKV。尾 query 的 online softmax/output 累积从2个1024-key分组变为4个512-key分组，实际浮点归约路径改变。因此注入的是 Batch 相关执行策略差异，不是向结果添加任意误差。计划推导不是硬件指令 trace。

两 forward 的实际数值首差均由有序输入/输出哈希独立推导为 layer1 输出。除这两次选中调用外，其余54次 FA 的 raw tiling 与 OFF 相同。差异传播到后续模型及 logits；Top-1 仍为3516。目标末尾 query 的2048个输出元素中34个变化，最大 B/F 绝对差为 `0.0001220703125`。同一已表示 FP16 输入的 [FP64参考](../../analysis/fault28_tiling_check_20261029/final_acceptance_high_a1/reference_forward0.json)保留了两边误差；没有据此宣称业务精度下降。

## msprobe 独立推断与统计精度

未修改官方默认统计中，SDPA 输出两组均为 Max0.640625、Min−0.67431640625、Mean0.00022029876708984375、Norm339.25。统计相同不代表元素相同；独立首差落在后续 pow 输入，只能证明更早已存在差异。

官方公开 `precision=high` 配置在当前26.1.0安装源码中跳过了非标量浮点 Mean/Norm。任务私有副本只修正 `PytorchDataProcessor.get_stat_info`：对所有 FP16/BF16 节点统一使用 detached FP32 临时量计算统计，保留原 Tensor、dtype/shape和API边界；不读取层、B/F或注入开关。原安装文件、原默认 B/F 和原分析结果均不改。

原 processor SHA `e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011`，修正后 `3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141`，stage manifest `743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af`。513个文件逐项核验；23项CPU检查、FP64参考、真实DataWriter和NPU数值透明性通过。修正只影响测量精度。

正式高精度 B/F 两步均有1621个对齐节点、28个SDPA和对应模块。只读取 config/dump/stack/construct 的 [独立分析器](../../analysis/fault28_tiling_check_20261029/analyze_msprobe_native.py)均首先观测到 `Functional.scaled_dot_product_attention.1.forward` **输出 Mean** 不同：

- B：`0.0002203190088039264`
- F：`0.00022031953267287463`

该 API 的输入统计匹配，精确输入等值另由运行时全字节哈希核验。首差前仍有50个不可观测统计字段；null保留为不可观测。msprobe直接记录的是SDPA边界，并未直接显示CANN内部指令或唯一证明Tiling根因；真实FA关联由原生launch/profile和匹配源码另行证实。

## 交付和复核

[正式包](../../artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029/README.txt)有208个文件、18,337,954字节，主数据仅B/F：每组step0/1的原生dump/stack/construct、配置、回执及运行证据；无完整 `.pt`、activation `.bin`、缓存、巨日志或已编译库。JSON中的raw_hex和其他校准包的bin是tiling元数据，不是激活张量。私有激活仅在124用于精确比较/FP64参考，不提交GitHub。

```sh
python artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029/tools/package_bf_evidence.py verify \
  --package artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029
```

包 manifest SHA 为 `561221711e723c891f2feb2729720f9a761e5452e42402ebf1c27f697f229eb6`。最终验收按源报告SHA绑定；打包器只证明完整性，不自行宣称科学验收通过。

结论范围：原生FA受控机理、真实HF模型数值传播、任务私有高精度统计的独立API边界定位。未复现历史38Bv2缺陷，未验证TP2、decode或业务质量；不能把本结果推广为当前vLLM FIA自动出现同样故障。
