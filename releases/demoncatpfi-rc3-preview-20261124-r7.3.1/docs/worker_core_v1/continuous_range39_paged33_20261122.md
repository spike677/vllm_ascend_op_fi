# 39 matmul_range / 33 公共入口验收

目标组合：124 A2（Ascend910B4），vLLM 0.26 / vLLM-Ascend 0.26.0rc1 / CANN 9.1.0，社区 Qwen3.8-27B W8A8，TP2。本文在实测完成前不授予组合准入。

## 公共契约

两项均复用 `install_continuous_binding` → `MultiRankTaskClient` → `WorkerFaultRuntime`，同一组 Worker 两个独立 owner。每个作用域 normal / fault / off / health，每个请求生成四个 token；预热最多六次，连续两次全有限、完整 hidden/logits 哈希精确一致后进入正式验收。

每 rank 上每次 eligible 的真实调用都执行，不在条件命中点新增 collective。计数区分 observed / eligible / attempted / applied / changed / consumed / restored / failed。changed 可以小于 applied，但本次请求必须有实际模型效果。非目标区域保留，设备事件完成后恢复保护区，OFF/health 新请求逐次对齐完整 hidden/logits 哈希。未覆盖或不适用的调用明确列出。

## 39 matmul_range

实现：`fault_injection/engines/matmul_range.py`、`sites/matmul39.py`，真实消费者和租约沿用 `sites/projection39.py`。

当前适配范围是无 gather 的 ColumnParallelLinear 及其子类、BF16 ND 二维输出。RowParallelLinear 输出已经经过跨 rank 归约，不可用本地部分和直接替换，明确返回 `ONLY_LOCAL_COLUMN_SHARDS_NO_COLLECTIVE`。其他 dtype、布局、量化规则也不悄悄套用。

W8A8 路径在已核对源码的 quant_method.apply 中记录真实 `npu_quant_matmul` 的 Q、权重、两种 scale 和返回对象。输出对象必须与模块返回对象相同，拒绝多次/chunked QMM、不支持的 bias。非量化参考分支读取真实输入和权重，核对布局及 bias 语义，但本次没有非量化投影实机证据，公共选点明确拒绝 `UNQUANTIZED_REFERENCE_CPU_ONLY_NOT_NPU_QUALIFIED`。

每个符合条件的调用对最后 token 的首个本地输出通道重算：完整 K 范围先与实际输出做 BF16 精度的控制核验；故障值仅累加前 3/4 K，模拟累加范围丢失。使用两条向量的 CPU FP64 重算，不复制完整权重，不事后乘缩放系数。仅此输出单元被写入；真实下游值消费者在同 stream 完成读取后字节精确恢复。该变体是错误累加范围的 torch 参考重算代理，不宣称改写原生 MatMul kernel。

## 33 paged_decode

实现：`sites/paged33.py`，页面读取及真实 o_proj 消费链沿用 `sites/gqa28.py`。

prefill 仅观察；decode 序列长度超过128时适用。输入来自当前 query 与实际 key/value cache、当前请求 block table，检查页号有效、不重复，最后一页按真实序列长度裁剪。逻辑 key 集合完整且每个 key 恰好访问一次，不重放分母、不改变因果语义。

控制分组128，替代分组32，均FP32 online softmax。为避免把 native 与 torch 算法差异误报为分组差异，仅对控制重算舍入后与原生值完全一致的单元采用替代分组结果；其余单元保持原生值。报告控制匹配数、两种分组的变化数、实际替换后的变化数及原始页面映射。没有变化是有效的未达标结果，不能人工扩大误差。OFF/health 始终走原生路径。

这一范围是 paged decode 合法归约分组的效果代理，不等于证明历史 PA kernel 的 blockSize 分支已复现。暂不包含 MLA、稀疏 attention、GDN 或跨请求批次。

## 实测记录

### 39：public-a26 通过

独立报告：[continuous_public_matmul39_review_20261122.json](../../experiments/worker_core_v11_20261118/continuous_public_matmul39_review_20261122.json)。源码清单 SHA256：`ebfaab29723359d6037e74dbe320681846a26f97bea61c2b226ddda834ccd11c`。

两个作用域使用同组 Worker；每rank每作用域304个真实投影、1216次观察，其中176个本地列分片投影、704次符合条件。704次均 applied / consumed / restored，failed=0；rank0 changed704、rank1 changed700。其余512次均为 `ONLY_LOCAL_COLUMN_SHARDS_NO_COLLECTIVE`，不是注入失败或成功命中。

所有704次使用真实 W8A8 QMM 输入，完整K范围控制核验通过。消费者包括真实 `npu_swiglu`、`qwen_gdn_attention_core`、`triton_split_qkv_rmsnorm_mrope` 及有独立存储和精确复制核验的 `aten::contiguous`。两个rank的完整hidden/logits均变化；OFF/health与正常完整哈希精确一致，租约/钩子/在途请求清零。无许可真实创建及耗尽后 advance 拒绝通过。

故障请求4.60998秒/4.16647秒，对应正常请求1.73715秒/1.84062秒，倍数2.65377/2.26362，均低于120秒单请求上限。窗口05:45:33.914804–05:56:32.351360 UTC，658.44秒，包含加载和控制操作，不等于 kernel 时间。有界清理器已结束，未停止外部容器、未改变派发器/Docker配置。

实测后额外收紧未验收的非量化路径准入，不改变上述704次全部W8A8调用的行为。该分支仅登记 BLUE_COMBINATION，不提升到A3或冻结r2。

### 33

a27、a28两次相同输入/策略均未通过，**33不登记、不进入r3**。a28只读诊断：每rank observed64 / eligible48 / applied48 / consumed48 / restored48，16次prefill不适用。rank0两次调用各改变1元素，rank1一次调用改变1元素；两种重算分组各有8个元素不同，只有原生等值控制匹配的2/1个元素被替换。

rank0公共query拒绝码为 `CONTINUOUS_NO_MODEL_PROPAGATION`：最终完整hidden/logits没有变化。rank1未被组协调器继续query，不能推断它的最终效果。不能据此定位差异在哪个后续算子消失。没有修改误差门槛、添加额外扰动或重放分母。

尚未完成OFF/health和两个作用域，不用容器清理代替同进程恢复证明。失败原件保留远端，紧凑报告见 `continuous-public-paged33-failure-20261122.json`。三窗口记录见 `range39-paged33-windows-20261122.json`：合计1143.07秒，全部解除隔离，无外部容器停止记录。45项相关CPU测试通过，不替代33实机验收。

39的独立回执报告保留原始 `SOURCE_REVIEW_STILL_REQUIRED` 字节；源码审查结论单独保存在 `continuous_matmul39_source_review_20261122.json`，通过后才登记该BLUE组合。冻结发行物不改动。
