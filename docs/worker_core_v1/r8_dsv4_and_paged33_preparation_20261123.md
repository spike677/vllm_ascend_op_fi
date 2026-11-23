# r8：V4 D 批预适配与33待观测策略

V4使用已核对的0.32源码候选24487f4f，精确Linux字节SHA见dsv4_0731_source_review_20261123.json。蓝区没有V4现役运行证据，不启动V4 NPU、不新增V4公开资格。V2-Lite 的0.26 A2 TP2预演单独记录，不代表V4或MC2。

新增 dsv4_pre_adapter.prepare_site 是关闭的预适配检查器：只输出计划、不装hook、不改张量。18 npu_rms_norm_cast 必须同时考虑BF16 hidden与FP32路由输入；model.py的rms_norm_cast在forward内将两个返回值分别送MLP，而不是只处理第一个输出。22 indexer的inplace_partial_rotary_mul使用真实partial_slice=[head_dim-rope_head_dim,head_dim]，不套用Qwen mRoPE。39op核对BF16 ND和真实消费者；39mr另需逻辑K、量化权重布局、scale和bias的实际对应。28的PA_ND/PA_BBND分支、压缩cache解码、稀疏合法key集合、causal请求映射和scale都必须由实测说明。

元数据检查器保留。新增 `fault_injection/sites/dsv4_d2_calls.py` 的真实张量调用适配器及CPU夹具，复用RegionLeases；实际安装factory始终返回 `DSV4_D2_GATE_CLOSED_D1_REVIEW_REQUIRED`，公共配置不能打开。

| 能力 | 关闭门禁后的调用适配 | D1待核对 |
|---|---|---|
|18|RmsNorm只改hidden，保留rstd；RmsNormCast分别处理BF16 hidden及FP32路由返回，并分别核验消费者和恢复|实际命中的模块、双返回值消费分支、调度行关联|
|22|原InplacePartialRotaryMul真实调用前改cos/sin行；保持partial尾切片和interleave；恢复临时lookup，不撤销Q的合法原地更新|真实positions→lookup来源、Q形状和消费者|
|39op|BF16 ND输出区域写Inf，真实存储区间进入消费后恢复|MatMulV2/QuantMatmulWeightNz实际调用和模块映射|
|39mr|逻辑K全范围对照后漏算最后1/4；量化输入须有解码后的int8 K×N、逐token/通道scale对应|NZ/W4解码、bias和scale绑定；当前packed输入明确拒绝|
|28|合法稀疏key、causal请求边界、sinks、浮点PA_BBND cache的分母重放；分子不重放|实际分支、cache解码及稀疏key关联；PA_ND packed路径仍拒绝|

`sparse_flash_mla.py` 实际加载 `cann_ops_transformer::sparse_flash_mla`，不是 `_C_ascend::sparse_flash_mla`。适配点位于上游wrapper完成参数改名/布局选择后；不把outer wrapper的kwargs直接当作底层schema。该源码把BF16分支描述为A5能力，不能据此推断A3采用同一路径。来源文件和精确git对象SHA另见 `fault_injection/sites/dsv4_d2_sources.json`。

CPU夹具执行实际torch张量变换、真实函数调用、消费者读取和字节恢复，不只传入“已通过”标志；覆盖双owner、错消费者、错误namespace、未知布局及关闭门禁。仍未安装到V4 Worker，未作V4模型/NPU验收。D1到达后核对源码、schema、实际tensor/请求/消费者关联，再决定候选，不补默认值。

33新增 prepare_paged33_observed_policy.py，将已观测的rank/长度矩阵、raw/POD、serializer/host/kernel SHA、实际key/核数/控制分块与≥99%控制重算证据合成关闭的候选。缺rank、重算不足、serializer不匹配拒绝。实验子类可接受受观测约束的分块，但公共factory没有接入，A3仍关闭。

不同tiling key可能改变累加与重标定顺序；观测到新key不能仅改分组参数就宣称覆盖新分支。需要对应源码算术审查和新的控制检查点。无A3结果时不决定r7.2、不发布资格。
