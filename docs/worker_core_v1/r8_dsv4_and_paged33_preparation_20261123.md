# r8：V4 D 批预适配与33待观测策略

V4使用已核对的0.32源码候选24487f4f，精确Linux字节SHA见dsv4_0731_source_review_20261123.json。蓝区没有V4现役运行证据，不启动DeepSeek NPU、不新增公开资格。

新增 dsv4_pre_adapter.prepare_site 是关闭的预适配检查器：只输出计划、不装hook、不改张量。18 npu_rms_norm_cast 必须同时考虑BF16 hidden与FP32路由输入；model.py的rms_norm_cast在forward内将两个返回值分别送MLP，而不是只处理第一个输出。22 indexer的inplace_partial_rotary_mul使用真实partial_slice=[head_dim-rope_head_dim,head_dim]，不套用Qwen mRoPE。39op核对BF16 ND和真实消费者；39mr另需逻辑K、量化权重布局、scale和bias的实际对应。28的PA_ND/PA_BBND分支、压缩cache解码、稀疏合法key集合、causal请求映射和scale都必须由实测说明。

检查器已用小型CPU元数据夹具覆盖18/22/39op/39mr/28的支持与拒绝条件；这还不是可执行模型adapter。D1只观测结果和源码指纹到达后，才把计划连接到真实张量和消费者；当前缺压缩布局/稀疏key运行关联，不能补默认值。

33新增 prepare_paged33_observed_policy.py，将已观测的rank/长度矩阵、raw/POD、serializer/host/kernel SHA、实际key/核数/控制分块与≥99%控制重算证据合成关闭的候选。缺rank、重算不足、serializer不匹配拒绝。实验子类可接受受观测约束的分块，但公共factory没有接入，A3仍关闭。

不同tiling key可能改变累加与重标定顺序；观测到新key不能仅改分组参数就宣称覆盖新分支。需要对应源码算术审查和新的控制检查点。无A3结果时不决定r7.2、不发布资格。
