# V4.1 稀疏 MLA CPU 适配进度

仅源码和 CPU 适配，未启动蓝区 V4.1 NPU、未创建黄区任务、未授予公共资格。真实源码依据 vllm-ascend v0.32.0rc1 提交24487f4f；文件指纹见同目录 dsv41_sparse_source_review_20261122.json。它不替代黄区实际安装字节。

## 已实现

`fault_injection/sites/sparse_mla_v41.py` 从实际 SMLA 参数结构解析 TND 请求边界、PA_BBND 页表、原始窗口和压缩索引。支持已审查的浮点连续 cache、原始 causal/window 模式和1/2压缩比；独立处理压缩 residual 时间坐标。逻辑索引与物理页分开，ori/cmp 使用独立命名空间；负哨兵和不可见索引过滤，重复有效key拒绝，未知布局/量化cache拒绝。

公共 reference attention 引擎增加可选 caller-owned per-head sink：它只计入分母且不重放，与 V4.1 上游 small_op_attention/golden 公式一致。默认无sink的旧路径不变。分母故障只重复选中的真实key块，不能重复sink或拿V2-Lite部分latent维度规则套用V4.1；V4.1实际KV的512维同时作为K和V。

CPU覆盖：完整softmax参考、分母重放参考、独立分页映射、两请求归属、稀疏哨兵、压缩边界、重复key拒绝、无效页/量化cache/缺sink拒绝、大sink数值稳定。真实ModelScope config派生夹具保留原0/2/20层、0/2/1压缩比，attention W8A8_DYNAMIC和expert W4A8_DYNAMIC。源config SHA43d3b2bc…与已接收社区原件一致；不是对黄区完整SHA的新核验。

## 尚未完成，保持关闭

- 当前是实际调用参数的 CPU lowering，不是已安装的运行时 hook。LightningIndexer→后处理→SMLA 调用关联、逐query request映射和同stream消费者仍须接入已审查的真实对象。
- packed_cache_ops/量化KV、非causal replay及未审查mask明确拒绝；不能据CPU参考推断全部现役层适用。
- 融合18、partial rotary22、Native41共享 metadata/context 的运行接线未完成。
- MC2真实源代码把dispatch返回的 assist_info/ep_recv_counts 等送到combine；该辅助布局不是topk数组，不能直接按索引张量猜测改写。M4仍缺实际dispatch/combine关联及动态物理专家放置证据，既有ep_context仅校验显式提供的归属。
- 状态残留兼容方案尚待负责人选择；本轮只交设计，不能以清零后稳定替代正确性证明。

本轮986项扩大CPU回归无失败/错误（2个已有skip），包含最初4项稀疏测试；随后真实配置切片补为5项，单独通过。此处全部不算A3验收。r3冻结包字节不变。
