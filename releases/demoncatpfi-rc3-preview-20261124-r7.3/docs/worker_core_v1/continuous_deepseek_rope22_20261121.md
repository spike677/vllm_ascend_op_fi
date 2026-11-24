# V2-Lite 公共持续22：真实MLA rotary输入

public-a22，A2 Ascend910B4 / CANN9.1.0 / vLLM0.26.0、vllm-ascend0.26.0rc1 / torch2.10+cpu、torch_npu2.10.post4 / DeepSeek-V2-Lite-W8A8 / TP2、EP关闭、eager。源码清单 `f0d76b62f3ca31f4ec404e6c6d744c5697c7bdc14115ece1ce421da2985ef9e9`。

同组Worker两个公共作用域通过；27个实际MLA实现，每请求prefill及3次decode。每rank每作用域108次observed/eligible/attempted/applied/changed/consumed/restored，failed0；351个rotary行和位置行改变，按调用对齐的108个Q/K输出指纹变化。效果和恢复不是仅凭最终token判定。

适配器包装源码指纹匹配的 `mla_preprocess_prefill/decode`。使用当前 metadata.input_positions 核对真实cos/sin与全局查表结果相同，然后租约改写当前cos/sin为position+1的合法错误行。观察原 `npu_interleave_rope` 和原 `npu_kv_rmsnorm_rope_cache` 分别消费这两个实际对象，要求Q→K顺序且各恰好一次，再按同stream设备完成恢复两个输入对象。共享cache表、positions、KV布局和第三方kernel不改写；故障请求的KV传播保留，OFF/health由全新请求验证。

所有阶段全有限，fault hidden/logits变化，OFF/health完整输出精确恢复，两个作用域的租约、hook和在途资源清零，负向创建和耗尽后advance拒绝通过。独立报告 `experiments/worker_core_v11_20261118/continuous_public_dsv22_review_20261121.json` 含原回执SHA。

fault/normal为1.067790×/1.061186×。两边都含观察器及Q/K指纹回传，不是相对无观察器生产部署的开销。本次窗口372.368秒：2026-11-21T13:38:58.327916Z至13:45:10.696303Z，卡2/5有界隔离已解除，无外部停止事件。

仅验证稠密V2-Lite单请求路径；不授予V4.1稀疏MLA、InplacePartialRotaryMul、EP或A3资格。r2保持不变，新源码仅进入后续候选。
