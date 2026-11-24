# V2-Lite 28 MLA：从实验binding到公共入口

public-a16组合：A2 Ascend910B4 / CANN9.1.0 / vLLM0.26.0 / vllm-ascend0.26.0rc1 / torch2.10+cpu / torch_npu2.10.post4 / DeepSeek-V2-Lite-W8A8 / TP2。模型config SHA `229913ea2a346ccdb571bb3cb23414ca3a0ee1a9455fe88a15a5788bc837cb75`；本次源码清单SHA `6d1610b620d447ab5fa1f42c4eba3107c94601634cba3af21172c69ee6898266`。

公共MultiRankTaskClient → Runtime → ContinuousProvider → 源绑定的DeepseekV2ForCausalLM/AscendMLAImpl实际prefill与decode调用 → 原真实o_proj消费者。无Qwen多模态门禁继承；独立核对token-only分支、实际模型forward源码、runner execute/preprocess/sample和真实并行组。未适配的模型/机制继续拒绝。

27层MLA，每请求prefill+3decode，两个作用域同组Worker。每rank每作用域108次observed/eligible/attempted/applied/changed/consumed/restored，failed=0。prefill使用实际展开QKV，decode使用吸收latent KV与真实V-up投影；分母块0重放、分子不重放，属于大误差效果变体，不宣称历史kernel复现。该结果不覆盖V4.1稀疏索引或EP。

新返回的实际重算张量持有公共强引用区间租约，o_proj读取值核验后等待同stream设备完成，再恢复该对象行。还保留未改目标外区域精确检查。失败不提前弹出pending租约。所有输出有限；故障hidden/logits变化，fresh OFF/health完整指纹精确恢复。两个作用域结束许可、hooks、租约和在途资源清零，实际无许可创建与耗尽后advance拒绝。

精简独立回执：`experiments/worker_core_v11_20261118/continuous_public_mla28_review_20261121.json`，含原件和比较器SHA。性能2.0268502×/1.8232932×。

窗口为2026-11-21 13:04:49.707410至13:09:32.658442 UTC，共282.951秒（含加载）。有界卡2/5清理器已停止、派发恢复；日志仅ISOLATION_START/END，无外部容器stop。旧实验证据保留。本次新增独立BLUE_COMBINATION，不修改r2，不授予87/A3资格。
