# V2-Lite 公共持续39：真实投影消费者

public-a25，A2 Ascend910B4 / CANN9.1.0 / vLLM0.26.0 + vllm-ascend0.26.0rc1 / torch2.10+cpu + torch_npu2.10.post4 / DeepSeek-V2-Lite-W8A8 / TP2、EP关闭、eager。源码清单 `59151d8cd67c5b79a307cce24aff87f6b3f9f2cc3e72557a5080ec69638d61b6`。

同组Worker两个公共作用域通过。每rank每作用域136个曾实际调用的投影site，observed463、eligible332；attempted/applied/changed/consumed/restored均332，failed0。另131个调用输出不满足BF16 ND，明确列为不适用；不能把463全算注入成功。52个结构候选没有模块forward（shared expert投影）；直接融合专家kernel不在本模块适配范围，完整名单保留在独立报告。

实际选点允许多维、strided的ND输出；以真实storage_offset和stride选最后leading row的首元素，不通过reshape复制后冒充原存储。每次写Inf，遇到已非有限输入仍统计：upstream_nonfinite324，already_positive_inf0；本轮NaN到Inf等新字节变化合计332，未把“上游已非有限”算成初始错误来源。

实际消费者包括aten::bmm/copy_、原生FIA、SwiGLU、真实all-gather/unpad和residual分块。MLA自定义算子会暂时屏蔽外层TorchDispatchMode，因此在源码指纹匹配的forward及preprocess边界内创建共享租约列表的观察mode；嵌套观察去重，同一设备消费只恢复一次。原实现只调用一次，没有人工加一个消费者。方法退出不能替代设备完成事件。

正常、OFF、health全有限且完整hidden/logits精确恢复；fault允许该机制预期的非有限输出并传播到真实消费者。两个作用域结束时许可、租约、hook和在途资源清零，负向创建和额外advance拒绝通过。耗时1.681254×/1.758787×；两边共有观察开销，非生产开销估计。

独立审查：`experiments/worker_core_v11_20261118/continuous_public_dsv39_review_20261121.json`，含原件SHA。a23在无注入观察阶段被二维连续门禁拒绝；a24补多维选点后被MLA内部消费者不可见拒绝；只有a25用于验收。a25窗口2026-11-21T13:55:31.867158Z至14:01:15.672674Z，343.806秒；三窗口均解除卡2/5有界隔离，无外部容器停止事件。

仅登记BLUE_COMBINATION，不扩展V4.1/EP/A3；r2不变。失败窗口清理不是同进程恢复证据，a25的公共回执在退出容器前完成两作用域恢复审查。
