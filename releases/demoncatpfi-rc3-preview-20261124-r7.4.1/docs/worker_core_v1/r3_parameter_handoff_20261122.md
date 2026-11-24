# r3 参数对齐：转交补充页

2026-11-22已执行 `git ls-remote --tags origin demoncatpfi-rc3-preview-20261122-r3`，远端tag对象为 `2d575cf38191bea83eb99e38055fba1c03d0984d`；其发行提交仍为交付单中的 `0a10cb2e`。冻结包不改。本页由负责人随[交付单](../../releases/RC3-PREVIEW-R3-HANDOFF-20261122.md)转交。

黄区A组使用卡8～15的独立实例，不接对外流量。必须按r3脚本重新启动：chunked prefill开启、`max_num_batched_tokens=1024`、`max_model_len=1024`、seed42、image=0、video=0；Batch `max_num_seqs=8`，其他项1；eager、串行、mp executor、异步调度/prefix cache/MTP/KV connector关闭。不能沿用chunked prefill关闭、max_model_len4096的旧模板。r3入口直接构造这些参数，不是对已有实例执行任意注入。

|偏差|实际保护与报告|
|---|---|
|非纯文本多模态额度|`V026_TEXT_ONLY_MODALITY_LIMITS_REQUIRED`；发生视觉调用还有独立拒绝|
|chunk拆分、padding或请求ID映射改变|`V026_SINGLE_CHUNK_NOT_CORROBORATED`、`V026_PADDING_UNSUPPORTED`及持续请求映射拒绝|
|异步、非eager、MTP、共享cache/connector|启动准入及 `CONTINUOUS_FRESH_CACHE_INFERENCE_REQUIRED` 等拒绝，回传完整首个错误|
|实际源码不同|`NEWSTACK_*`/source fingerprint拒绝；不得修改指纹绕过|
|seed或max_model_len单独改成别值|未必有独立的等值拒绝码；它们仍改变冻结验收参数，不可沿用此次资格。宿主必须核对完整启动参数，不能把“没报错”当已对齐|

不同能力可以分配到不同专用实例并行；同一能力的两个作用域必须复用同一组Worker/model，不能每作用域新起实例。TP1与TP2分别记录。TP2按负责人给定拓扑选同一芯片相邻逻辑设备，例如8/9，但仍需核实实际映射；不能仅凭编号推断芯片。每个rank记录torch_npu设备名原值、逻辑设备及group身份，npu-smi泛称只作参考。

验收完成按交付单打印安装检查、summary全文、原回执文件名/大小/SHA和错误码。通过后仅登记 `YELLOW_DEDICATED_INSTANCE`（A3、实际0.26组合、现役Qwen3.8、实际TP1或TP2）。Codex不直接向黄区发送任务。
