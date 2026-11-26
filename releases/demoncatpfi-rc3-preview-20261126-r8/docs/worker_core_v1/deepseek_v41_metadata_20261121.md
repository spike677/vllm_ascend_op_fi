# DeepSeek V4.1 元数据接收与结构适配边界

2026-11-21：按负责人决定，模型元数据由蓝区直接从 ModelScope 获取，不再要求黄区传文件。config 一致作为负责人允许的工作前提，不作为黄区原件 SHA 已独立核验的声明。没有下载权重、执行仓库代码或安排 V4.1 NPU 运行。

## 下载与身份

来源为 [Eco-Tech W4A8](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8) 和 [Eco-Tech W8A8](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8)。每个下载 URL 固定到文件 revision，校验 API 所列字节数与 SHA256；`metadata-manifest.json` 记录来源、版本、字节数、SHA。下载器只允许元数据后缀，拒绝 safetensors/bin 权重与路径穿越，单文件上限64MiB。

|版本|接收目录（D:/DemonCATPFI/intake/models/Eco-Tech/ 下）|文件数|总字节|
|---|---|---:|---:|
|W4A8|DeepSeek-V4.1-Flash-1017-w4a8/20261121-875289d09b19|8|35,763,895|
|W8A8|DeepSeek-V4.1-Flash-1017-w8a8/20261121-fe432e3770db|8|28,896,619|

每版包含 config、configuration、quant_model_description、权重索引、tokenizer、tokenizer_config、best_practice YAML 和 README。仓库未列出 generation_config；视觉配置嵌在 config.vision_config 中，没有单独文件。原件在 Git 外；仓库仅保存派生身份清单 `deepseek_v41_metadata_20261121.json` 和脚本。

两版 config 逐字节相同，SHA256 为 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`，与黄区提供的 `43d3b2bc` 前缀一致。前缀匹配不等于独立比对黄区完整 SHA；社区制品不替代现役权重。

## 量化对照与例外

- 视觉262键成立：259个 `vision.*` + `image_start/image_end/image_newline`，均 FLOAT。
- W4A8 主干 `layers.*.ffn.experts.*` 184,320项均 W4A8_DYNAMIC。不能把包含 MTP/共享专家的所有 `experts` 键都归到同一类型。
- 主干 attention `wq_b` 两版均 W8A8_DYNAMIC。W4A8 的 `mtp.0/1/2.attn.wq_b.weight` 三项为 FLOAT；W8A8 对应项为 W8A8_DYNAMIC。这是分支差异，记录保留。
- norm 248项均 FLOAT；embed 七项中五项 FLOAT，两个 Engram `embed.quant_type` 为 INT8_SYM。黄区“embed 为 FLOAT”只能按普通 token embedding 解读，不能外推 Engram。
- W4A8 quant SHA：`1bc65590c213a620738477020127e99a8df5394cfec284198005df33d526178c`。W8A8 实值及所有文件指纹见派生 JSON。

不一致/例外不得自动修改成期望值。部署时仍以黄区现役的实际观测与制品为准。

## CPU 结构夹具与 site 判定

`fault_injection/sites/deepseek_v41_metadata.py` 读取真实 config 与量化键。主干40层、64头、head_dim512、384路由专家、top-k6；从原层0/2/20截取结构，保留原层号及压缩率0/2/1，不重排共享索引关系。两版各产生67个实际路径 leaf，采用67个标量 dummy 权重，避免分配真实专家/Engram。夹具拒绝 forward；它验证结构发现，不模拟 vLLM 执行或证明算子消费者。

|能力|待验证真实边界|当前判定|
|---|---|---|
|28|SparseFlashMla + LightningIndexer 实际稀疏 key、causal/长度及消费者|需适配；V2-Lite 稠密 MLA 通过不能继承|
|8|MoeDistributeDispatchV2/CombineV2 的动态 token/request 映射与专家位置|需 M4 EP 运行证据|
|Batch|独立 DynamicQuant → 原 QMM|需调用证明；融合进 GroupedMatmul 的量化不适用本变体|
|18|RmsNorm/RmsNormCast 的实际 dtype 和消费者|结构有候选，待运行边界|
|22|InplacePartialRotaryMul 的 position 维度和 rotary 区间|需适配；不套用 Qwen mRoPE|
|39|MatMulV2/QuantMatmulWeightNz 输出 BF16 ND 与消费者|量化键不能证明输出 dtype，待实测|
|41|MLA cache 活跃对象、布局、写读同 stream 与恢复点|需 Native 接线；元数据不证明活跃地址|

上述名称来自任务指定的适配目标，不是假称从 config 观测到了运行算子。全部公共准入保持关闭。M4 的专家实际分布、token/request 归属及稀疏 key 必须来自运行时；不能由384专家简单平均分到8 rank，也不能按基线调用清单强制 fault 调用出现。当前没有0.32镜像，未执行 DeepSeek V4.1 NPU；现有8-rank CPU协调测试不等同于 M4 完整验收。

## 黄区待到达的只读结果

DeepSeek 同进程确定性与 Qwen3.8 首批 B8 新结果尚未作为本轮输入收到。收到后另写接收记录，再判断预热上限和 Batch 门禁；本轮不更改连续两次全有限且完全一致的预热规则。
