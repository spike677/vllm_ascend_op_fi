# 0.26 运行时中和版本绑定的检查清单

按 D74 提供源码咨询。位置以冻结 r8 `e38c4113b421ac979699071a60e80d3babd05e07` 为准，不以蓝区开发线替换它。包内 `reference/r8/` 是选定文件的原始字节快照，`runtime-source-locations.json` 给出 SHA、函数/类和行号；不是可安装运行时。下表 `W/` 表示 `src/demoncat/runtime/vllm_ascend/worker_extension/`，`F/` 表示 `fault_injection/`。

这是已识别的迁移检查面，不是对 0.30 源码已完成审查，也不是所有未来机制的完整覆盖承诺。黄区以实际安装代码扩充清单，并由独立会话审查。

|检查位置|现有目的/绑定假设|迁到 0.30 必须重新审查与验证|
|---|---|---|
|W/v026_text_observer.py：PROFILES、_verify_function；v026_source_review.json、v026_installed_source_review.json、preferred_a2_source_review.json|用实际函数来源和文件 SHA 对照已审列表，拒绝未审代码|逐个审实际 execute_model、sample_tokens、_preprocess、_model_forward、model.forward、embed_input_ids 及父类委托；记录签名、返回值和源码变化。不能只增加 SHA；插件包装和继承实现要单独核查|
|W/v026_text_observer.py：SourceBoundTextObserver.validate/install/close|限定 runner/model 类、NPU、同步 eager、无 speculative、无既有实例包装；只拆自己持有的包装|0.30 的 runner v1/v2、代理、graph、异步、draft 模式是否改变边界；未审模式保持拒绝。保持 kwargs、原异常、原返回身份；卸载后恢复原方法，不能删除其他扩展|
|W/v026_text_observer.py：TextPrefillTrace；W/v026_generation_observer.py：FirstDecodeTrace|校验 preprocess 六元返回、同一 execute_model_state、embed/forward/logits/sample 顺序；B8 诊断仅首个 decode|重审返回结构、hidden/sample_hidden/logits 的真实生成/消费顺序、sample_tokens 的异步返回与采样次数。4 token 不等于必然 4 次 compute_logits；按请求/step 解释额外调用，不机械丢弃|
|W/v026_inspection.py：parallel_assessment|仅已审 TP1/TP2、PP1/DP1、PCP/DCP=1；配置与实际组互证|TP8/EP 不能改一个上限就放行；取得 TP/EP/MC2 有序成员、global/local rank、通信语义，重审 PP/DP/SP/PCP/DCP 的映射。字段缺失不得默认 1|
|W/v026_inspection.py：inspect_single_chunk；W/serving_context.py|InputBatch req_ids、req_id_to_index、computed tokens、scheduler 新/缓存请求与 CPU mirror 证明真实行归属|0.30 InputBatch 重排/压缩、cached req 结构、padding、混合 prefill/decode、请求结束/取消、chunk 切分。预算充足不是单 chunk 的运行时证明|
|W/product_observer.py：AutomaticRequests、make_automatic_observer|发现开启期间请求；execute→sample→finished/max_tokens；串行边界与自动包装所有权|动态 batching/异步顺序、请求在 enable 前已开始、EOS/中断、采样与请求 ID 对齐；不能把旧 session 的输出归给新 owner。若更改支持范围，形成独立版本和验收|
|W/product_request.py：ProductRequest.install/collect；W/continuous_binding.py：ContinuousRequest|真实请求范围、持续触发、站点命中和恢复|检查调用层次和实际算子消费者；不能只沿用旧模块名。每 rank observed/eligible/attempted/applied/changed/consumed/restored/failed 的意义保持；changed<applied 不自动失败|
|W/product_batch.py：AutomaticBatches、classify_batch|当前真实 decode 批次至少两请求；单请求不注入；串行金丝雀|0.30 decode/mixed 批识别、行集合与请求 ID、max_num_seqs、quant→QMM 的实际消费、并发请求的结束；维持 scale 精确、Q ±1 的原标准。旧冷启动异常不得混作注入恢复失败或偷偷降低比较标准|
|F/product_contract.py：request_range；W/product_config.py：load/configure；W/product_request.py|mode=all，真实调度事实决定范围，旧范围/context/token/视觉/新 cache 限制|Qwen 与 V4 分离部署 profile。2500–4092 是 D 参考方案，不是全机制已授权范围；图像、speculative、prefix cache、LoRA、prompt embeddings 等超范围请求说明原因且不注入|
|W/product_binding.py：product_binding_rpc；W/continuous_binding.py：check/install/remove|模型、generation、owner、机制绑定与实际调用准入|多 owner 顺序、第二 owner 拒绝不清理第一 owner、部分 rank 成功时仅回滚本 owner 已取得的资源；拒绝保持类型/拒绝码、CLI exit4、引擎存活|
|W/product_host.py：ProductClient._binding/_all/enable/disable/detach；W/product_errors.py|全 rank 返回、控制边界、drain、金丝雀、恢复与失败传播|0.30 executor 的 rank 顺序/调用序号/同步行为；无 rank 回包、部分失败和超时不能吞。冻结 r8 不包含后来的 C2 最小修复，不在本包暗中替换；根据复现证据处理|
|W/product_registration.py：implementation_identity/observe_identity/validate_identity/validate_record/select_registration|将逐 rank 硬件、软件、模型 SHA、模式、范围、变体、实现 SHA 绑定到审查证据|重新生成 0.30 身份和登记，不能复制 0.26 expected_ranks。旧 observe_identity 对 EP 开启给未知值，rank 范围也按 TP；需真实 EP/TP 身份设计，不能补造 false 或空组来绕过|
|W/deployment_identity.py；F/environment.py|模型 config/量化描述/所有 index SHA，实际版本与加载库映射|模型/库以实际运行进程为准。镜像文件和 toolkit 版本不能替代 loaded-library 事实；maps 多段不是多个 rank。原件缺字段不能回填默认身份|
|W/product_native.py：validate_blue_native_library/validate_blue_native_worker；W/product_config.py|Native 路径/SHA、硬件、TP、CANN9.1 配套与加载来源|0.30/CANN9.2.1 是新组合：原库不因路径可读即可使用。核对构建源码、compiler、recipe、产物/收据/sidecar、ACL/torch_npu ABI、真实 stream；错 SHA/路径需加载前拒绝|
|W/serving.py；W/serving_context.py；W/continuous_binding.py|张量对象/存储区间、字节快照、设备消费与恢复|重审 0.30 布局、视图、生命周期、算子异步 stream。Python return 不等于读取完成；不能把旧快照写回已复用对象或覆盖后续合法写入|
|W/product_request.py：install_visual_site；W/v026_text_observer.py：visual/mrope 检查|纯文本排除视觉执行；视觉路径单独证明图像与 patch_embed/Conv3d 关联|0.30 多模态数据结构、positions/mRoPE、processor/image grid 与内置确定性金丝雀；既有 17 资格只覆盖其已验收模型/范围|
|W/external_provider.py、external_control.py、external_sampling.py|适配器身份、受控 site、采样边界、v1 非持久区域|0.30 实际 dispatch/采样边界、传入对象、consumer 和请求归属；provider 名称存在不是 site 可用证明。D13 的 router 分数不能是内部 router 入口的 hidden|
|W/external_persistent_runtime.py、external_persistent_regions.py、persistent_product_host.py、persistent_kv_release.py|v2 权重/KV 的独占、派生缓存、请求页面生命周期和安全恢复|0.30 权重预打包/NZ 副本、offload、KV 页重用、prefix cache、stream、跨 rank 部分失败。关闭时读者排空，不能写回已复用 KV；恢复语义不得迁成简单 copy-back|

## 每个被包装函数的迁移审查记录

至少记录：实际文件/字节数/SHA、类/MRO/实例包装来源、旧/新签名与 kwargs、输入输出对象及 dtype/layout、调用前后状态、请求/row/token 关联、执行线程/stream/事件、异常与 owner 恢复、CPU fixture、实机证明、独立审查结论。只有源码列表 SHA 更新、没有语义记录的条目不放行。

## 黄区执行与登记顺序

R0 在 r8 基础上另开 0.30 迁移分支，保留完整差异。R1 用 18.api_half、39mr 做 TP2 两 owner、真实效果/消费者、三项恢复和负例。R2 的 D13/公共 8 按现有门槛独立判定；R3 逐项迁移和重新验收。正式登记绑定新 implementation/source fingerprints、软件栈、硬件、模型制品、TP/EP、模式、范围和变体；0.26 PASS 只作为对照证据，不能当成 0.30 PASS。

负例至少覆盖未审函数、错误实现身份、未登记组合、跨 TP/EP、缺字段、错误 Native 路径/SHA、包装已占有、第二 owner、部分 rank 安装/恢复失败、超范围请求。保留拒绝码、全 rank 审计、引擎存活和后续正对照。黄区独立审查会话负责审核，不再等待蓝区常规复审。

本清单不新增/放宽任何 runtime profile；本包无 wheel、Native 库、许可和登记记录。0.32 按同样流程重新审查，不能因 0.30 通过而直接放行。
