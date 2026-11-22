# 产品入口：Worker自动发现的真实HTTP验收

蓝区 A2（Ascend910B4）/ CANN9.1.0 / torch2.10.0+cpu / torch_npu2.10.0.post4 / vLLM0.26.0+empty / vllm-ascend0.26.0rc1 / 社区Qwen3.8-27B-W8A8 / TP2。不是黄区现役权重或A3验收。

用户直接调用真实vLLM `/v1/completions` 或 `/v1/chat/completions`，控制走HTTP `/collective_rpc`；没有宿主before_request/after_request。Worker从实际scheduler_output发现内部ID、按实际请求判定范围。每个机制同组Worker两个独立owner作用域，每作用域三条不同用户prompt和一条生成5token的范围外请求。故障请求prefill+三步decode；关闭后由相同推理API发金丝雀，比较完整hidden/logits。用户请求不与金丝雀做相等性比较。

## 结果

|能力|每请求、每rank eligible|两作用域|真实消费者/区域恢复/金丝雀精确恢复|
|---|---:|---|---|
|18|644|通过|通过|
|39 output_proxy|1216|通过|通过；新增Inf与上游已Inf分别统计|
|22|64|通过|通过|
|28|64|通过|通过|
|17|1|通过|真实视觉编码器→embedding消费者；调用参数恢复、金丝雀精确恢复|

前四项由同一个HTTP服务的同组Worker依次验收，每项独立两作用域。来源清单adbe4894c4ab1da529519bdd97c22a86f922980c7257e52e457eb4ea517e6f7f。逐请求、每rank八项计数、消费者与非有限值见[product_http_20261122](product_http_20261122)的独立review JSON。每项所有eligible applied=consumed=restored，failed=0；范围外请求不注入并返回原因；租约、hook、owner清零。审查器仅判蓝区声明组合，不生成部署登记。

17独立视觉服务来源清单c65101b049cc81d023da6b878d4e5a734c2d77a76869385cdd299ad3e675b6c4；每作用域三条不同prompt使用两张不同64×64用户图片，金丝雀为第三张固定图。prefill真实视觉调用一次，decode不补调用。

## 真实发现的缺陷

首次18 HTTP尝试失败：Norm dispatch mode跨出了inference guard，后续无请求调度执行torch操作触发Python dispatch上下文错误。修复在真实compute_logits结束、最终norm消费者完成且没有pending/parents后，于同一个inference上下文退出mode。失败attempt保留；终止失败服务不作为恢复通过。修复后的独立18复测及四项合测均通过。

首次17 HTTP尝试失败于覆盖校验：误把四次语言前向作为四次视觉前向分母。改为实际encoder_calls（一）核对视觉forward_calls，仍要求全部事件数量一致；两个针对性CPU测试防止放松文本decode覆盖。复测通过，旧失败不抹除。

## 范围与未完成项

仅串行max_num_seqs=1、单chunk、fresh cache、上下文≤1024、生成1～4token、关闭异步调度/prefix cache/MTP/KV connector。18/39/22/28纯文本；视觉范围由17独立实测。没有观察到外部request ID字段时标UNKNOWN，不假装HTTP completion ID是相同ID。

测试profile与测试许可只用于验收。长期无人关闭、无限请求积累、取消/故障流量下的服务级稳定性及黄区主项目闭环未验收。CPU覆盖早停调度、启用前请求、并发拒绝、owner改变；不把CPU结果包装为NPU这些负向场景通过。耗时只记录，不优化。

详细原回执和日志保留私有实验目录，仅发布审查摘要和指纹。冻结r4不变；黄区缺字段和未完成的33/V4适配不因本页蓝区结果默认放行。
