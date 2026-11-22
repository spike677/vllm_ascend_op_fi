# RC3 产品入口：Worker 自动发现请求

本接口的五项蓝区真实HTTP验收已通过，详见[实测报告](product_http_acceptance_20261122.md)。黄区仍待接线和登记，未登记组合默认关闭。冻结 r4 不变。此前依赖宿主逐请求钩子的蓝区结果保留在历史报告和提交 9ec051ea，不能作为本接口的 HTTP 验收结论。

## 用户契约

用户选择故障与已公开变体，然后开启、查询、关闭。作用范围只支持 `RequestScope(mode="all")`：开启期间每个 rank 每次符合条件的真实调用持续触发。没有请求数量配额，也不接收层名、rank、地址、原始分块数字或模型路径。

宿主走现有 HTTP `/collective_rpc` 控制通道，用户推理直接走 vLLM `/v1/completions` 或 `/v1/chat/completions`。宿主不接收用户请求开始/结束通知，也不向请求添加标记。可信部署侧仍负责安装惰性 binding、许可和来源身份，不能向用户开放任意 Python callable 或任意文件加载。

## 控制顺序

1. `ProductClient.from_deployment(...).attach()` 获取实际 Worker 身份，不创建 Worker。
2. `enable()` 安装部署固定 token 的金丝雀观察器，部署侧通过普通 vLLM API 发出金丝雀，最多6次、连续两次完整 hidden/logits 有限且精确相同后启用。
3. Worker 在 `execute_model` 的真实 scheduler_output 识别内部 request ID，读取实际 prompt、生成上限、单chunk、缓存和视觉映射。符合范围的请求安装现有 site；每步真实消费后按原租约恢复。条件钩子内没有全组通信。
4. `status()` 分页读取每个请求的各 rank 记录：实际注入点、八项计数、变化元素、非有限值和恢复证据。实际 Worker 没有外部 request ID 字段时返回 UNKNOWN，不把 HTTP completion ID 当作 request ID。
5. `disable()` 在 Worker 控制边界移除观察器、核对区域恢复与资源零，再由普通推理 API 发金丝雀、比对完整指纹；`detach()` 仅在恢复确认后释放 binding。关闭时仍未收到真实结束证据的用户请求标为 observation ended / completed=false，不伪造完成。

```python
client = ProductClient.from_deployment(
    http_executor, deployment_json, operator=trusted_operator,
    run_canary=lambda spec: service_inference_api(spec),
    drain=lambda: None,
)
client.attach()
client.enable()  # only all enabled requests
status = client.status()
client.disable()
client.detach()
```

金丝雀 tokens 必须由可信部署使用真实服务 processor 生成并冻结。图像金丝雀包含实际展开后的 tokens；不能用未展开的文本 token 替代。普通用户 prompt 不需要预登记，也不与金丝雀比较。

## 当前范围与负向行为

18、39 output_proxy、22、28：文本、max_num_seqs=1、真实单chunk、fresh cache、无LoRA/embeds/speculative、上下文不超过1024、生成1～4token。17：同范围且真实64×64单图、未复用encoder缓存；decode没有视觉调用时不补造。超出范围正常推理、不注入并返回原因。开启前已开始的请求不从中途接管。调度器违反串行部署契约时失败关闭并保留恢复责任。

状态记录按请求分页导出，每页一条请求的所有rank；宿主应及时持久化。长期无人关闭的海量请求容量尚未验收，不能外推生产吞吐或内存边界。恢复不要求已污染的用户请求变回正常，只要求写入区域恢复、租约资源清零以及新金丝雀精确恢复。

## 部署登记

现役量化描述与全部 safetensors 索引 SHA，以及 max_model_len/max_num_batched_tokens/max_num_seqs 是登记前提。配置 SHA 相同不足以证明相同权重。CPU 测试或持有测试许可不产生验收登记；A2 不代表 A3。

实机入口为 `experiments/worker_core_v11_20261118/product_http_probe.py`，在隔离 localhost vLLM 服务上验收，同组Worker两个作用域、三个不同用户prompt、一个范围外请求；17使用两张不同用户图片。该脚本的专用可信 bootstrap 仅用于隔离验收，不应作为对外不可信 RPC 部署。
