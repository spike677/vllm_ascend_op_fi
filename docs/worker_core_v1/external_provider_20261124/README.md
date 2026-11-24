# 外部 provider 的 A2 产品模式示例验收

范围：Ascend910B4（A2）、vLLM 0.26 / vllm-ascend 0.26.0rc1 / CANN 9.1、Qwen3.8 社区 W8A8、TP2。
示例机制 `external.gain_three_quarters`；真实 norm 输出区域缩放 3/4。它不在内置故障 catalog 中，仍能通过可信部署注册、动态 capabilities、公共 ProductClient 与原 Runtime 运行。

| owner | rank | observed / eligible / attempted / applied | changed | consumed / restored | failed |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 1932 / 1932 / 1932 / 1932 | 1930 | 1932 / 1932 | 0 |
| 0 | 1 | 1932 / 1932 / 1932 / 1932 | 1930 | 1932 / 1932 | 0 |
| 1 | 0 | 1932 / 1932 / 1932 / 1932 | 1930 | 1932 / 1932 | 0 |
| 1 | 1 | 1932 / 1932 / 1932 / 1932 | 1930 | 1932 / 1932 | 0 |

每 owner 三条普通 HTTP 用户请求，均包含 prefill 和后续 decode（4 个生成 token）。第 4 条请求为 5 token 超范围负例，未注入。
真实消费者包括 `aten::index`、`npu::npu_dynamic_quant`、`vllm::unified_attention_with_output`。
所有区域恢复哈希一致；关闭后故障租约/hook/inflight 为零；新金丝雀 hidden/logits 精确恢复。changed<applied 表示部分写入没有新增数值变化，不是调用失败。

独立审查使用现有 `tools/review_product_mode.py`，两个 owner 的 Worker 身份不变。首轮和最终复核分别见 `a1_review.json`、`a2_review.json`；
适配器文件 SHA256 为 `27396562629c4db0c7dcb23ae89d5a250b9e112b03ce0cc4bd75cf27050f978a`。
最终复核包含跨 rank 源码一致性和事件请求归属检查。其后补 Host 内部异常 exit 5 分类、未完成清理的公共 disable 重试，以及可信 bootstrap 结构化拒绝的 CPU 检查；没有更改设备或数值路径。这些后补控制边界未重复 NPU 验收。

两个窗口分别为 274.280 秒、276.252 秒，包含模型加载、请求和退出，不是 kernel 用时。两个窗口均只使用卡2/5，有界清理器已停止并恢复派发能力；没有修改 Docker 入口、宿主驱动/CANN 或删除外部数据。
控制器、原始回执、隔离日志的 SHA 在各 review 中。原件留在本地/124，不提交完整大回执。
vLLM 退出时的 Python resource_tracker 提示与故障 Runtime 资源计数分开记录，不声称检查了所有宿主 IPC 对象。

准入结论：**BLUE_COMBINATION 示例通过；外部 provider 仍为临时许可验收模式**。不是黄区十六项能力的验收，也不授予 A3、DeepSeek、EP 或免许可产品资格。
黄区各适配器仍需自己的源码审查、组合身份、自然调用、真实消费者、恢复与负例。接线协议见 [接口说明](../external_provider_v1_20261124.md)。

CPU：完整开发门禁 1238 项，1236 通过、2 项废弃研究入口按既有理由跳过；最终控制边界补充 26 项通过（包含 bootstrap 鉴权拒绝不逃出 RPC、清理重试）。完整门禁和补充检查重叠，不能相加作独立用例总数。
