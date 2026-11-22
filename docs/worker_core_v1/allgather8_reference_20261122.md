# V2-Lite AllGather 输出行错位参照验收（2026-11-22）

结论：参照实验通过；**不授予 MC2、EP、V4 或黄区资格**。现有 restore-index 变体及冻结 r5 保持不变。

组合：124 A2 / CANN 9.1.0 / vLLM 0.26.0、vllm-ascend 0.26.0rc1 / 本地 DeepSeek-V2-Lite-W8A8 / TP2、EP关闭。完整安装与权重身份见 [独立复核摘要](allgather8_reference_20261122/review.json)。源码清单 SHA：`086a437dd22c3efdfaf46c5cf757e0480a0b31866182b25aeefff7a323dc27cf`。

边界：真实 dispatch → expert MLP → 原生 `npu_moe_token_unpermute`，元数据与原算子调用保持不变；在实际返回值上按当前请求的行做循环错位，真实 `aten::add` 消费完成后恢复同一存储。单行 decode 无可错位的请求内行组，明确记为不适用，不补造调用。与 MC2 输出行代理复用 `MC2OutputConsumer` 和 `RegionLeases`，请求映射仍由 AllGather 实际路由与调度元数据证明。

同一组 Worker、两个不同 owner 作用域，normal/fault/off/health 各生成4 token；每作用域先完成连续两次有限且精确一致的预热。

| 每作用域、每rank | 结果 |
| --- | --- |
| 真实调用 observed | 104，覆盖26个MoE模块×4个forward |
| eligible / attempted / applied / changed / consumed / restored | 各26 |
| failed | 0 |
| 单行decode不适用 | 78 |
| 动态producer/combine join | 104，真实native combine104 |
| 消费者 | `aten::add`，设备同stream完成后再恢复 |
| 效果 | hidden、logits均改变 |
| 恢复 | 注入区字节精确恢复；OFF/health hidden/logits精确；每rank租约、hook、在途数0 |
| 性能 | 两作用域分别1.13196×、1.07648×，只记录 |

独立复核要求显式 `--reference-variant allgather_output_row_proxy`；默认公共验收器拒绝把本回执作为公共能力资格。复核同时检查当前请求归属、104次producer/combine调用、26次有效改写、78次不适用、真实消费者、双作用域Worker身份和恢复。原回执186,089,851字节，SHA256 `50ff55110b5db2690b5d8cb123138faf9dc90f5e405d0e84d22b4ac9b0c6c400`，留在蓝区和本地私有临时目录，只提交摘要。

窗口：2026-11-22 13:32:54.400833–13:38:01.586789 UTC，307.186秒。卡2/5有界Docker stop清理器，未改Docker入口/配置/驱动；本窗口无外部容器停止记录。结束已停止并join清理器，恢复派发能力。详见 [窗口回执](allgather8_reference_20261122/window.json)。
