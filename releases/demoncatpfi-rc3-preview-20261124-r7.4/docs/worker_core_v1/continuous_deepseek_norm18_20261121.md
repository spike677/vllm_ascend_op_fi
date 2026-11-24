# V2-Lite 公共持续18：实际RMSNorm模块范围

public-a21 在 A2 Ascend910B4 / CANN9.1.0 / vLLM0.26.0、vllm-ascend0.26.0rc1 / torch2.10+cpu、torch_npu2.10.post4 / DeepSeek-V2-Lite-W8A8 / TP2、eager、EP关闭上通过同组Worker两个作用域。

每rank每作用域55个真实调用的AscendRMSNorm模块×4次forward，observed/eligible/attempted/applied/changed/consumed/restored均220，failed0。每次将输出中一个实际元素乘0.5，跟踪真实读消费者和同stream完成事件后逐字节恢复；不是调用返回即判消费。所有阶段全有限，fault hidden/logits变化，OFF/health完整输出精确恢复，最终许可、hook、租约和在途资源清零。

**范围限制：**另有27个 `kv_a_layernorm` 结构候选没有执行module forward。安装源码显示其权重由 `npu_kv_rmsnorm_rope_cache` 融合消费；本次模块适配器不覆盖融合算子内部RMSNorm。原始审查明确列出全部27项，不把“没有模块调用”说成整个模型归一化语义已全覆盖。Qwen的融合Q/K适配器不能据此扩展到DeepSeek融合KV。该限制随BLUE_COMBINATION保留，不授权V4.1/A3。

两作用域耗时1.723290×/1.651745×。源码清单SHA `c821325e9e0189b797eb60b16d3900916fd55ccfdb5e69a4ead897f5a8a64e4c`；独立审查 `experiments/worker_core_v11_20261118/continuous_public_dsv18_review_20261121.json`，附原回执SHA。

窗口2026-11-21T13:34:00.842766Z至13:38:29.515616Z，268.673秒（含模型加载、预热、退出）。卡2/5有界隔离已解除，无外部容器停止事件。只加入开发线候选，不修改r2。
