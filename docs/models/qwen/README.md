# Qwen 文档

当前推荐：[Qwen3-0.6B单卡用户手册](../../inspect/qwen06b_single_npu_user_walkthrough_zh.md)和[源码基线记录](../../releases/source_baseline_v1/README.md)。28层112目标、三题N=1；不是下方旧双卡R1的PPL/任务报告。

以下保留 `Qwen3-32B-w8a8-nopdmix` 双卡历史结果。不同CANN/vLLM/torch_npu、硬件、rank或模型revision必须重新验证；不同故障配置的指标不可互借。

- [FakeRequant 效果报告（中文）](fake_requant_effect_v1.md)
- [FakeRequant 效果报告（English）](fake_requant_effect_v1_en.md)
- [模块注入面发现 V1（中文）](module_injection_surface_v1_zh.md)
- [效果与发现汇总（中文）](fake_requant_and_discovery_summary_zh.md)

完整原始 response、mechanism trace 和远端路径仅作为 evidence pointer 保留在各报告中，不复制到文档目录。
