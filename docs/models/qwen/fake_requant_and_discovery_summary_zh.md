# Qwen FakeRequant 效果与模块发现汇总

本页是面向汇报的简表；具体数值见 [FakeRequant 效果报告](fake_requant_effect_v1.md)，可观察面见 [模块注入面发现](module_injection_surface_v1_zh.md)。

```text
冻结 FakeRequant R1
  → 66/66 机制有效，rank0 命中，rank1 预期拒绝
  → PPL 17.8507 → 19.0203（1.0655×）且 3 cycle 重复
  → 短输出 3/4、长输出 5/5 发生轨迹分叉
  → 未见 repetition/spam/early-stop/runaway/crash
  → GSM8K-10 未重复证明任务准确率损伤

Qwen Module Surface
  → qkv_proj / o_proj / gate_up_proj / down_proj
  → 64 层 × 2 rank 均有真实 BF16 input observe-only evidence
```

这些结论仅适用于本报告所列的私有 Qwen 双卡环境；它们既不代表 DSV4 的 live 结果，也不表示发现入口已经执行故障注入。
