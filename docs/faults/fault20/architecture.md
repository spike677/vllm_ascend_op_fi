# Fault20 架构与语义边界

核心语义链为：

```text
真实 aclnnQuantMatmulWeightNz
→ 已证明的语义输入 slot / immutable metadata
→ temporary device buffer substitution
→ 原算子真实执行
→ 受控数值退化或无损验证
→ 计数、rank、release、health/recovery evidence
```

已验证的实现以 executor-cache `OpExecCache::Run`、typed GWS metadata、shared control 和 fail-closed guard 为中心。它不创建新的 `aclTensor` 回填 opaque cache，不长期缓存 activation address，也不原地改写调用方的 address vector。

INT4/INT2-equivalent 模式保持 INT8 storage 与布局，通过控制内容的有效精度模拟数值退化；它不是 packed INT4/INT2 kernel，也不宣称硬件自然发生了相同故障。request-scoped 与 arm-lifetime bounded hit policy 都须有明确 request/control 证据。

完整原始架构说明：[`fault20_final_architecture.md`](../../../fault20_fast_validation/reports/fault20_final_architecture.md)。
