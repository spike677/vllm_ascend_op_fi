# Fault41 设计边界

受控 corruption 的最小安全顺序为：

```text
real writer
→ 同步并保存逻辑 slot tail 原始内容
→ 在真实 storage 上执行有限值 mutation（例如 zero_tail）
→ paired consumer 确认使用同一 storage 并完成
→ 恢复 snapshot、同步、readback 校验
```

selection 至少关联 request generation、rank、cache kind、storage identity/allocation generation、layer/cache ordinal 与 slot/region。仅按 occurrence 可能误中 warmup、另一个 storage 或另一个 layer。`changed_elements > 0`、consumer-before-restore、`restored_hash == original_hash` 和 worker/HTTP health 是机制 gate。

Fault41 的语义是受控 state/cache corruption proxy；它不声称底层 writer 自然发生 OOB 或硬件自然破坏 KV cache。
