# Fault26 设计与物理边界

Fault26 的关键区分是：

```text
生命周期关系：QMM/FIA 执行窗口中 residual 是否 live、何时被 consumer 消费
物理邻接关系：output/workspace/temporary allocation 是否真的与 residual 相邻
```

调用顺序不能证明物理地址相邻。只有对 `[Q_begin, Q_end)`、`[R_begin, R_end)` 及实际 allocation span 的只读审计支持邻接时，才可讨论受控 underflow/overrun 路线。若当前 allocator 不提供可控邻接，wrong-destination proxy 需要独立的 live target、lossless output substitution、consumer-before-restore 与 byte-exact recovery gate。

历史 source 区分 `QMM output-base underflow`、`FIA wrong-destination` 和 residual writer timeline。它们是不同物理代理路线，不能混称为“自然 OOB”。

参考：[`fault26_qmm_residual_physical_adjacency_audit_20260818.md`](../../../fault20_fast_validation/reports/fault26_qmm_residual_physical_adjacency_audit_20260818.md)。
