# Fault15：量化 scale 粒度

Fault15 关注量化 scale 的统计粒度错误，例如 batch 内不同 token/sequence 之间不应共享 scale 却发生共享的情形。本目录是文档入口，不在此次整理中引入或升级 runtime capability。

- [设计/语义边界](design.md)
- [验证状态](validation_status.md)

原始 trace 报告：[`fault15_scale_granularity/fault15_scale_granularity_trace_report.md`](../../../fault15_scale_granularity/fault15_scale_granularity_trace_report.md)。
