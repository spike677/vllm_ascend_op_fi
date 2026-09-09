# Fault39 验证状态

canonical reports：

- [`fault39_float_matmul_finite_nonfinite_validation.md`](../../../fault20_fast_validation/reports/fault39_float_matmul_finite_nonfinite_validation.md)
- [`fault39_float_matmul_cancellation_validation.md`](../../../fault20_fast_validation/reports/fault39_float_matmul_cancellation_validation.md)
- [`fault39_float_matmul_output_nonfinite_proxy_validation.md`](../../../fault20_fast_validation/reports/fault39_float_matmul_output_nonfinite_proxy_validation.md)
- [`fault39_matmul_fault_reproduction_summary.md`](../../../fault20_fast_validation/reports/fault39_matmul_fault_reproduction_summary.md)

这些 source 支持受控 BF16 temporary input、有限输入到真实 non-finite、排列相关的数值范围现象和 post-Run output proxy 的不同证据层级。本轮没有重跑数值实验、没有修改故障强度，也没有把 proxy 结果升级为历史 kernel 根因证明。
