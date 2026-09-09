# Fault39 数值路线与命名边界

Fault39 不应把所有极值输入实验统一叫作 accumulator overflow。当前路线分为：

```text
真实数值路线
  finite BF16 输入 → real aclnnMatmul → 自然 non-finite 或有限输出

排列/抵消路线
  数学上可抵消的正负排列 → layout-dependent non-finite 行为观察

结果级 proxy
  real MatMul 先正常执行 → 向真实 output storage 写入 +Inf/-Inf/qNaN → 下游消费
```

只有实际定位 accumulator 类型、位置与 overflow 时刻才可称为 `ACCUMULATOR_OVERFLOW_PROVEN`。现有事实更适合命名为 `MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED`、`MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED` 或 `MATMUL_OUTPUT_*_PROXY_PROVEN`。
