# Fault39：浮点 MatMul non-finite 与结果代理

Fault39 研究矩阵乘数值范围异常如何形成 Inf/NaN 表征，并服务于诊断工具验证。该工作分开保存“更接近数值根因的真实 MatMul 路线”和“确定性结果级 non-finite proxy”。

- [数值路线与命名边界](architecture.md)
- [验证状态](validation_status.md)
- [历史探查目录](../../archive/fault39/README.md)
