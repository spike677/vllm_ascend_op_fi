# Fault26：受控 residual stomp proxy

Fault26 用于研究“真实算子执行窗口与 live residual/state 受害对象之间的受控错误写入效果”，并验证下游残差链、模型输出和恢复诊断。它必须明确称为 proxy，不能被写成健康 kernel 自然发生 OOB 的证明。

- [设计与物理边界](architecture.md)
- [验证状态与限制](validation_status.md)
- [历史探查目录](../../archive/fault26/README.md)
