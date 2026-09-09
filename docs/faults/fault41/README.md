# Fault41：State/Cache Tail Corruption

Fault41 关注真实 writer/consumer 之间的 state/KV cache 逻辑 slot 尾部发生有限值 corruption 后的传播与恢复。它与“整个 storage 物理末尾写 32 字节”不同：target 必须是已解析的逻辑 slot、有效 token/head/column region 和当前 live storage binding。

- [设计边界](architecture.md)
- [状态与已知观测限制](validation_status.md)
- [历史归档定位](../../archive/fault41/README.md)
