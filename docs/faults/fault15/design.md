# Fault15 设计边界

Fault15 的语义是“量化参数统计粒度不正确导致跨 token/sequence 的不应有耦合”。它应与单纯的 INT8 数值截断、scale 固定倍率放大和下游 residual corruption 区分。

潜在实现需要证明：真实 scale 的逻辑归属、当前请求/sequence 边界、替换或观察的 binding、实际数值改变，以及 downstream consumer 在正确生命周期内消费。未证明这些条件时，不应把任意输出差异归为 Fault15。
