# Fault41 状态与已知观测限制

当前文档状态为：

```text
FUNCTIONALLY_COMPLETE
WITH_KNOWN_MSPROBE_OBSERVABILITY_LIMITATION
```

含义是：已有受控 writer/consumer、slot mapping、snapshot/restore 与 effect 研究路线，但 msprobe 打开时可能改变 writer selector、phase classifier 或 storage identity 的可观察形态。因此 msprobe 是 observer/运行条件，不是第二个 fault backend；不能由“msprobe 下 selector 未命中”推翻无 msprobe 的已验证 post-write 机制。

本 canonical repo 当前没有在本轮导入新的 Fault41 runtime、远端 artifact 或 raw evidence。文档整理不把 Fault41 升级为新的 committed capability，也不修改其 effect claim。后续应先进行 msprobe OFF/ON 的 call timeline/selector transparency 审计，再决定是否增加 pre-read timing binding；不得借此文档状态直接执行新 injection。
