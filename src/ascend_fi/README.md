# ascend_fi：工具主体与共享合同

## 职责和边界

Inspect、发现、验证、效果评估和profile工具已实现，不再只是占位scaffold；native整合仍保留各目录阶段边界，不代表所有C++已迁移。

| 包 | 输入 → 输出／下游 |
| --- | --- |
| [inspect](inspect/README.md) | worker/model/实参 → descriptor/绑定计划 → 执行器 |
| [contracts](contracts/README.md) | JSON DTO ↔ 对象；不是生产registry |
| [discovery](discovery/README.md) | 记录证据 → Candidate；不是实时计划 |
| [validation](validation/README.md) | 候选＋无损证据 → ValidationResult |
| [evaluation](evaluation/README.md) | B/F/R raw → EvaluationResult/报告 |
| [profiles](profiles/README.md) | 通过验证的记录 → profile |
| [backends](backends/README.md)、[strategies](strategies/README.md)、[observers](observers/README.md) | 执行面整合索引；原native路径保留 |

执行环境、状态与失败处理见组件文档。[tools](../../tools/README.md)提供入口，[tests](../../tests/README.md)说明验证范围。跨组件字段以[产物合同](../../docs/reference/artifact_io_contracts_zh.md)为准。

隔离DemonCAT产品负责生产 `position + primitive + target + params`、策略与生产控制器；这里的演示owner不是第二套生产registry。见[合并边界](../../docs/overview/demoncat_merge_contract.md)。
