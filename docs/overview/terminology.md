# 术语

| 术语 | 含义 |
| --- | --- |
| Fault | 对历史或目标异常的语义分类；不自动等于某个具体 kernel 根因。 |
| Primitive | 面向产品/诊断的故障语义，例如输入精度退化、输出 non-finite proxy、state corruption。 |
| Execution Plane | 落地实现所在层，例如 vLLM backend、torch_npu、op_api、native executor cache。 |
| Capability | 某环境中可发现、可验证的实现能力及其条件、限制和 evidence pointer。 |
| Candidate | 由观察事实解析出的潜在目标；尚未通过 lossless gate。 |
| Runtime Binding | 一次具体运行中由逻辑目标解析出的 symbol、tensor、slot、storage、stream 等物理事实。 |
| Lossless Validation | 临时绑定/替换但保持输出精确等价的验证门禁。 |
| Mechanism Evidence | 证明 target resolve、hit、changed elements、rank policy 等实际发生的证据。 |
| Effect Profile | 汇总 task、probability、output、phenotype、reliability 与 recovery 的评估结果。 |
| Proxy | 受控模拟某种故障表征的能力；不宣称健康运行自然产生该物理缺陷。 |
| Evidence | 原始 trace、响应、运行日志、hash、manifest 或可复核结果；不等同于 Markdown 解释文档。 |

## 两个必须区分的关系

```text
semantic position / primitive
    ≠
execution plane / library symbol
```

同一个 `op_api` 导出符号可能因 shape、dtype、format、tiling 或硬件走不同底层实现；反过来，同一个语义 fault 也可能由不同执行平面实现。模型层名称（例如 attention 或 MLP）仅是调用上下文，不能替代根因分类。

```text
trajectory divergence
    ≠
pathological collapse
```

输出从 baseline 分叉是 output effect；重复、刷屏、早停、失控或崩溃是更具体的 phenotype，必须分项统计。
