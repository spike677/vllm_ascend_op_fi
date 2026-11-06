# 故障 33 · a3 桥接拒绝记录

**正式验收拒绝了 a3 的私有 OFF 桥接。** 124 的 910B4 上实际执行了 4 次分页 IFA，分别为 2 次 stock 和 2 次私有 OFF。两次 OFF 的 `native_scope=false`、`eligible=false`，本次运行在进入故障组 F 前结束。原验收状态为 `INVALID_OR_INCOMPLETE_EVIDENCE`。[原始拒绝回执](bridge_a1_acceptance.json) · [独立复核报告](a3_bridge_rejected_review.json)

| 实际观测 | 结果 |
|---|---|
| 执行条件 | B8、KV 长度 16,384、物理页长 128、FP16。4 次调用的 NNOP 确定性标记均为 1。 |
| 可见执行计划 | 4 次的 tiling key 均为 `10000000000200000`，均使用 20 个 blocks、schedule mode 1，两份 SDK BMM 记录也一致。 |
| 数值结果 | 重复比较及 stock/OFF 对位比较中，16,384 个 attention 元素和 2,048 个合成投影元素的完整 FP16 字节均相同。 |
| 私有 OFF 的公共 host 记录 | 两次均取得相同的 3,000 字节序列化数据。72 个投影字段均与独立 ABI 解码一致，`SInner=8192`、循环数为 2。 |
| 拒绝条件 | 两次均记录 `fixed_inputs=true`、`actual_lengths_all_equal=true`、`legal_request=true`，同时 `native_scope=false`、`eligible=false`、`applied=false`。 |

源码排查指向保护条件中对 `inputKvLayout_` 的判断。在该普通分页分支，`ProcessPageAttentionFlag` 设置 `inputLayout_`，`inputKvLayout_` 可能保留默认值 `BSH_BSND`，从而与 a3 预设的 `BNSD` 不符。这是根据源码作出的解释。a3 的原始运行记录只给出聚合的 `native_scope=false`，未分别打印这些内部变量。公共 host raw 的证据范围为最终序列化 tiling 数据，设备内核实参字节未直接采集。[原始 OFF 事件](host_events.jsonl) · [实际 NNOP 日志摘录](native_launch_excerpts.json)

## 文件

- [原始拒绝回执](bridge_a1_acceptance.json)、[独立复核报告](a3_bridge_rejected_review.json)及[两条原始 host 事件](host_events.jsonl)均保留来源文件的原始字节。
- [a3 私有 vendor 清单](vendor_manifest.json)绑定同代的 [ABI 布局](ifa_abi_schema.json)、[编译回执](ifa_abi_compile.json)和 [CPU 往返自检](ifa_abi_selftest.json)。这里归档的是构建元数据，私有共享库保留在验证服务器。
- [原生日志摘录](native_launch_excerpts.json)由复核时同一版本的解析器从两份完整日志派生。每一条保留的 NNOP/BMM 行均有原文件行号，文件中还记录整份日志的 SHA-256。私有输出仅通过复核报告中的完整哈希和逐元素统计公开。
- [SHA-256 清单](SHA256SUMS)覆盖本目录其余全部文件。可在本目录运行 `sha256sum -c SHA256SUMS`。

正式 B/F 首差、故障命中及模型传播均不属于本反例的结论。

