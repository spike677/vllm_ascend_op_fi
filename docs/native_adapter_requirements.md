# 原生故障统一桥接与部署边界

`fault_injection.native.NativeSession`负责自动发现当前模型的Linear、量化method、共享Attention或原生Attention实现对象，并追踪其真实forward作用域。用户不提供层路径或地址。模块hook只建立归属关系，不直接arm CANN控制面，也不由shape[0]猜测请求数。

2026-11-16 的实际 provider 与推理验收见
[部署说明](../fault_injection/deployments/README.md)及
[本轮结果](../experiments/provider_adaptation_20261116/README.md)。以下合同仍是每次调用的
要求，不因有一次模型通过就取消。Fault39 独立包的版本验收不提升本表两个39桥的状态。

## 逐项状态

| 机制 | 统一入口进展 | 原native适配仍必须提供的事实 | 既有验收范围 |
| --- | --- | --- | --- |
| `20.buffer_substitute` | 自动模块发现、原生边界callback与逐调用证明门禁已接入 | ABI握手、formal tensor→runtime slot、真实storage、具体输入变换 | 新provider已验收Qwen3及SmolLM2/Llama；BF16/NZ29限定路径，不能继承旧INT2/INT4结论 |
| `26.residual_stomp` | 量化模块候选自动发现，独立的邻接写入合同 | allocator live相邻分配、QMM writer边界、residual consumer及恢复生命周期 | 新provider真实32B W8A8 TP2推理通过；非邻接不触发；统计首差与不可观测项另列 |
| `26.fia_destination` | Attention对象候选自动发现，独立destination合同 | FIA真实writer、当前forward的live residual、原输出与consumer恢复关系；不要求underflow邻接 | 真实32B W8A8 TP2推理通过，每rank126次完整恢复；4/8请求token变化；不能继承underflow证据 |
| `33.paged_decode` | Attention候选与paged实现对象发现；原生callback合同 | 真实page table、KV cache寿命、请求映射、build/tiling/launch；不能把dense prefill转换当已证实分页 | 新provider已验收Qwen3全部28层、B8 teacher decode；统计首差在后续mean，不能唯一定位IFA |
| `39.matmul_range` | Linear对象自动发现，有限输入机制独立登记 | Matmul ABI/slot/storage及有限输入范围；不能声称已定位accumulator类型 | 旧固定Matmul数值实验 |
| `39.output_proxy` | Linear对象自动发现，输出代理独立登记 | 真实返回写入边界、bounds、dtype、恢复；不冒充输入计算根因 | 旧非有限输出代理 |

原完整控制源码仍在`fault20_fast_validation/source/l4_buffer_substitute_allhit_v1`、`l4_metadata_lifecycle_proxy`、`fault26_readonly_trace_v1`、`fault26_fia_writer_proxy_v1`。本轮加入显式宏控制的桥接入口，未定义宏时保留原实现。原算法、历史采集和证据不覆盖；live地址始终来自真实调用与allocator。

## Provider是部署内部适配器

统一入口可接收部署提供的`runtime`。它不是让模型用户填层名：provider会收到自动发现的模块对象，在真实native调用时回报`NativeCall(owner, call_id, operator, ...)`，并带上与本进程、模块对象、call_id、机制和operator绑定的证明文件。

Provider需实现`install/check/execute/finish/abort`：

- install只安装只读native边界观察，并返回自身的移除函数；安装必须事务化，不得先arm再注册。
- check核对版本特定支持范围；不支持可按显式策略透传，不能计为注入覆盖。
- execute负责原控制面的真实arm/disarm及每种机制的语义，调用原算子闭包恰好一次，不在异常后偷偷重试。
- finish返回本次native观察；abort幂等释放本次控制状态。失败与清理失败同时记录。

缺provider时统一入口会在安装模型hook前拒绝。已交付的部署适配器须匹配其库、ABI、allocator、cache或scheduler条件；其余调用明确拒绝或按配置透传。应分别记录**统一路由、具体部署支持、实机验收范围**，不能标记“全部模型可执行”。

Proof JSON需满足`demoncat.native.proof.v1`，带有非空details、kind、mechanism、operator、call_id、owner_id、pid与status。`NativeProof`给出原文件路径及SHA。SHA与call绑定只是证据完整性门禁，既不是真实性签名，也不替代独立native审计。报告中的provider观察单独保存，`native_fault_hit_proven`与`npu_validation_claimed`不会被本Python桥接设为true。

## 覆盖与测试

按实际活跃模块对象、原生调用序号记录observed/routed/passthrough/failed；未观测候选保持未观测。其他模型及模块forward之外的调用透传。没有native回调不能通过覆盖验收。CPU测试provider只执行真实CPU Linear来验证桥接与拒绝规则，证明文件明确标记fixture；它不模拟或验收原NPU故障。

CPU测试覆盖任意模块路径、多层多次调用、缺provider、缺/变更证明、错误机制、其他模型隔离、显式透传、原操作重复执行防护、观察与cleanup失败以及无命中。本轮另取得新B/F/OFF、实际native回执及独立msprobe比较，详见总结；新增版本、模型或形状仍必须重新验收，不能沿用旧模型PASS。
