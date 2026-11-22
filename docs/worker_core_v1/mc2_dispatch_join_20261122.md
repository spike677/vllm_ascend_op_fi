# M4：MC2 实际调用关联的 CPU 准备

本轮新增 `sites/mc2_v41.py`，尚不授予故障8或DeepSeek资格。依据为vllm-ascend v0.32.0rc1源码提交 `24487f4f48ee99cf45d86d1c504e645549b8c1ef`，`token_dispatcher.py` SHA `43e57927ed54a2d2a1193ccf8bbcd56d1751417e460954bc0b8b967d2a23b228`。没有蓝区V4.1 NPU运行。

## 已实现

- source-bound `TokenDispatcherWithMC2.token_dispatch` / `token_combine` 包装，实际 `npu_moe_distribute_dispatch_v2` / `combine_v2` 参数及输出的观察。调用点不增加collective，不替换算子返回值；没有当前作用域的调用透传。
- 从实际 topk_ids、active mask 和可信调度行映射记录请求/位置/route；每次独立观察路由，不要求fault与baseline的路由清单一致。inactive padding不生成请求，缺失active行映射拒绝。
- 检查实际dispatch七元返回的 assist、recv counts、expand_scales 是否原对象进入combine metadata；检查combine真实expert_ids、权重、通信组/rank/world size。opaque assist保持原样，不把它当topk数组修改。
- 维持调用级关联，未combine或外部替换wrapper时拒绝释放；仍由现有Runtime控制请求生命周期和公共边界汇总。
- CPU测试包含动态路由变化、padding不补调用、旧assist/变更索引/权重拒绝；使用原版本两个方法的源码摘录，验证上游关键字调用 `token_dispatch_input`、`hidden_states`、`combine_metadata` 的真实签名。摘录仅替换Native执行和结果dataclass，不代表MC2硬件通信通过。

## 未完成且保持关闭

1. 当前observer需要可信 `context_provider` 提供实际分片后的请求行映射；尚未自动接入0.32 runner → MC2 prepare/pad/split的完整链。不能从shape或rank猜行归属。
2. 当前记录是源dispatch请求与原生combine的关联，不是接收rank上每个expert GMM行的完整反解；`destination_expert_row_decode_proven=false`明确保留。已有 `ep_context` 只验证显式给定的placement，不能补造该证据。
3. 冗余专家/EPLB、MXFP预量化改变dispatch x、FusedMC2路线未适配，明确拒绝。W4A8中哪些层实际走该MC2类需黄区调用证据核对。
4. 故障8如何扰动MC2元数据并保持通信/恢复安全尚未批准为可执行site；本轮不改assist或分布式算子的参与者。
5. 残留兼容方案A/B/C待负责人选择；最终A3验收步骤与组合登记尚未完成。蓝区V2-Lite AllGather通过不代表本路线通过。

这些边界与 `dsv41_sparse_cpu_20261122.md` 的稀疏MLA限制并列记录，下一步是补齐真实runner分片行映射和接收端expert归属，而不是把CPU原生API替身测试包装成A3实测。
