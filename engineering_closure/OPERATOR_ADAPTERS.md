# E3 算子适配器

[`operator_adapters.py`](operator_adapters.py) 按显式机制选择已有执行入口。发现阶段复用 `RuntimeCandidateInventory`，由真实观察器提供完整模块和 Worker 生命周期、健康重复调用及不适用原因。`identify` 使用 `CandidateSet.select`，同形状的不同模块仍有不同身份。`inspect_metadata` 返回当前观察到的执行边界、算子、Tensor 签名及 Provider 证据字段。

`bind_target` 接收已经按原入口配置的 Provider 或控制器，核对机制和控制器类型。指针、工作区和无损验证由对应 Provider 在实际调用时绑定。`observe` 区分只读采集、无损基线运行和受控调度阶段。`inject` 执行普通同步回放，或给现有模型/Worker 安装目标并等待调用。`verify_effect` 返回原 Provider 的证据或结束一笔真实 Worker 请求。`restore` 和 `cleanup` 分别进入各自的 OFF 与资源释放入口。

| 算子边界 | 执行路径和仍需核对的范围 |
| --- | --- |
| MatMul / QMM | `20.buffer_substitute` 和 `26.residual_stomp` 复用 `ModelFaultBackend`、实际 Native Runtime 和 Provider 的槽位/consumer 校验。`39.matmul_range` 复用 `demoncat_fault39.backend.NativeBackend`。`39.output_proxy` 从真实投影返回到合法 BF16 Storage 和 value consumer，由 `Fault39OutputProxyProvider` 驱动。 |
| RMSNorm / Conv3d / AsStrided | `18.api_half`、`17.spatial_padding`、`15.alias_detach` 复用模型会话。RMSNorm 同时支持 `NaturalTensorGroupProvider`，由原 group 维护全量分母和 OFF。Conv3d 走真实参数通道，AsStrided 需证明 view、copy_ 和原始 Storage consumer。 |
| FIA / FA / Paged IFA | `26.fia_destination` 进入实际写入侧；`28.fia.*` 和历史 `33.paged_decode` 使用各自的 Native Runtime；FA 使用 `NativeStrategyBridge` 的 S1S2 证书路径。8.5.1 TND page16 的独立 FAI 地址、Host 与 Kernel 校验见 [E4](TILING_KERNEL_ADAPTERS.md)。 |
| RoPE / IndexSelect | `22.wrong_row` 支持已有 `WrongRowSession` 和 `NaturalTensorGroupProvider`。执行前在实际合法索引范围检查行映射，返回实际 consumer 与恢复记录。待补齐公开 Worker 的自然运行覆盖。 |
| MoE Unpermute | `8.restore_offset` 复用 `RestoreOffsetSession`，在实际 `npu_moe_token_unpermute` Dispatcher 中核对 indices、权重及配对映射。多 Rank 生产入口仍需相应请求和模型证据。 |
| DynamicQuant | `quant.batch_shared_scale` 复用量化会话，要求已部署的量化 Runtime 及实际调度、输入和 Q/scale 绑定。 |
| KV Cache | `41.tail_column` 的自然 Worker 采用 `Postwrite41WorkerProvider`，现有调度器调用 `expect` 及 `finish_request`。受控模型可绑定 `Fault41TailColumnProvider`，其 writer 为实际 `index_copy_`。 |
| ReduceScatter | `25.reduce_scatter` 复用 `ReduceScatterSession`。观测取得同一活跃 process group 的原始完成记录，故障模式需要同一输入的真实测量对。后端实际 Kernel 数值路径仍需单独观测。 |

## 输出和缓存的租约

[公共 ACL 实现](../fault_injection/native_adapters/postwrite/postwrite.cpp) 为每段实际 Storage 分配独立 handle，以同一进程中的 owner、generation、线程和设备作为作用域。地址区间重叠、陈旧句柄和外部作用域会拒绝。每段限 1 MiB，共 256 段和 160 MiB device span 总额，主机长期保留两份快照。旧 ABI1 动态库由 Python 按单租约能力使用。

[`ProjectionCoverage`](../fault_injection/deployments/postwrite_coverage.py) 以每个独立健康输入回放隔离完成 C2。每次命中模式仍须同一进程的活动 C2 回执，显式提高 `max_live_leases` 后，独立投影 Storage 能共同保留至一次真实模型前向结束，完整观察窗口内拒绝提前恢复。保存每个输出的值变化、消费者和重放后的恢复结果。随后检查相同输入的完整 OFF 输出和每个目标的原始字节。已观察到的分母保持完整。共享 Storage、非有限原始值、超预算及下游不可达仍按实际拒绝记录。上游注入传播到后续原始输出时，后续的非有限 span 会在 Native 有限值检查中拒绝，实际目标模型 C3 需结合此次传播情况判定。

恢复检查在同步后核对当前整段字节。发生部分失败时，仅释放正常恢复的租约，并保留失败的储存引用和控制器。`retry_recovery` 在同一 group generation 重试，并核对实际健康 OFF。第一次失败的 coverage 迹线保持原样。标准 [自然模型探针](../experiments/c2c3_postwrite_20261117/natural39_coverage_probe.py) 的 `--try-c3 --max-live-leases N` 显式启用；缺省 `N=1`。源码改动后需新构建产物与目标平台实机验证。

[`LiveKVSlotLease`](../fault_injection/deployments/postwrite_kv.py) 在实际 writer 返回后，核对 live value Storage、Block Table、Slot、offset 和对应 TND FIA consumer。实际调用完成前拒绝提前恢复。恢复失败时保留对象；已改动的外部字节可由原持有者在确认安全后按原 provider 重试。自然 28 目标验证仍待有资源的 Worker 窗口执行。

CPU、native mock 和数值引用分别验证接口、租约与底层编译、真实 CPU 张量效果。当前工程回执的运行范围以测试环境为准，Ascend NPU、公开 Worker 及跨版本资格需另附实机证据。
