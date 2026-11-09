# FIA 图模式受控 M/causal 尾块策略：构建前合同

此文件在构建前冻结，描述候选实现及必要验收，不能当作构建成功或 NPU 通过回执。实际阶段结果见上级 README、独立 build/stage completion 和实际 NPU 验收报告。固定源为官方 ops-transformer `c77c0829f1f0f0b8f882db1d9b051a758b605472`（v9.1.0），镜像为既有 CANN9.1 `sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9`。

原生图发现中，B1/B8 都选 M512/S2Base512，且默认20/40核预算存在SplitKV。不能把该输出差异宣称为故障28。控核入口按[官方 TorchAir 限核文档](https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/limit_cores.md)同时指定Cube/Vector，使用 `16|32`，并以实际FIA key、分核数组和 fdParams 证明无跨核归约。仅设置 `16` 会形成16:40非法比值，已经保留失败结果。

本候选对真实FIA增加独立key的合法causal尾块计算变体，而非复刻已修复历史版本的源码。Batch策略只选择M；S2Base固定512。kernel按每个M块内**最大的query**，裁剪该块最后一次核内S2计算中全部为未来mask的列：

```text
q_end = (gs1_start % actual_s1) + actual_m_rows  # exclusive
kv_begin = s2_index * s2_base
actual_inner = min(native_inner, align32(q_end - kv_begin))
```

因此同一个query在M512、M256、M128的组合中，实际softmax/Reduce的列数可以不同；保留的有效key始终为0..query，每个一次。原native K/V步进仍按S2Base512进行，不拼接、不重排、不遗漏任何合法key，不修改dtype、scale、QKV、mask内容或算子返回值。这个候选需要host与kernel一起适配，不能只改最终raw中的M。

- host在 `FiaTilingNonQuant::CalcMBaseSize` 原生选择完成后切换合法M，再由原流程生成分核、workspace、softmax flags和完整tiling。若选择结果存在SplitKV则拒绝本受控scope。
- `off` 完全保留原生选择。`m128/m256` 为无裁剪的M对照。
- `causal_base512/causal_m128/causal_m256` 使用同一裁剪规则。M512与512-token S2边界对齐，应与stock逐bit等值，必须以实测桥接证明。
- `batch_causal_m128` 从实际B读取：B1=M512，B8=M128；同一裁剪规则保持启用。环境变量不直接改S2或payload。图编译后策略固化于图，每次回放须核对实际kernel执行；关闭控制以新进程/重新编译验证，不把运行时改env当作旧图已恢复。
- 独立key `109000000000000000` 与 `109000000000400000` 仅选择FP16/BNSD/非PA/非FD的裁剪变体。原生key保持原代码。FIA公开serializer结构保持不变，新增key注册相同类型；完整raw保留并用同源SDK测得的layout独立解码。

作用域要求实际进程token/PID、FIA节点名、B1/B8、N16/Nkv8/G2、S1=S2=2048、D128、FP16、inner_precise0、BNSD、相同实际长度、causal sparse2、无quant/rope/prefix/padding/pse/sink，以及16/32核心预算。每个符合条件的图节点均选择策略；不匹配的调用透传。host记录本次实际编译线程，不能把Python线程名当作GE tiling线程。

CPU构建使用独立source、dependencies、Python deps、evidence和vendor generation，无网络、无NPU设备、4CPU/16GiB，1800秒上限；安装只写本任务私有目录。沿用此前实际成功的离线CMake/pip配置，SDK及旧任务只读。CPU检查证明索引/覆盖和序列化自洽，不替代NPU合法性或数值验收。包包含JIT源码；CPU不预编译设备kernel，必须在NPU验证新key实际编译、launch及结果。

msprobe首差分析只读原始GE统计/图映射，不读命中日志来指定首差。跨batch目标完整字节比较与同B8的B/F统计比较分开。stock/OFF兼容、M-only控制、固定M跨batch控制、自动Batch策略、独立重复及关闭恢复全部保留；无差异和不可观测项不升级为成功。若数值现象或机理任一门槛不满足，则该候选不通过。
