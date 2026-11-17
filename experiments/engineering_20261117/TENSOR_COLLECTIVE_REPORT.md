# C 工作流：本轮 Tensor / Collective 实机审计

**15、17、18、8、22 在构造 NPU 模块中完成实际 B/B/L/F/OFF；五项独立 msprobe 比较均拒绝对齐，尚无可接受的独立首差。25 完成两 rank 原生通信观察，但没有原生差异样本、没有 F，通信组退出与恢复未通过。** 本报告没有启动新测试或改变 Provider。

审计对象为 2026-11-17 的 `tensor-a4`（`source-a7`）和 `collective25-a2`（`source-a6`），均运行于 124 / Ascend910B4 / CANN 8.5.1。镜像为 `quay.io/ascend/vllm-ascend:v0.18.0`，实际 image ID、源码包 SHA、结果路径、原始三件套 SHA 和阶段回执见 [机器可读审计摘要](tensor_collective_summary.json)。本轮不增加其他 CANN 版本的验收声明。

## 五项构造 NPU 机制结果

五项使用卡 5，均完成 `baseline → baseline_repeat → lossless → fault → off`，各容器退出 0。真实完整输出的 SHA 证明 B/B-repeat/L/OFF 精确相同，F 不同；五次调用的输入描述及哈希相同，输出均有限。F 各干预一次，原算子或原模块真实执行。该结果限定于构造模块，不是预训练模型或标准 vLLM Worker 的正向验收，也不授予 `all_eligible` / `every_match`。

| 机制 | 实际观察到的干预与结果 | 限制 |
| --- | --- | --- |
| 15.alias_detach | NZ29/BF16 base `[16,16]`；视图 `[8,8]`、stride `[16,1]`、offset 17。返回视图初值正确但不再关联原 base；真实 `copy_` 写入视图成功，原 base 目标区域保留旧值，非目标区域不变；实际 `aten.add.Tensor` 消费原 base | 固定合法几何。`consumer_result_reference_verified=null` 保留，不能宣称一般消费者 oracle 已验证 |
| 17.spatial_padding | 标准 `nn.Conv3d` 实际调用的 padding 变为 `(0,7,7)`；原算子一次，输出仍为 `[2,8,1,1,1]`，F 完整输出变化 | 固定 16×16 空间 patch；没有为任意自定义或融合 Conv3d 授权 |
| 18.api_half | 标准 `nn.RMSNorm` 原返回 `[4,32]` 后应用 0.5 增益，后续模块输出变化 | 返回增益代理，不是 Native Reduce 根因复现，也不涵盖融合/自定义 RMSNorm |
| 8.restore_offset | 实际 `npu_moe_token_unpermute` 的 I/P 按同一组映射 `[1,2,3,3]`；H/I/P 原存储不变，真实原生消费者一次；**完整 F 原生输出精确等于 B 原生输出按该映射取组** | T=4、K=2、输出 `[4,128]` BF16 的构造 workload；没有证明生产 MoE 调度和 producer 归属 |
| 22.wrong_row | 实际 `aten.index_select` 一次，输出 `[4,8]` 的第 3 行由 source 第 3 行改取第 0 行；source/index/非目标行不变；后续两个真实 `aten.mul.Tensor` 消费同一 lookup 存储 | 构造 RoPE 运算链的错行输出代理，不是 uint32 溢出或融合 RoPE 的验收 |

Python Provider 的执行边界是 `python_hook / OWNED_PYTHON_HOOKS`，`native_loaded=false`、`native_registered=false`。清理回执均为 `registered=false`、`enabled=false`、`hook_owned=false`、`leases=0`、`inflight=0`。原始 `recovery_confirmed=null` 未改写；本轮局部恢复依据是实际 OFF 完整输出精确恢复及无活动 hook/lease/inflight，不是将 null 解释为通过。

## 独立 statistics/mix：全部保留拒绝

每组只采 B 的真实 `step0` 和 F 的真实 `step3`，其余三次只做运行时完整输出检查；这些 step 是模块重放阶段，不是 decode token。采集配置为 `statistics / mix / summary_mode=statistics`，相同原生 collector 未被实验改写。比较器只读取原始三件套、配置和 collector 身份，不读命中日志或完整张量；固定 `|F−B| > 1e-8 + 1e-5 × |B|`，baseline 为参考。

| 机制 | B/F 独立结果 | 拒绝原因字段 | B/F 的 null 字段数 | B 自比记录数 |
| --- | --- | --- | --- | --- |
| 15 | `ALIGNMENT_MISMATCH` | `structure` | 52 / 58 | 23 |
| 17 | `ALIGNMENT_MISMATCH` | `structure` | 16 / 16 | 8 |
| 18 | `ALIGNMENT_MISMATCH` | `structure` | 10 / 14 | 5 |
| 22 | `ALIGNMENT_MISMATCH` | `stack` | 24 / 24 | 12 |
| 8 | `ALIGNMENT_MISMATCH` | `stack` | 12 / 12 | 6 |

五项均为 `whole_dump_comparable=false`、首差未确定。B 自比是**同一份 dump 与自身比较**，结果为 `NO_OBSERVABLE_STATISTICS_DIFFERENCE`；它不是两份重复采集的独立一致性结论。null 只能说明不可观测，不能按零值或相同处理。机制运行时证据不能绕过统计对齐门禁，不能据此指定首差为注入算子。

原件位于 `/home/17-fault-engineering-20261117/tensor-a4/tensor{15,17,18,8,22}/statistics/observation/step{0,3}/rank0/`；独立报告位于审计机 `.codex-tmp/engineering-20261117/tensor-a4-analysis/`。摘要保留精确绝对路径与 SHA，不提交原始统计、完整张量或历史日志。后续需要解决真实调用关系/结构的可证明对齐后重新比较，不得靠忽略不匹配或预指定首差来通过。

## 25：原生观察有证据，注入与整组恢复没有通过

卡 2/5 上两个真实 HCCL rank 各完成两次同步 `reduce_scatter_tensor(SUM)`：FP32、发送 `[256]`、接收 `[128]`、成员 `[0,1]`，实际 native sequence 分别为 `0→1`、`1→2`。输入哈希在各 rank 内保持一致，原通信没有新增或跳过，原生返回真实完成。两份 rank 回执均为：

```text
status = NO_REPLAY_SAMPLE
reason = NO_NATIVE_SAME_INPUT_DIFFERENCE_OBSERVED
repeat_exact = true
off_exact = true
hits = 0
```

随后执行的是无注入的 OFF 原调用；没有校准样本签发，也没有 F 或 B/F statistics 首差。无差异是有效负结果，不能视为故障注入成功，OFF 相同也不能证明未执行过的故障已恢复。

rank 回执写出后 `torchrun` 长时间未退出，本轮主控对自己拥有的容器执行 `docker stop`。实际控制回执为 `CASE_FAILED` / **exit 1** / `OOMKilled=false`，日志显示 PID 1 收到 **SIGTERM（15）**；不是外部任务干扰。以该原始回执为准，不采用先前口述的 exit 137。`LOCAL_SYNCHRONOUS_DRAIN_ONLY_CROSS_RANK_RECEIPTS_REQUIRED` 原字段保留，**没有跨 rank drain/正常进程退出证据，E4 不通过**。

结果及控制回执根目录为 `/home/17-fault-engineering-20261117/collective25-a2/`。后续仍缺真实同输入原生差异样本、实际 F、通信组恢复及预训练模型/标准 Worker 验收；不能用合成 delta、两项求和顺序或 LCCL 历史结论填补这些缺口。
