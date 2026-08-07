# Fault 20 最终能力矩阵

日期：2026-08-07

## 状态定义

| 状态 | 含义 |
|---|---|
| `PROVEN_E2E` | 真实模型请求、真实目标算子 event、runtime 状态、baseline/fault/post 证据闭环 |
| `PROVEN_RUNTIME` | trace/metadata/调用关系成立，但该能力没有独立模型输出闭环 |
| `IMPLEMENTED_NOT_VALIDATED` | 已有实现或离线测试，未在目标模型路径完成 E2E |
| `NOT_STARTED` | 没有实现或验证；设计文档不改变此状态 |

所有 `PROVEN_*` 都附加 `VERSION_SCOPED=true`。

## A. Runtime 能力

| 能力 | 状态 | 证据/说明 | 边界 |
|---|---|---|---|
| 精确 `LD_PRELOAD` symbol interpose | `PROVEN_E2E` | 最终 DSO 导出 Run/Add/AddParam；all-hit 七阶段使用 | 不是 broad dlsym/dlopen hook |
| `OpExecCache::Run` hook | `PROVEN_E2E` | QMM 与 MatmulWeightNz 均由真实 Run 使用 temporary 地址 | repeat cache only |
| `AddOpExecCache` raw cache 注册 | `PROVEN_E2E` | typed GWS TLS 与单次 Add 关联，随后同 cache Run 命中 | CANN 8.3.RC2 内部 ABI |
| `AddParamToBuf` 参数顺序收集 | `PROVEN_E2E`（Matmul 组成能力） | self/mat2/out 三项顺序与 typed GWS 指针一致 | 当前两个重载和 ABI0 |
| immutable metadata registry | `PROVEN_E2E` | 两个 adapter 都通过 `cache -> metadata` 完成替换 | 不缓存 activation address |
| address vector copy substitution | `PROVEN_E2E` | 复制 vector，只改 slot0；lossless/numeric 全通过 | 禁止原地修改调用方 vector |
| POSIX SHM control v5 | `PROVEN_E2E` | magic/version、enable/disarm、mode/API/mask/shape/hit 全贯通 | `/fault20_l4_buffer_substitute_ctl_v5` |
| bounded/single-hit selector | `PROVEN_E2E` | QMM/Matmul 单 hit E2E | exact guard only |
| same-rank multi-hit | `PROVEN_E2E` | QMM lossless + INT4，各连续 3 hit | 非并发压力 |
| dual-rank/device mask selector | `PROVEN_E2E` | mask `0x3`，TP rank0/rank1 独立计数 | 当前两设备映射 |
| all-hit selector | `PROVEN_E2E` | 短/长请求，per-device candidate=hit=event | eligible guard 通过后的 all-hit |
| per-device atomic candidate/hit/status | `PROVEN_E2E` | 短 QMM 每 rank 1920；长任务 15488/16256；无旧 1024 上限 | 64 个 control 槽位，实际只验 device0/1 |
| fail-closed | `PROVEN_E2E`（代表性路径） | disabled、错误 guard、失败时 real Run；构建/ABI/runtime 门禁 | 不是所有错误分支的故障矩阵 |
| D2D lossless temporary lifecycle | `PROVEN_E2E` | QMM/Matmul 两 adapter | 同步回收 |
| D2H→CPU transform→H2D lifecycle | `PROVEN_E2E` | INT2/INT4/BF16；changed-elements 与 real Run event | 同步、单请求实验 |
| structured JSONL logging | `PROVEN_E2E` | event、status、timing、changed、free 状态均有证据 | 不是生产 telemetry schema |
| event-fenced async buffer pool | `NOT_STARTED` | 只有设计方向 | 当前每 hit 同步分配/回收 |
| non-repeat-cache runtime | `NOT_STARTED` | 无 adapter 或 E2E | cold/new shape 不能外推 |
| cross-version adapter selection | `NOT_STARTED` | 无版本矩阵和选择器 | 当前仅单栈 |
| concurrency/soak/performance qualification | `NOT_STARTED` | 多 hit/多 rank不等于并发或稳定性 | 需独立阶段 |

## B. Operator Adapter 能力

### B1. QuantMatmulWeightNz Adapter

| 字段 | 当前值 |
|---|---|
| 状态 | `PROVEN_E2E` |
| API | `aclnnQuantMatmulWeightNz` |
| 输入语义 | dynamic activation `x1` |
| target slot | `addresses[0]` |
| dtype/format | INT8 / ND |
| 核心 decode shape | `[1,5120]` |
| stride/offset/bytes | `[5120,1]` / 0 / 5120 |
| producer guard | `aclnnAscendQuantV3` output address == 当前 QMM `addresses[0]` |
| metadata 来源 | typed QMM GWS 正式 `aclTensor` getter |

验证矩阵：

| 场景 | 状态 |
|---|---|
| lossless single hit | `PROVEN_E2E` |
| INT4-equivalent single hit | `PROVEN_E2E` |
| INT2-equivalent single hit | `PROVEN_E2E` |
| same-rank 3 hit | `PROVEN_E2E` |
| TP dual-rank each-one | `PROVEN_E2E` |
| dual-rank all-hit lossless | `PROVEN_E2E` |
| dual-rank all-hit INT4 | `PROVEN_E2E` |
| dual-rank all-hit INT2 | `PROVEN_E2E` |
| long task all-hit | `PROVEN_E2E` |
| prefill/warmup/non-cache | `NOT_STARTED` |

### B2. MatmulWeightNz Adapter

| 字段 | 当前值 |
|---|---|
| 状态 | `PROVEN_E2E` |
| API | `aclnnMatmulWeightNz` |
| formal parameter/storage slot | `0=self`、`1=mat2`、`2=out` |
| target slot | `addresses[0] = self` |
| observed dtype/format | BF16 / ND |
| 核心 decode shape | `[1,12800]` |
| stride/offset/bytes | `[12800,1]` / 0 / 25600 |
| metadata 来源 | typed MatmulWeightNz GWS + `AddParamToBuf` 顺序证明 |
| registration guards | 3 params、all present、no empty storage、no shared storage |

验证矩阵：

| 场景 | 状态 |
|---|---|
| 三槽位 trace | `PROVEN_RUNTIME` |
| lossless single hit | `PROVEN_E2E` |
| BF16 truncate single hit | `PROVEN_E2E`，早期请求产生 3-token 差异 |
| dual-rank all-hit lossless | `PROVEN_E2E` |
| dual-rank all-hit BF16 truncate | `PROVEN_E2E_MECHANISM`；23,033,817 elements changed，短请求 token 不变 |
| FP16 truncate | `IMPLEMENTED_NOT_VALIDATED`；仅离线穷举测试 |
| other shape/prefill/cold/non-cache | `NOT_STARTED` |

`PROVEN_E2E_MECHANISM` 是对证据强度的说明，不是新的全局状态枚举：该阶段原始 `evidence.json` verdict 为 `PASS`，证明算子侧数值变化和真实 Run；它没有证明本次短请求输出一定变化。

## C. Fault Primitive 能力

| Primitive | 输入/输出 storage | 状态 | 模型侧结果 |
|---|---|---|---|
| lossless D2D | same dtype/shape/bytes | `PROVEN_E2E` | token 与 baseline 一致 |
| INT4-equivalent | INT8 → INT8，16 levels | `PROVEN_E2E` | 短/长任务乱码和语义破坏 |
| INT2-equivalent | INT8 → INT8，4 levels | `PROVEN_E2E` | 短/长任务重复 token 坍缩 |
| BF16 mantissa truncate keep=3 | BF16 → BF16，低 4 mantissa bits 清零 | `PROVEN_E2E` | single-hit 有 token 差异；all-hit 短请求无差异但机制成立 |
| FP16 mantissa truncate | FP16 → FP16 | `IMPLEMENTED_NOT_VALIDATED` | 65,536 raw encoding 离线测试通过；无真实 FP16 event |

## D. Control / Harness / Evidence

| 能力 | 状态 | 说明 |
|---|---|---|
| CLI reset/status/disarm | `PROVEN_E2E` | v5 control |
| CLI exact API + mask + all-hit arm | `PROVEN_E2E` | QMM/MatmulWeightNz |
| baseline/fault/post 固定流程 | `PROVEN_E2E` | runner 保证 finally disarm |
| response token ID 采集 | `PROVEN_E2E` | `return_token_ids=true` |
| coverage 断言 | `PROVEN_E2E` | candidate==hit==event per device |
| changed-elements 审计 | `PROVEN_E2E` | numeric stage 必须 >0 |
| 历史 failed/invalid evidence | 本地保留 | 不进入公开仓库 |

## E. 明确未支持

- 任意算子自动适配；
- kernel 内部、指令级或硬件级故障；
- packed INT2/INT4；
- QMM scale/weight/bias 修改；
- FP16 模型级 E2E；
- prefill 大 shape、warmup/cold、non-repeat-cache；
- 并发请求、长期 soak、性能 SLA；
- CANN/torch_npu/vLLM 跨版本；
- x86、MoE、GroupedMatmul、MatmulAllReduce；
- 外部故障框架集成不在本次统计范围；当前设备不存在 DemonCATPFI。

## 关联文档

- [最终架构](fault20_final_architecture.md)
- [primitive 设计](fault20_fault_primitive_design.md)
- [当前注入能力统计](fault20_injection_capability_inventory_20260807.md)
