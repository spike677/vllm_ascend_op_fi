# 故障效果评估设计 V1

## 适用范围

本设计适用于已通过 developer-mode 授权、目标绑定和 lossless validation 的本地 Fault。它不实现 native mutation，也不将一个模型/版本/硬件上的结果推广到其他环境。

## 执行状态机

```text
capability / discovery
        ↓
lossless validation
        ↓
freeze fault configuration
        ↓
clean baseline
        ↓
arm + mechanism validation
        ↓
fault measurement
        ↓
finally: disarm / remove
        ↓
recovery measurement
        ↓
repeat N complete cycles
        ↓
effect profile
```

一个完整 cycle 必须包含 baseline、fault、recovery；多个样本的 re-arm 不能计为多个 cycle。所有 phase 应固定模型、版本、TP/DP 拓扑、请求集、temperature、seed、max tokens 和请求顺序；唯一有意变化的是冻结的 fault configuration。

## 运行前门禁

有损 phase 前，必须完成与同一 logical target 相匹配的 lossless gate：真实 target 可解析、临时 binding/替换可执行、输出与 baseline 等价、临时资源释放、service health 正常。任一 binding/ABI/metadata/stream/restore 不确定时 fail closed。

每次 arm 后必须在 `finally` 中 disarm/remove；未观察到真实 hit 或 `changed_elements > 0` 的请求标记为 `INVALID_FAULT_RUN`，不得被解释成“fault 没有效果”。

## 输入与数据协议

| 类别 | 用途 | 必需元数据 |
| --- | --- | --- |
| 固定任务集 | Accuracy、Retention、Paired Damage | sample ID、reference、raw output、解析结果 |
| 短输出 probes | 格式、换行、短路径分叉 | prompt、response、比较结果 |
| reference text | teacher-forced NLL/PPL | dataset/path、窗口、计分 token 数 |
| 长输出 probes | 重复、刷屏、早停、runaway 等 phenotype | prompt、预注册阈值、raw output、分类 |

Parser、finish reason、protocol compliance 与 task correctness 必须分别记录。答案提取不应使用“全文最后一个数字”；优先使用第一个行首 `FINAL_ANSWER=<number>`、第一个 `#### <number>` 或第一个明确 answer expression。

## 多 rank 机制记录

每次 fault request 至少保存：request/request generation、fault config/runtime hash、target/layer/module、expected/observed rank、candidate/hit/changed elements、reject reason、worker PID/liveness、HTTP/health、CANN/HCCL/OOM/worker-exit 指示和 disarm/remove 状态。

对于 rank0-only policy，rank1 的 `hit=0` 与 `rank_filter` 是 expected reject，不构成失败。

## 概率评分

概率效应仅用 teacher-forced reference-token log probability：

```text
NLL = -Σ log p(reference token | preceding reference tokens)
PPL = exp(NLL / scored token count)
NLL Delta = NLL_fault - NLL_baseline
PPL Ratio = PPL_fault / PPL_baseline
```

它不是 hidden-state 统计，也不是自由生成文本的评分。必须保存 reference 数据、固定窗口和实际 scored token count。

## 输出与表征

若服务没有公开整数 token IDs，则 `token_ids=null`，比较基准写为 `token_pieces`，不得混称。短输出记录 equality、首个分叉、长度比、格式/换行和答案解析；长输出记录重复、周期 loop、标点刷屏、early stop、runaway、empty、garbage heuristic 与 crash/error。

判定阈值必须在运行前冻结。输出分叉仅证明 trajectory effect；它不自动等于病态 collapse。

## 重复性、恢复与结果等级

```text
Injection Hit Rate = valid fault requests / expected fault requests
Effect Repeatability = 预定义效果条件满足的完整 cycle / valid complete cycles
Recovery Rate = clean match + health + 无残留状态的 cycle / valid complete cycles
Crash Rate = crash fault requests / valid fault requests
```

N=3 可作为重复性证据，不能宣称统计显著性。Recovery 是实验有效性/因果 gate，不是 severity metric；若 clean baseline 自身波动，则 task effect 和 recovery 的表述必须降级。

报告等级：`FIXTURE`、`LIVE_SMOKE`、`LIVE_REPEATABLE`、`FORMAL_BENCHMARK`。最终状态为 `PASS`、`PARTIAL`、`FAIL` 或 `INVALID`，并指出未满足的预先声明 gate。

## Trace policy 与交付物

默认保存配置、聚合计数、请求/响应、phase 结果与 hash。逐算子 metadata/run trace 仅作为带配额、轮转、磁盘余量 guard 和归档 manifest 的 forensic 选项。JSONL、dump、raw trace、gzip、binary 均不迁入 `docs/`。

一轮完整评估至少交付：

```text
manifest.json
frozen config / runtime hash
cycles/cycle_N/{baseline,fault,recovery}/raw results
mechanism.json
evaluation_result.json
repeatability.json
effect_profile.json
effect_evaluation_summary.md
baseline/parser audit（如适用）
trace archive manifest（如启用 forensic trace）
```
