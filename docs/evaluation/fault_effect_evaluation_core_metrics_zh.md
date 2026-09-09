# 故障效果评估：核心逻辑与指标

> 面向汇报、PPT、新开发者与技术评审。本文只说明：给定一个已经通过绑定与无损门禁的 Fault，DemonCATPFI 如何判断它是否真正发生、造成何种影响、是否稳定，以及解除后能否恢复。

## 一句话概括

```text
一个 Fault 的“有效”不是看到输出变了，
而是：真实命中 → 数值改变 → 影响被分层度量 → N 次可重复 → 解除后无残留。
```

## 主流程

```text
                 Validated Fault
                        │
                        ▼
                 Clean Baseline
                        │
                        ▼
                 Inject Fault
                        │
                        ▼
                Mechanism Evidence
                hit / rank / change
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Task Effect   Probability    Output Effect
      Accuracy      NLL / PPL      Divergence
      Retention                    Phenotype
      Damage
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                 Repeat N Cycles
                        │
                        ▼
                    Reliability
                        │
                        ▼
                 Disarm / Recovery
                        │
                        ▼
                 Fault Effect Profile
```

`Validated Fault` 的前提是：目标语义和运行时 binding 已解析，lossless validation 已通过，并且当前环境满足 ABI、版本、模型、设备、rank 和 developer-mode 等门禁。它不代表任意模型和环境都可直接 arm。

## 核心指标总表

| 层次 | 当前核心指标 | 得到什么结论 |
| --- | --- | --- |
| Mechanism | hit、changed elements、rank conform | 故障是否真的发生 |
| Task | Accuracy、Retention、Paired Damage | 任务能力损伤 |
| Probability | NLL Delta、PPL Ratio | 概率质量退化 |
| Output | divergence、first divergence | 生成轨迹变化 |
| Phenotype | repetition、spam、early stop、runaway、crash | 具体坏成什么样 |
| Reliability | N-cycle repeatability | 效果是否稳定 |
| Recovery | clean match、health | 故障是否可逆 |

这些层次必须分开汇报：PPL 上升可能早于答案翻转；输出分叉不必然是刷屏或崩溃；任务准确率不变也不表示 fault 没命中。

## 1. Mechanism：故障是否真的发生

最小证据：

```text
target resolve
hit count
changed elements > 0
expected rank 与 observed rank
reject reason
request/worker/service health
```

它回答“Fault 是否在真实请求的目标位置生效”，而不是“hook 是否加载”。例如 rank0-only Fault 中，rank0 的 hit/changed elements 应满足预期；rank1 的 `hit=0` 与 `rank_filter` 是正确对照，不是失败。

如果请求未 hit、目标未解析、数值没有改变或 native/runtime 证据缺失，则该请求是 `INVALID_FAULT_RUN`，不得把“输出没有变化”计作 fault 无效。

## 2. Task Effect：任务能力损伤多少

在同一固定任务集、同一提示词、同一 generation protocol 下比较 baseline 与 fault。

```text
Accuracy = 正确且可解析样本数 / 固定样本总数
Retention = Fault Accuracy / Baseline Accuracy
Paired Damage = baseline 正确且 fault 错误的样本数
Paired Damage Rate = Paired Damage / baseline 正确样本数
```

最重要的规则：**baseline 本身错误的样本不能算 fault damage**。`WRONG_ANSWER` 与 `UNRESOLVED_ANSWER` 也必须区分。finish reason、是否输出了 `FINAL_ANSWER` 和答案是否正确是独立字段；模型在答案后继续输出，不能让 parser 去取全文最后一个无关数字。

固定 GSM8K-10 这类数据是 smoke set，只能表述为“在该固定 10 题集上观察到的 paired damage”，不能替代正式大样本 benchmark。

## 3. Probability Effect：正确 token 的概率质量坏多少

任务答案还没有翻转时，teacher-forced reference scoring 可以显示更早的退化：

```text
reference tokens
      ↓
model log probability
      ↓
NLL
      ↓
PPL
```

```text
NLL Delta = NLL_fault - NLL_baseline
PPL Ratio = PPL_fault / PPL_baseline
```

PPL 必须来自模型对固定 reference token 的 log probability；它**不是** hidden-state 数值统计，也**不是**对自由生成文本的评分。报告必须携带本地数据路径、窗口协议和实际 scored token count。

## 4. Output Effect：生成轨迹是否变化

对相同请求，记录：

```text
response equality
first divergent token-piece
token/token-piece divergence
generation length delta
output length ratio
```

若 API 没有整数 token IDs，则必须记作：

```text
token_ids = null
comparison_basis = token_pieces
```

不能把 token pieces 冒充 token IDs。Output divergence 只说明生成轨迹变化，不表示模型一定崩溃或任务一定做错。

## 5. Phenotype：模型具体坏成什么样

在短输出和长输出中独立检测：

```text
repetition / periodic loop
punctuation spam
early stop
runaway generation
unresolved answer
empty output / obvious garbage
HTTP error / worker or service crash
```

核心区分：

```text
trajectory divergence
    !=
pathological collapse
```

例如长输出从 baseline 分叉，但没有重复、刷屏、早停、失控或崩溃，应被准确表述为“trajectory divergence”，不能夸大为“模型崩溃”。所有阈值（循环周期、重复次数、标点比例、runaway 长度等）应在运行前固定。

## 6. Reliability：效果是否偶然

一个完整 cycle 是：

```text
clean baseline
→ arm
→ fault
→ collect mechanism evidence
→ disarm/remove
→ recovery
```

N 次请求各自 re-arm 不是 N 个完整 cycle。建议至少记录 `N=3` 个独立 cycle，并计算：

```text
mechanism repeatability
effect repeatability
crash rate
```

`effect repeatability` 的条件要预注册，例如每个 cycle 至少出现 Paired Damage、PPL ratio 超过固定阈值或 baseline-relative phenotype 中的一种。N=3 可说明可重复性，不代表统计显著性。

## 7. Recovery：故障解除后是否恢复

Recovery 的步骤是：

```text
disarm / remove fault
→ 确认 controller/hook 无残留
→ service health
→ 同一请求重新运行
→ 与 baseline 比较
```

它回答的是“变化是否可逆、是否存在残留污染”，因此是**实验有效性 Gate**，不是 severity metric。若 baseline 自身不稳定，task recovery 不得强行写成精确恢复；报告应保留 `PARTIAL` 并标出是 baseline variability，而不是掩盖问题。

## 最终交付：Fault Effect Profile

最终 profile 应同时包含：

```text
Mechanism
Task Effect
Probability Effect
Output / Phenotype
Reliability
Recovery Gate
环境、模型、拓扑、runtime/config hash
raw artifact 与 evidence pointer
不支持的结论
```

完整的工程协议、数据要求、公式、trace policy 与结果等级见[故障效果评估设计 V1](fault_effect_evaluation_design_v1.md)。
