# 故障覆盖验收标准

本轮主目标是唯一目标、无损、单次注入和恢复。多目标与`every_match`必须逐机制、
逐实际环境取得资格，不能修改`QualifiedNativeSession`现有单次门禁来绕过资格检查。

## 相互独立的等级

| 标记 | 证据要求 |
|---|---|
| AUDITED | 实际源码、入口、边界、门禁及历史证据逐项核对 |
| IMPLEMENTED | 存在可执行当前实现，不只是注册名或设计 |
| CPU_VERIFIED | 实际CPU/合同/mock测试回执；执行种类保留CPU |
| NATIVE_VERIFIED | 当前源码/构建、实际SoC/CANN、实际native调用、干预/无损及恢复回执 |
| MODEL_CLOSED | 上述机制发生于真实模型链、请求/调用身份正确、B/F/OFF及独立统计复核完成；模型和路径范围明确 |
| COVERAGE_VERIFIED | 记录范围内独立分母、完整符合条件调用、预算、rank隔离和整组恢复全部通过 |

小型构造模块可以取得NATIVE_VERIFIED机制级结果，不因名字叫Model就成为预训练模型闭环。
旧standalone实验、CPU桥、共用Worker、图模式和eager分别记录；不能跨入口、build、
模型实现或硬件转移资格。合法策略切换已执行与数值误差已出现分列。

## 需要同时记录的分母

1. 结构候选数：真实模型发现的module实例，别名合并但路径保留。
2. 动态观察数：实际完成forward的operator/API调用数。
3. 条件匹配数：观察者独立判断符合完整合同的调用数。
4. 干预数、执行完成数、consumer完成数、恢复完成数：分开统计。

`hit==hit`不能证明覆盖；0/0不能标为100%。只发现但未走到的动态分支必须标NOT_OBSERVED。
按照scope限定rank/request/forward时，未选范围也要有透传/零变更证据。
FAULT26等延后consumer必须等consumer及storage恢复完成才能把一次claim算完成。

## 每个Provider的资格维度

| 维度 | 必需验证 |
|---|---|
| single_target | 真实唯一目标、无损成功、单次实际命中、OFF恢复 |
| multi_target | 目标身份无歧义、各自owner/资源、并行或串行调用互不污染 |
| every_match | 独立eligible分母等于实际干预及完成数；有界连续执行和未匹配透传 |
| rank_isolation | 实际rank/device映射，目标rank有效、非目标rank零干预 |
| resource_budget | 临时字节、并发/在途、调用数、生命周期均受界；OOM/超时不继续复用 |
| group_recovery | 全组停止新入场、drain、逐目标释放/恢复、fresh-state对照；一个失败不能报整组成功 |

单点qualified会话保持单次限制。旧all-hit research harness的结果仅属于其冻结输入/环境，
不自动开放新共用Worker的`all_eligible + every_match`。

## 当前容易混淆的事实

- 20的WeightNz slot证明不能替代QMM；Qwen与SmolLM2两个模型不等于GLM/DeepSeek原生架构通过。
- 26 FIA历史最终是126完成/rank；第一层无当前forward producer应透传，不是漏注入。
- 26 QMM rank1 step0统计NaN阻断首差；native恢复和统计独立定位是两项结论。
- 共享量化只选首个实际eligible模块；scheduler行映射已证不意味着每个请求的scale/Q都变化。
- 33是28层×3轮=84次launch，每次处理8请求，不能再乘8宣称672次独立native调用。
- 28合法切分与分母重复累计必须分开；0.7968%历史Norm不能归给合法切分。
- 15自然模型必须有view原地写入与同base消费，只读取私有权重view不能算同一链路。
- Fault39独立包和共用Extension必须分别验收；输出代理不能借范围异常模式的证明。

## 门禁失败与推进

每条依次记录G0源码、G1环境、G2模型、G3算子、G4身份、G5无损、G6干预、
G7观察、G8恢复、G9独立统计。最多三轮有明确目的的探索：源码/历史检查、
最小路径、针对单个明确阻塞修复验证；不得把三段重复扫描伪装为三次实测。
无损失败不进入有损；恢复失败隔离当前Worker。结构性阻塞保存回执与具体下一步，
标`BLOCKED_WITH_EVIDENCE`并转下一机制，不伪造缺少的后续阶段。
