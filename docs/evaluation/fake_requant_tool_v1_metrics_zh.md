# FakeRequant 工具 V1：指标、字段与结论边界

本说明描述**现有实现**，不修改检测阈值，不要求四模块族×20重新测量全部指标。默认评分器是 `DISTORTION_SCORER_V2_TASK_TERMINATION_PATHOLOGY`；配置文件虽名为 `distortion_probe_suite_v1.json`，其当前内容版本为 `NORMAL_DIAGNOSTIC_SUITE_V2`。版本与源码哈希见 [发布清单](../releases/fake_requant_tool_v1/release_manifest.json)。

## 1. 先判断证据在哪一层

| 层次 | 核心指标 | 回答的问题 |
| --- | --- | --- |
| Mechanism | 目标、hit、changed elements、rank conformity | 故障真的施加了吗 |
| Task | Accuracy、Retention、Paired Damage／Improvement | 哪些原本正确的任务被打坏 |
| Probability | reference-token NLL、PPL及其比值 | 对相同参考序列的概率质量如何变化 |
| Output | 全文／token-piece差异、长度 | 生成轨迹是否变化 |
| Phenotype | 重复、词汇坍缩、标点、字符异常、提前终止、runaway、请求错误 | 输出具体出现什么异常 |
| Reliability | 完整周期、同题同异常重现 | 是否具有有限的重复验证证据 |
| Recovery | clean、health、全文／pieces精确恢复 | 可逆性和残留污染；是有效性 Gate，不是严重度 |

以下简写只为定位现有字段，不是新 schema：

- `R`：`evaluation_result.json`。
- `M`：`R.distortion.cycles[i].matrix[j]`，固定 cycle/sample 的配对结果。
- `F`：`pair.json.pair.fault`，完整原始故障请求；`W`：`F.fault_transaction.measurement_query.results[k]`。
- `P`：`target_plan.json`；`A`：`R.analysis_details.cycles[i]`（仅完整历史 replay 有此层）。
- `S`：`W.effect_summary.per_module[module_name]`，有界逐模块累计统计。

缺失、未采集、证据失配、不适用必须区分。顶层 `quality_performance=null字段` 不表示逐周期一定没有概率数据；应查 `analysis_details`。不存在的数据不能填0或借用另一配置。

## 2. 目标集合与三个不同“命中率”

| 指标／实际字段 | 来源与计算口径 | 状态与限制 |
| --- | --- | --- |
| 结构候选 `model_target_reports[].targets` | 实际 config／类／父路径／角色规则；保留 blockers | 已实现；不等于经过真实入口 |
| 已观察 `targets[].observed` | 同 epoch 的 Inspect 调用样本 | 未观察不能自动算不可执行，也不能算支持 |
| 已绑定 `targets[].bound/binding` | 真实 args／kwargs、类型、layout、局部分片维、别名保护 | Qwen实测；DeepSeek只读候选未认证 |
| 计划集合 `P.selected_modules/workers[].targets` | 本次选定的明确集合，融合去重、禁止父子重叠 | 分母来自架构预期和计划，不只来自“找到几个” |
| 实际安装／命中 `W.matched_modules/per_module` | 在真实 worker 查询；按rank/layer/module聚合 | 安装≠命中；不按 results 数组顺序推断rank |
| `W.hook_seen/candidate_seen/hit_count` | 入口看到、符合选择条件、实际变换的计数 | 具体控制流程由 frozen reference 决定；不能混用 Fault20 native 计数 |
| `W.reject_reasons` | 原始原因计数，如rank_filter/max_hits | 未选rank的rank_filter是预期行为；目标rank有未解释拒绝不能认证调用窗口 |
| `R.reliability.injection_hit_rate` | 当前 V2 = 机制 PASS 请求数 / 已审计请求数，含存在的概率 auxiliary 请求 | **请求级机制通过比例**，不是所有算子调用都命中的比例 |
| 已观察合格调用注入比例 | `hit_count/candidate_seen`，必须说明reference选择阶段和rank | 分母0/未知应留空；比例1仍可能遗漏完全绕过hook的路径 |
| `R.model_targeting.window.target_call_window` | 逐目标、逐rank hook/candidate/hit一致且未触顶、合同有效 | Qwen PASS；只覆盖观察到的目标入口调用 |
| `R.model_targeting.window.full_request_window` | 还要求独立 scheduler execution-step 与请求/epoch计数对应 | 当前 UNPROVEN；不得从hook分子分母一起遗漏推导全程覆盖 |

默认机制检查在 [mechanism.py](../../src/ascend_fi/evaluation/distortion/mechanism.py)；新计划附加检查在 [plan_executor.py](../../src/ascend_fi/inspect/plan_executor.py)。缺失 arm/measurement/disarm、rank不符、hit=0、changed=0、输入合同失配的有损请求标为 `INVALID_FAULT_RUN`，不按“模型无效果”算分。

## 3. 数值效果：全事件聚合与采样分开

实现：[effect_stats.py](../../src/ascend_fi/inspect/effect_stats.py)、[experiment_extension.py](../../src/ascend_fi/inspect/experiment_extension.py)、[exploration.py](../../src/ascend_fi/inspect/exploration.py) 的 `dose_summary`。这些数据不等于任务准确率。

| 指标 | 实际字段／计算 | 分母、空值与限制 |
| --- | --- | --- |
| 累计改变元素 | `S.changed_elements`；总和进入 `R.mechanism.changed_elements` | 每个被接受的activation调用中 `x' != x` 的元素事件；同一存储多次参与会重复计数，不是唯一权重数量 |
| 累计元素 | `S.total_elements` | 所有接受调用的元素事件总和，不是模型参数量 |
| 改变调用 | `S.changed_calls/S.hit_count` | 一次调用至少一个元素改变；IDENTITY应为0 |
| 绝对误差 | `S.max_abs_error`、`abs_error_sum/total_elements` | 聚合既有reference统计；不是每个tensor等权平均 |
| 裁剪率 | `sum(clip_count)/sum(total_elements)` | 全接受事件；倍率大导致归零不一定导致clip |
| 新增归零率 | `sum(sampled_new_zero_elements)/sum(sampled_nonzero_elements)` | **只在采样调用内**，分母为原非零元素，不是全部元素；分母0无定义 |
| 相对L2误差 | `sqrt(sum(sampled_error_sq_sum)/sum(sampled_input_sq_sum))` | FP32诊断和的采样聚合；不是逐模块相对L2简单平均；输入平方和0则空 |
| 实际fault scale | `S.fault_scale_min/max`；逐族取min/max | 参考步长已乘m；原reference尺度可由fault_scale/m推导，不冒充原生量化参数 |
| 日志完整性 | `effect_summary.event_count/contiguous_hit_indices/errors` | 聚合覆盖全部接受调用；单事件只保留首16/尾16，不能从32条日志推定仅32hit |

附加 zero/L2 采样为每模块第1、65、129…次接受调用；采样规则不决定是否注入。phase与生成token位置在此层目前为 `UNKNOWN`，不能按 `[1,K]` 断言decode。量化级占用 `effective_code_values=NOT_SAMPLED`；不是已交付的全量统计。

当前 Qwen×20 样例的changed为1,850,723,432；这仅说明数值影响确实发生。裁剪率0与大量新增归零可以同时成立。

## 4. Task：答案、未交付和配对损伤

默认 V2 使用 [detectors.py](../../src/ascend_fi/evaluation/distortion/detectors.py) 的 `task_fact`，结果在 `M.task_outcomes.{baseline,fault,recovery}`，以及 `M.effect.detectors[detector=task_answer]`。

数字答案按规则优先级查找：第一个行首 `FINAL_ANSWER=<number>` → 第一个 `#### <number>` → 第一个明确 `answer/result is/:/=` 表达式 → 整个回答是裸数字。每条规则用首个匹配；不是取最后一个后续题答案。逗号移除、Decimal规范化；不要求marker在末尾、不要求自然停止。文本答案只接受冻结expected及有限别名。

注意：这是“规则优先级内第一个匹配”，不是跨所有格式按文本位置取最早。混合marker协议仍须审计。旧 `metrics.extract_gsm8k_answer`／`extract_explicit_final_numeric_answer` 保留末匹配语义供历史链路，**不是默认V2入口**；不把旧parser的错误结论搬进本版。

| 状态 | 判定与含义 |
| --- | --- |
| `CORRECT` | 有已冻结答案且解析相等；即使finish=length也可能正确 |
| `WRONG_ANSWER` | 已解析但不相等；不代表已定位推理中哪步算错 |
| `UNRESOLVED_ANSWER` | 未解析且非length；格式/表达等原因待查 |
| `ANSWER_UNDELIVERED_WITHIN_BUDGET` | 未解析且finish=length；不能自动叫数学算错、胡言乱语 |
| `STRUCTURE_VALID/INVALID` | 仅依既有结构validator，不等于语义正确 |
| `NOT_EVALUATED` | 没有独立expected／validator的开放题，不能给Accuracy |

设固定可配对样本总数N，baseline正确数CB，fault正确数CF：

```text
Accuracy_B = CB/N                    Accuracy_F = CF/N
Accuracy Retention = Accuracy_F/Accuracy_B
Paired Damage = count(baseline正确 AND fault不再正确)
Paired Damage Rate = Paired Damage/CB
Paired Improvement = count(baseline不正确 AND fault正确)
```

Damage应进一步按错误、未解析、预算内未交付分类，并保存sample IDs。baseline已错的题不计Damage。Retention不是“原本答对的题保住了多少”；其他题改善会抵消损伤。分母0时留空；整体Accuracy以N为分母，不能只报告解析成功样本准确率。

实现状态：当前默认dataset入口保存逐题Task事实，但开放MT-Bench没有expected，未输出GSM8K总分。`effect_profile.aggregate_paired_effects` 提供已有逐组配对损伤字段；旧 `suite._gsm_metrics` 与 `analysis_details` 可读历史Accuracy，但必须核对parser版本。Evaluation旧worktree的未提交GSM8K开发不是本版默认入口，也未偷偷合入。

旧 `resolved_answer_accuracy` 以已解析数为分母，是**条件准确率**；如引用必须同时给resolved/N，不能叫总体Accuracy。

## 5. Probability：token-PPL与旧word-perplexity不能混称

来源必须是同一参考文本、tokenizer、窗口和排除首token规则下的 teacher-forced/echo reference-token log probabilities；不能来自自由生成文本或hidden-state。

```text
NLL_sum = -sum(log p(reference_token_t | prefix_t))
NLL_mean = NLL_sum/T
token_PPL = exp(NLL_sum/T)
NLL_delta_sum = NLL_F_sum - NLL_B_sum
PPL_ratio = PPL_F/PPL_B
```

`T` 是实际评分token数，不是生成token数。delta为累计还是均值必须标注；窗口/token数不配对则不能直接比较累计delta。

| 字段／实现 | 准确含义和现状 |
| --- | --- |
| `A.probability.{baseline,fault,recovery}.nll/ppl/scored_token_count/window_count` | 完整历史FakeRequant周期的实际记录；当前默认重放保留它，不用新生成文本重新估PPL |
| `A.probability.nll_delta/ppl_ratio` | 历史配对累计NLL差和PPL比；与该历史故障配置绑定 |
| 历史示例：NLL_sum=544.7058697，T=189，PPL=17.8506806 | 符合 `exp(NLL_sum/T)` 的token归一化；只是旧R1示例，**不是四模块族×20结果** |
| `live.LiveQwenFaultExecutor.prompt_loglikelihood` | 已实现echo API调用，取prompt范围logprob；属于旧native评估owner，不是新的通用TargetPlan概率driver |
| `suite._run_ppl → word_perplexity(nll, word_count)` | 已实现但明确是按空白词数归一化的 **word-perplexity**，`exp(nll/word_count)`；即使另有scored_token_count，也不能重命名为token-PPL |
| `R.quality_performance.nll_delta/ppl/ppl_ratio` | 统一schema可承载；默认Distortion入口不填，可能在逐周期analysis_details有值 |
| 本轮新配置 | 概率效果 `NOT_MEASURED`；不填0，不借用旧QMM INT2/INT4或旧FakeRequant值 |

若API不返回合法参考logprob、有效token数不足、非有限值或配对窗口失配，应失败／留空并记录原因。不能为了指标齐全用其他配置补位。NLL/PPL是辅助概率指标，不参与本版 Distortion PASS。

## 6. Output与Phenotype：精确规则表

原文不strip掉换行用于全文比较；`M.{baseline,fault,recovery}.full_text` 保留完整输出。excerpt600字符只是旧展示，dataset Dashboard默认全文。空输出、未采集文本与展示截短不是同一个状态。

token序列选择：优先整数 `token_ids`，否则 `token_pieces`，再否则正则 `\w+|[^\w\s]` 的 lexical tokens；两边basis不同则不可比。Lexical统计单独移除纯空白字符串token。字符位置仅在pieces可精确拼接原文或lexical索引可证明时给出，否则null；不能把pieces冒充IDs。

| 指标／字段位置 | 计算与适用条件 |
| --- | --- |
| `M.effect.response_diverged` | 原始全文不相等；缺一侧原文则null |
| `M.effect.first_divergence/comparison_basis` | 同basis首个不同位置，0-based；仅长度不同则公共长度处；相等为null |
| `metrics.token_divergence.count/rate` | 对齐位置差异数加长度差，除以较长序列长度；非编辑距离。旧helper双空序列返回0，不能推广到未知分母 |
| `generation_length/delta/ratio` | token预算和API实际token数分别保留；`F/B`只在B>0有效；主matrix保留长度，非所有派生值都自动写顶层 |
| `repetition.seq_rep_4` | `1 - unique_4grams/(n-3)`；n<4的已观察短串返回0，不代表缺数据 |
| `same_token_max_run` | 相同token最长连续长度；`periodic_loop`检测精确相邻周期 |
| `repeated_fragment` | 首个周期或最常重复4-gram、前12位置、occurrence_count；位置区间半开 |
| `lexical_collapse.distinct_1/distinct_2` | 去纯空白后，唯一token/n、唯一bigram/(n-1)；短/空串有约定0，检测最小长度门禁仍有效 |
| `dominant_token_ratio` | 最常见token数/n；另存前三token及频数 |
| `punctuation.ratio/max_run` | Unicode类别P数量/非空白字符数；连续标点遇空白即断；不是所有非字母字符都算标点 |
| `garbage.bad_character_count/positions` | U+FFFD，或Cc/Cs但排除换行回车tab；位置最多32；不识别一般罕见Unicode、不推断语义错乱 |
| `generation_termination` | finish_reason、实际token、max_tokens、`budget_exhausted=(finish_reason==length)`；不等于pathology |
| `crash` detector | baseline健康后新出现REQUEST_ERROR或SERVICE_CRASH；HTTP失败单独归请求错误，不能据此定位进程或CANN根因 |

冻结阈值如下；正式值来自suite，源码比较运算符也属于版本合同：

| 检测器 | 当前触发条件 |
| --- | --- |
| Repetition | fault至少16token；新周期循环（周期1..8，至少3遍），或连续相同token≥6且baseline<6，或rep4≥0.30且增量≥0.20 |
| Lexical collapse | 两边去空白后至少32token；fault Distinct-1和2都严格小于baseline的0.5倍，dominance增量≥0.15 |
| Punctuation | fault比例≥0.35且增量≥0.20，或最长run≥8且增量≥4 |
| Early termination | baseline≥32token，F/B≤0.35，且独立事实表明空输出／答案未解析／结构不完整；正确答案／有效结构不算。开放短回答无独立不完整证据仅记录需复核 |
| Runaway | fault≥64token，F/B≥1.5，同时finish=length或实际token达到max_tokens；单独length不触发 |
| Garbage | fault异常字符≥2且baseline<2 |
| Crash/error | baseline健康、fault worker_alive=false或service_healthy=false为SERVICE_CRASH；非2xx或error为REQUEST_ERROR |

结构保护：**已有声明的结构validator**认定有效时，repetition和punctuation不触发；不是自动识别所有合法Markdown表格。开放题表格仍可能触发标点规则，需要单独上下文说明，不能改分数制造阴性或阳性。

V2会按实际输出长度新增通用检测器：任一侧≥16token增加repetition；≥32增加lexical；≥16或标点run≥8增加punctuation。因此“短题输出很长的循环”不会因原category不含检测器而漏掉。新加入的规则使用同周期baseline相对比较；旧calibration的NOT_APPLICABLE不是无异常证据。

`M.effect.output_pathology.confirmed` 表示已执行确定规则的告警，不是LLM语义裁判；`semantic_incoherence=NOT_PROVEN` 保留。`DIVERGED_VALID` 仅表示未触发覆盖规则，不保证语义正确。`UNRESOLVED`单独不计可见病态；WRONG_ANSWER可算已知任务损伤但未定位数学错误。

## 7. Repeatability与Recovery

| 字段 | 分母和边界 |
| --- | --- |
| `distortion.total_cycles/complete_cycles` | 完整B/F/R周期，不是请求数或逐题re-arm次数 |
| `same_probe_repeatability[id].distorted_cycles/total_cycles/stable` | 同题跨周期，而非不同题数量相同；达到原最小周期且全部周期同题异常才stable |
| `stable_phenotype_families` | 稳定样本在各周期共同出现的异常族，不是union |
| `distortion.crash_count` | 代码按verdict=CRASH计数，包含REQUEST_ERROR和SERVICE_CRASH；必须继续读detector subtype，不能宣称均为进程崩溃 |
| `reliability.effect_repeatability` | 默认Distortion入口顶层未填；已有EffectProfile/旧suite按各自记录定义，引用须写明分母和effect判据 |
| `recovery.per_cycle_scopes.all_text_exact` | 所有probe全文精确数/所有probe数 |
| `all_task_validator` | 有适用validator的恢复数/适用数；NOT_EVALUATED不进入此分母 |
| `fault_induced_phenotype` | 故障引起异常的pair里恢复数/异常pair数，不是全量恢复率；零异常时不可写100% |
| `runtime_clean_health/same_process` | 全worker query已disabled且uninstalled，health，配对进程身份；非仅HTTP可达 |
| `pair.json.recovery_exact` | 新计划额外要求HTTP200、全文相同、token pieces相同；不能用宽松的表型恢复替代 |

默认V2综合recovery gate联合完整周期、runtime clean、同进程、全部task validator无失配、全部恢复HTTP成功及适用表型恢复。它不强制所有开放题全文相等；**新TargetPlan的最小Qwen验收在此之外要求全文和pieces精确恢复**，保持既有更严格标准。

原 `ROBUST_VISIBLE_DISTORTION` 条件不改：冻结16题、≥3完整周期、每周期≥4异常、≥3同题稳定异常、≥2稳定异常族、0crash、机制和恢复通过。MT-Bench单题最小功能验收标 `NOT_APPLICABLE`，不把它冒充该严格门禁PASS。N=1或3都不声称统计显著性。

## 8. 保留字段逐项审计

| 字段 | 本发布主体实际情况 | 四族×20样例 |
| --- | --- | --- |
| `quality_performance.mean_logprob_delta/max_logprob_delta` | 仅schema保留；未找到本版生产者/计算实现 | 未采集，null |
| `quality_performance.kl_divergence` | 仅schema保留；无完整分布KL测量链路 | 未采集，null |
| `quality_performance.ttft/tpot/throughput` | 仅schema保留；没有流式时钟分解/吞吐计量器 | 未采集，null；请求耗时不冒充TTFT |
| NLL/PPL | 旧owner采集、历史cycle导入和解释已实现；新TargetPlan通用概率事务未实现 | NOT_MEASURED |
| Accuracy/Retention/Paired Damage | 逐题validator和已有配对聚合实现；无expected开放题不计分 | NOT_EVALUATED |
| token ids | API可选择记录；当前实测没有整数IDs | null，实际比较token_pieces |
| phase/prefill/decode、独立step coverage | hook采集缺少可证明的阶段/执行步来源 | UNKNOWN/UNPROVEN |
| 原生DeepSeek效果 | 只读规则存在；尚无具体实例实测 | PENDING_INSTANCE |

本版能够冻结的事实是：Qwen最小功能链路、数值变化、明显重复样例、精确恢复和可追溯的离线报告；不是新强配置的完整任务/PPL性能报告。
