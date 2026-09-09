# Distortion Evaluation V1：可见退化优先

## 定位

Primary Mode 是 **DISTORTION**：给定一条已验证的故障配置，是否在正常输入下稳定产生用户可见、可恢复的异常。
Secondary Mode 是 **ANALYSIS**：任务正确率、PPL/NLL、输出轨迹与细粒度统计，用于解释效果，不代替可见异常验收。

本实现只评估已有能力，不改变 FakeRequant 数学、native runtime、rank、模块范围或强度；不新增 controller。它借鉴 Probe → Detector → Hit Log 和 assertion/matrix 的组织方式，没有攻击探针、ASR、LLM 强制裁判或加权总分。

## 两张一级结果卡

| Fault Configuration：注入了多少、在哪里 | Visible Effect：用户实际看到了什么 |
| --- | --- |
| capability / strategy | 稳定失常 probe 数量 X/N |
| 实际模块名与 module families | 稳定 phenotype family 与对应 probe |
| targeted / available layers | 同一道题在全部完整 cycle 中是否异常 |
| targeted ranks / TP | Crash 单列，不视为更强的效果 |
| primitive strength、budget、request window | 故障引起的失常是否恢复 |
| runtime/config hash、model fingerprint | Mechanism Gate 不通过的 pair 不参与归因 |

两者不能压缩成“Fault Score=0.73”。剂量不等于严重度。未知 hash、physical device mapping 或模型权重 fingerprint 必须保留 UNKNOWN/null。

## 数据与实现入口

- [冻结的 16 条正常探针与阈值](../../profiles/evaluation/distortion_probe_suite_v1.json)
- [轻量 contracts](../../src/ascend_fi/evaluation/distortion/contracts.py)：`DistortionProbe`、`DetectorResult`、`FaultConfigurationRecord`。
- [原始机制门禁](../../src/ascend_fi/evaluation/distortion/mechanism.py)
- [独立 detectors](../../src/ascend_fi/evaluation/distortion/detectors.py)
- [配对、周期、稳定性 evaluator](../../src/ascend_fi/evaluation/distortion/evaluator.py)
- [只读 replay](../../src/ascend_fi/evaluation/distortion/replay.py)、[独立 HTML/JSON 输出](../../src/ascend_fi/evaluation/distortion/report.py)
- [CLI](../../tools/run_distortion_evaluation.py)、[已有生命周期适配入口](../../src/ascend_fi/evaluation/distortion/live.py)

仍使用现有 `EvaluationResult`，通过其 `extra` 的扁平扩展字段保存 `fault_configuration`、`distortion`、`analysis_details`，没有建立第二套结果根 schema。

`DistortionProbe` 字段：`probe_id/category/prompt/generation_config/expected/applicable_detectors`。配置由 4 条 exact-short、4 条 structured、4 条 long-reasoning、4 条 open-generation 组成。

冻结 hash：`abfc3c9a26597e3eefc7204306553dd381353c45dfe5b3d135632bbd8856d502`。
生产加载器及 fixture 都核对该 hash。改内容并重算 hash 也不能继续冒用 V1；后续调整必须用新版本，不能为了 R1 PASS 调参。

## Mechanism 是有效性 Gate

从 `fault_transaction` 的原始 arm / measurement_query / transaction_query / disarm 验证：

1. 由 `matched_modules.tp_rank` 识别 rank，而非结果数组顺序；模块、layer、TP 与配置一致。
2. 目标 rank 的实际 measurement hits > 0；逐条 `tensor_effect_history` 的 hit_index、target、layer、rank、strength 和 `changed_elements` 有效。
3. 非目标 rank hit=0，并有 `rank_filter` 原始 reject 证据。
4. arm、measurement、disarm、完整 transaction 关联到同一组 worker pid；request generation 一致。
5. 请求 HTTP、服务健康、worker query 响应与清理证据完整。

缺字段为 UNKNOWN、明确矛盾为 INVALID；两者都归类 `INVALID_FAULT_RUN`，排除 Effect 归因。旧 summary 中的 hit_count 不能填补 changed_elements 或 rank 缺口。

历史短请求可能未耗尽 4096 hit budget，之后单独 padding 才耗尽。必须分开保存 **measurement hit** 与 **transaction budget**；padding 的数值变化不归到被评分文本。

## Detector 与证据边界

| Detector | 规则/记录 | 范围与限制 |
| --- | --- | --- |
| TaskAnswer | 首个行首 FINAL_ANSWER → 首个 #### → 首个明确 answer/result expression；固定答案比较 | baseline 正确才允许记 damage；区分错答与未解析；协议是否自然停止不影响答案判定 |
| StructureIntegrity | 数列缺项、重复、顺序、逐行完整性；列表/表格；JSON 必填字段、类型、额外字段 | 仅对冻结 expected structure 使用；baseline 已失败不能归因 fault |
| Repetition | seq-rep-4、最长同 token run、周期 1–8 重复 ≥3、dominant ratio | 要求 baseline-relative；最少 16 token；seq-rep-4≥0.30 且提升≥0.20，或新出现 run≥6/周期重复 |
| LexicalCollapse | Distinct-1/2、dominance | 两侧≥32 token；Distinct-1/2 同时降到 baseline 的 0.5 以下且 dominance 增≥0.15 |
| PunctuationCollapse | 标点比例、最长连续标点 | 比例≥0.35 且增≥0.20，或连续≥8 且增≥4 |
| LengthCollapse | 长度比、finish_reason、是否达到 max_tokens | baseline≥32，fault≤0.35×baseline 为早停候选；fault≥64 且≥1.5×baseline 并达上限为 runaway |
| GarbageHeuristic | replacement/control/surrogate 字符及位置 | 新出现至少 2 个异常字符；不把正常中文/罕见语言当乱码；不宣称语义 incoherence/topicality |
| Crash | HTTP/error、worker/service 失败 | 独立 CRASH，不计可见失常数量；任何 crash 均阻止 Robust PASS |

V1 的 early-stop=0.35、period=1–8 与历史 Analysis 使用的 0.5、period=1–16 是两个明确版本；历史阈值保留，不回写旧证据。

各 detector 返回 `triggered`（true/false/null）、baseline/fault value、severity、reason、evidence、confidence/source。无整数 token IDs 时采用 token pieces；两侧依据不同则 UNKNOWN，不冒充相同 token 序列。

## Verdict 与稳定性

优先级：`CRASH > GARBAGE_INCOHERENT > DEGENERATION > TASK_OR_STRUCTURE_FAILURE > DIVERGED_VALID > HEALTHY`。

UNKNOWN 使用独立 observation 状态与空 verdict；baseline 已不符合任务协议时也不能标“新的故障”。`DIVERGED_VALID` 仅表示已配置检测器没有发现新的失败，不是语义正确性的证明。

Stable Distorted Probe 要求同一个 probe 在至少 3 个**完整** cycle 中全部 visibly distorted。每轮坏不同题，即使数量相等，也不稳定。Phenotype breadth 对每个稳定 probe 求跨轮 family 交集后再统计，不把跨轮切换的表现混成稳定。

Robust V1 Gate 全部满足才 PASS：

- Mechanism PASS、clean baseline 与 request pairing PASS。
- 真正冻结的 16-probe V1，不是 legacy 数据伪装；至少 3 完整周期。
- 每轮至少 4 个可见失常 probe。
- 至少 3 个相同 probe 在全部周期均失常；至少 2 个稳定 phenotype family。
- Crash=0；Recovery PASS。

PPL 上升、RDR 高、旧 L4 分类都不能单独驱动 PASS。R1 FAIL 是允许且有价值的结果。本轮不做 coverage/strength search。

## Recovery

对 fault 引起的失常 pair，要求同一请求的 recovery 回到 baseline 等价或已配置检测器的健康范围；还要求 hook removed、disabled、worker/service 健康。短任务精确文本改变但答案仍对可以恢复。

未发生 fault-induced distortion 的 probe 不充当恢复计数分母。baseline 原本不稳定或错误不变成故障损伤；全量 exact recovery 与本门禁是不同字段，不能混称。

## 用法与交付

```powershell
python tools/run_distortion_evaluation.py --mode both `
  --replay <existing_result_directory_or_readonly_export_zip> `
  --output <new_independent_output_directory>
```

默认 mode=distortion；analysis/both 保留二级细节。输出目录不得覆盖原始 evidence 或已生成结果。

输出：`evaluation_result.json`、`distortion_profile.json`、`effect_card.json/md`、`effect_dashboard.html`、`distortion_hits.jsonl`、`replay_manifest.json`。Hits 日志只含机制与配对有效的可见异常 pair。Dashboard 顶部两卡，下面是 probe×baseline/fault/recovery 矩阵，支持失常/重复/结构/乱码/未解析/crash/恢复失败筛选；模型输出经 HTML 转义，无网络请求或控制接口。

Live 入口接受 `--live-adapter existing_module:factory`。它只调用已有 owner 的 preflight、锁、request、fault_request、install、disarm_and_remove 等接口；不自行实现 RPC 或 NPU lock。owner 必须保持已有逐请求 try/finally、原始 evidence 落盘和退出时服务恢复。没有已验证的 adapter/锁/资源，明确 BLOCKED，不自动临时造 controller。

## 历史回放

旧三轮数据标为 `LEGACY_FAKE_REQUANT_EVAL_SET`，包含 10 GSM8K + 4 short + 5 long 可见 probe，另有 3 WikiText 机制/概率请求。不能称其为新 16 条探针的 live validation。

历史原始 ZIP/JSONL 只读，保留来源路径、文件 SHA 与缺字段状态。旧 Effect Profile 的 summary-only replay 已改为 UNKNOWN/UNCLASSIFIED，不再把缺数值证据的汇总当已验证 L3/L4。旧已提交 artifact 仍保留，属于历史分析，不是本轮的权威 Mechanism 结论。

更多维度及公式见[核心指标说明](fault_effect_evaluation_core_metrics_zh.md)与[完整 Analysis 工程设计](fault_effect_evaluation_design_v1.md)。
