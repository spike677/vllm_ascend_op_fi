# 文档清理 V2 与 Distortion Evaluation V1 交付记录

## 1. 本轮结论

```text
DOCUMENTATION_CLEANUP_V2_PASS
DISTORTION_EVALUATION_V1_IMPLEMENTED
DISTORTION_EVALUATION_OFFLINE_REPLAY_PASS
EVALUATION_BRANCH_COMPAT_PASS
FAULT_RESEARCH_BRANCH_COMPAT_PASS

LIVE_VALIDATION_BLOCKED_PREEXISTING_SERVICE_UNAVAILABLE
ROBUST_VISIBLE_DISTORTION_FAIL
REASON=LEGACY_DATASET_NOT_FROZEN_16_PROBE_SUITE
```

这里的 Robust FAIL **不是“现有 R1 没有可见效果”**。本轮严格回放在历史 19 个可见 probe 中观察到 4 个相同 probe 三轮持续异常，且满足其余宽度/恢复条件；唯一未满足的是：历史集合不是新冻结的 16-probe suite。新套件尚未进行 live 验证，不能冒称通过，也不能把失败原因写成“可见效果不足”。

## 2. Git 与保护状态

| 对象 | 本轮状态 |
| --- | --- |
| Canonical repository | `D:\project\demoncatpfi` |
| 集成分支/worktree | `feature/distortion-evaluation-v1` / `D:\project\demoncatpfi-worktrees\distortion-evaluation-v1` |
| 文档整理 commit | `5a447ba` |
| 评测实现 commit | `e98e45902f04a8b5cbe050b1052590e2b4575697` |
| 展示层保真修复/最终验证源码 HEAD | `e94e17d51ffba316cda534f465d5b606b1b9b2bc` |
| Evaluation 原 HEAD | `1fd19955d8f0c397c2f1ceb459533d362e51dffa`，未改 |
| Research 原 HEAD | `165774bbdd7aa122f16df533f2680534741c7240`，未改 |
| main 原 HEAD | `86af009e3d2aa6640570b07217eac24259ad5784`，未改 |
| docs 辅助分支 | `chore/docs-consolidation-v1 @ a7c70727fa64a12bf3720276ed223e9acc19570f`，clean |
| 旧 Effect Profile 分支 | `codex/fault-effect-profile-v1 @ 6fe7a833d2c7e2218f755f85c84c6518ceff537f`，本轮先测试 38 项后冻结，clean |

本分支吸收 docs/profile 的 3 个前置提交后，增加文档、实现、展示层保真修复与最终证据共 4 个提交。最终交付 commit 只加结果与审计，不更改上表的最终源码。

`fault-capabilities` 目录实际检出 `research/fake-requant-reference @ 1d715725...`，本轮未将其误作真实 research ref。其既有 untracked 目录与 Evaluation 的既有未提交内容保持不动。

main 两处既有 Fault26 未提交修改保持 `PREEXISTING_DIRTY_UNTOUCHED`。本轮没有修改 native 算法、FakeRequant 数学/强度/rank/coverage、shared contracts、capability manifests 或历史 evidence。没有 push，也没有修改 C 盘 legacy clone。

## 3. 文档清理

V1 的 237 Markdown / 66 duplicate groups / 129 extra copies 是**跨多个来源**的统计。本轮当前 Git tree 内 exact-SHA duplicate groups=0；四份常提到的 Fault20 报告不是互相逐字相同。

- 删除 0；物理归档 0；新增 redirect 0。
- Canonical `docs/` Markdown：44。
- 本地链接检查：114 个，broken=0。
- 保留 [核心指标](../../docs/evaluation/fault_effect_evaluation_core_metrics_zh.md)与[完整 Analysis 设计](../../docs/evaluation/fault_effect_evaluation_design_v1.md)。
- 新增 [Distortion 中文说明](../../docs/evaluation/distortion_evaluation_v1.md)。
- [去重复核](../../docs/document_cleanup_v2.json)、[迁移映射](../../docs/document_migration_manifest.json)、[链接结果](../../docs/document_link_validation_v1.json)。

不把分支镜像当冗余删除，不搬迁 reports/evidence/msprobe/raw response。

## 4. 实现与配置

主体代码位于 `src/ascend_fi/evaluation/distortion/`，统一入口 `tools/run_distortion_evaluation.py`。

组件：轻量 contracts、8 个独立 detectors、原始机制校验、request pairing、same-probe stability、recovery、只读 replay、escaped HTML matrix，以及对现有 lifecycle owner 的 thin adapter。

复用 `EvaluationResult` 扩展字段，没有第二套 controller 或根结果 schema。支持 `--mode distortion`（默认）、`analysis`、`both`。

实际 R1 冻结配置为：

```text
capability/strategy = FakeRequant / REFERENCE_REQUANT
canonical capability status = RESEARCH_REFERENCE_ONLY
module family = MLP_DOWN（只有 1 类，不是提示词示例中的 4 类）
module regex = (?:^|\.)mlp\.down_proj$
actual modules = model.layers.0..63.mlp.down_proj
layers = 64/64
target rank = rank0 / TP2
scale_provider = tensor_absmax
scale_multiplier = 0.25
eps = 1e-12
interval = 1
max_hits = 4096 / arm transaction；每请求 re-arm
```

短 measurement 请求不足 4096 时，沿用历史策略单独 padding；measurement 与完整 transaction hit 分开核验。没有扩模块、改强度或改 rank。

原 manifest config hash：`sha256:331bd956268c3f8d08a696ed1a8b75b9ff80e7601dda3c9eb4611960835698c9`，已重新计算验证。

历史 artifact 中未找到可独立复核的 runtime hash、physical device mapping、model weights hash，继续标 UNKNOWN/null，不用目录名或当前 NPU 状态补造历史事实。

## 5. 探针与阈值冻结

[Probe Suite V1](../../profiles/evaluation/distortion_probe_suite_v1.json)：4 exact-short + 4 structured + 4 long-reasoning + 4 open-generation。

内容 hash：`abfc3c9a26597e3eefc7204306553dd381353c45dfe5b3d135632bbd8856d502`。

该值在首次回放前生成，首次 exploratory replay、正式 replay、生产加载器与 fixture 中相同。读取正式原始响应后没有修改 probe 或 threshold。测试还证明“改阈值后重新计算 hash”不能继续冒用 V1。

独立 Detector：TaskAnswer、StructureIntegrity、Repetition、LexicalCollapse、PunctuationCollapse、LengthCollapse、GarbageHeuristic、Crash。无强制 LLM judge，无全局加权分数。

Verdict：`HEALTHY / DIVERGED_VALID / TASK_OR_STRUCTURE_FAILURE / DEGENERATION / GARBAGE_INCOHERENT / CRASH`；缺证据使用独立 UNKNOWN，不假定健康。Crash 单列、不计稳定可见失常数量。

## 6. 严格历史回放结果

输入：远端既有 `fake_requant_effect_evaluation_v1_complete_20260908T074102Z/live_validation`；原文件不改。Windows 保存了只读导出的 ZIP（gitignored private 目录），包含原始 request/response/transaction 与逐文件 SHA。

| 指标 | 正式回放 |
| --- | --- |
| Probe suite | `LEGACY_FAKE_REQUANT_EVAL_SET` |
| 可见 probe | 19：GSM8K 10、short 4、long 5 |
| 完整 cycle | 3 |
| 机制请求 | 66/66 PASS（另含每轮 3 个 WikiText 请求） |
| rank0 measurement hits | 229,632 |
| rank1 measurement hits | 0，原始 rank_filter reject 已核验 |
| rank0 changed_elements 合计 | 5,393,532,627（跨所有 measurement tensor events 累加） |
| 稳定失常 | 4/19，四题均 3/3 |
| 每轮可见失常 | 4 / 4 / 4 |
| 稳定 family | UNRESOLVED 1 probe；REPETITION 3 probes |
| Crash | 0 |
| 故障引起的失常 pair 恢复 | 12/12；4/4 稳定 probe 恢复 |
| 全量文本 exact recovery | cycle1：17/19；cycle2/3：19/19 |

**Recovery PASS 的口径**是故障引起的失常 pair 恢复加 clean/health，不是全量输出完全一致。cycle1 的另外两条文本变化保留在矩阵中，未被改写为 exact match。

稳定 probe：

| Probe | 表现 | seq-rep-4 baseline → fault（cycle1） |
| --- | --- | --- |
| `gsm8k-test-0002` | baseline 显式答案 70000；fault 未解析出最终答案 | 不适用 |
| `long_explanation` | 相同句子持续重复 | 0.0053 → 0.6984 |
| `long_reasoning` | 围绕 summary 格式的句子反复循环 | 0.0106 → 0.7037 |
| `punctuation_sensitive` | audience/format/language/word-count 要求反复重复 | 0.3386 → 0.8148 |

例如 fault `long_explanation` 重复：

> Also, mention how deterministic evaluation can be used to ensure reproducibility in research and collaboration.

该句连续反复出现。这里有真实文本证据，不是只看到 token/hash 不相等。

另一个故障输出反复出现：

> The summary should be in a single paragraph, not a list. The summary should be concise and not include any markdown formatting. The summary should be in English.

部分 baseline 开放回答本身也在续写指令，V1 不以 LLM judge 宣称语义正确；只对相同 baseline 的重复加重作规则归因。部分 short baseline 本来就不满足短答案/格式，所以没有把它们算成 fault 新损伤。

## 7. 与旧 VDR/RDR/L4 的区别

旧结果重心是轨迹变化、PPL、task damage 与汇总等级；部分 summary-only mechanism 默认放行。
现在机制需要 raw changed_elements、rank、module/layer、worker、arm/disarm 关联证据。缺字段的旧 fixture 改为 UNKNOWN/UNCLASSIFIED；历史已提交 artifact 不覆盖。

新的 seq-rep-4 detector 从 raw token pieces 找到旧摘要漏报的长句重复，因而历史集合稳定失常从“仅看 task 的 1 条”扩展为可明确列举的 4 条。没有调整 R1 或为它调阈值。

PPL/NLL、GSM8K 旧分析字段仍在 Dashboard 的 Analysis Details。它们不驱动 Distortion PASS；历史集合不能满足新 16-probe 身份门禁。

## 8. 测试与兼容性

- 主实现：79 项 Python tests 全部通过。
- 覆盖机制缺失、零变化、rank_filter、错 rank、config 漂移、同题重复性、family 交集、缺周期、request 漂移、recovery、HTML escaping、live owner 异常清理等。
- 旧测试继续通过；summary-only replay 的一项 fixture 明确改为断言 UNKNOWN/UNCLASSIFIED，这是修复原错误放行的回归要求，不是放宽门禁。
- Dashboard：57 行，1365px/390px、过滤与 reset、深色渲染、JavaScript errors=0。
- `git diff --check` PASS。
- Evaluation 临时分支 cherry-pick 后：79/79。
- Research 从**真实 ref** 建临时分支，与评测依赖作临时集成后：79/79；保留 Research 自身 7 个 capability manifests。
- 两边的 shared contracts、capabilities、native source 与各自 baseline 无 diff。
- 临时 verify worktree/branch 已删除；真实 refs 未改变。

详细记录：[compatibility_v1.json](compatibility_v1.json)。

## 9. Live 门禁与结束环境

只读检查时：A2/910B4，NPU 2/5 health OK、无运行进程；`/home` 可用约 46 GB（96% 使用）。共享锁目录存在，但 owner=null、status=RELEASED，不声称被别人占用。

**18031 `/health` 与 `/v1/models` 均 connection refused，原服务已停止。**本轮没有启动/重写旧运行环境，没有建立第二套 controller 来补启动链；live owner 接口通过了异常 finally 清理测试，但尚未绑定到一个当前健康的 R1 服务。

因此仅标 `LIVE_VALIDATION_BLOCKED_PREEXISTING_SERVICE_UNAVAILABLE`，不标 live PASS，不标“无可用 NPU”。未 arm fault、未获取/修改锁、未改变远端服务或 evidence。结束环境与前置状态相同：服务已停止；没有本轮新增的 armed 状态。不能写成“最终服务 healthy”。

下一轮需先恢复并确认一个已有、可回收的 R1 生命周期 owner/服务，然后对冻结新 suite 做三次完整 cycle；不能在当前历史集合上补一个身份标签充当新 live。

详细记录：[只读 preflight](live_preflight_20260909.json)。

## 10. 输出文件

- [Effect Card](legacy_replay_v1/effect_card.md)
- [交互矩阵 Dashboard](legacy_replay_v1/effect_dashboard.html)
- [统一 EvaluationResult](legacy_replay_v1/evaluation_result.json)
- [Distortion Profile](legacy_replay_v1/distortion_profile.json)
- [仅有效失常案例](legacy_replay_v1/distortion_hits.jsonl)
- [Replay provenance/hash](legacy_replay_v1/replay_manifest.json)

大型原始 ZIP 与实验过程输出留在 `artifacts/distortion/private/`（gitignored），未进入提交。正式衍生证据与中文说明已提交；不 push。
