# Distortion 评分解释修正与完整原文审计 V2

日期：2026-09-09。本轮是离线工具修正与历史原始数据重放，不是新一轮模型实验。

## 三项独立结论

1. **评价工具：本次明确缺陷的回归通过。** 91 项测试通过，Dashboard 桌面/手机宽度检查通过；不能据此宣称所有语义异常都能自动识别。
2. **R1 可见效果：旧 completion 协议下有真实重复证据。** 全部 19 个用例、3 个完整周期重新评分后，5 个相同用例在每个周期均触发重复，共 15 个配对；不是手工修改旧计数。
3. **正常聊天协议下的稳定交付用例：仍待当前 V2 live 完整周期验证。** 旧结果不能替代新协议，局部表型恢复也不能替代全测试集恢复。

## 修改边界与版本

- 独立分支：`codex/distortion-scoring-audit-v2`，从 `557d6aa08e0564b64999a5b47e7e0ec76e01a27e` 创建。
- 不修改 FakeRequant、Fault20 native、R1 强度、prompt、生成参数、阈值或运行中的采集脚本。
- 保留现有 `EvaluationResult`；新增解释字段，不新建一套评分模式。
- Scorer：`DISTORTION_SCORER_V2_TASK_TERMINATION_PATHOLOGY`。
- Scorer 源码集合 SHA256：`69a2ae6346809dcb38bab813bdad3b0c4215617dfd3c7d50ffabae321cc3ff4e`。单文件哈希见结果中的 `scorer.source_files`；重放 manifest 另记录全部评分模块哈希。
- 原始 ZIP SHA256：`34d462531bfd1d24ce559c6920288b5fb7435f8abf76cd5977af626c29c14dd7`。
- 旧报告、ZIP 和中间报告未覆盖。新报告见[完整 Dashboard](../../artifacts/distortion/legacy_replay_scoring_v2_final/effect_dashboard.html)、[评分结果](../../artifacts/distortion/legacy_replay_scoring_v2_final/evaluation_result.json)、[逐题原文审计](../../artifacts/distortion/legacy_replay_scoring_v2_final/case_explanation_audit.json)。

## 解释规则修正

任务答案 validator 仍由任务意图决定；重复、低词汇多样性和标点异常的适用范围改为查看实际输出长度/形态，而不是仅限 `category=long`。原阈值与合法 JSON/列表等结构保护保留。

旧校准中的 `NOT_APPLICABLE` 可能只是旧类别路由未运行该检测器，不等于已经证明该检测器不适用。新评分对新增通用检测路由单独记录 provenance；不篡改原校准，也不把它声称为已经执行过的新规则在线校准。

| 独立判断 | 含义 |
| --- | --- |
| Task outcome | 正确、错误、无法解析，或达到预算仍未交付答案 |
| Generation termination | 原请求的 stop/length、token 数与预算 |
| Output pathology | 独立检测重复、刷屏等；仅 unresolved 不计作明显病态 |
| Mechanism attribution | 请求级 target、hit、数值变化、rank 和清理证据 |

答案正确不能短路重复检测；未交付答案不能自动证明数学算错或语义错乱。内部兼容字段 `HEALTHY`/`DIVERGED_VALID` 不作为界面的语义健康结论；界面明确标注已覆盖规则的结果及语义未验证边界。

## 全文复核结果

答案和片段位置均为完整 Python Unicode 字符串的零基、右开区间；token-piece 位置不冒充整数 token ID。

| 用例 | 完整原文证据 | 修正后的解释 |
| --- | --- | --- |
| `gsm8k-test-0002` | baseline 1425 字符、512 tokens、`length`，答案 `70000` 在 `[680,685)`；fault 1429 字符、512 tokens、`length`，无最终答案 | 固定预算内未交付答案；独立通用病态规则未触发，不计为已证实明显扭曲 |
| `gsm8k-test-0001` | fault 1196 字符、512 tokens，先答对 `3`；随后 `#### 3\n\n` 连续重复 92 次，首个所记录重复块在 `[455,463)` | 答案正确，同时存在机械重复；原类别路由漏检已修复 |
| `gsm8k-test-0004` | fault 1742 字符、512 tokens，先答对 `20`；末尾 `#### 20\n` 连续重复 10 次，起于 `[1662,1670)` | 全量重放新增发现的漏检；600 字符摘录看不到这段尾部循环 |
| `gsm8k-test-0008` | cycle 1 baseline 答对 `45`，recovery 达到长度预算但无法提取答案 | 全用例任务恢复不通过；不能被其他失常用例的局部恢复分数掩盖 |

`0002` 的完整 fault 尾部实际是 `... $80,000 + $50,000 = $130,`，不是 Dashboard 第 600 字符的展示停止点。没有进一步运行更长预算，因而不推断继续生成后一定正确或一定错误。

稳定重复用例为：`gsm8k-test-0001`、`gsm8k-test-0004`、`long_explanation`、`long_reasoning`、`punctuation_sensitive`。其中 `0001` 同时触发词汇集中和标点异常，是同一循环的相关表征，不算三个独立根因。三个旧长输出用例保留 completion 续写协议限制，不能包装成全部健康聊天回答被破坏。

所有 171 条 B/F/R 完整响应均校验全文哈希。请求级机制仍为 66/66 通过（含 WikiText 辅助测量请求），measurement hit 合计 229632，changed-elements 事件累计 5393532627；后者不是唯一内存元素数量。

## 恢复必须明确分母

| 恢复范围 | cycle 1 | cycle 2 | cycle 3 |
| --- | --- | --- | --- |
| 已识别失常用例的表型恢复 | 5/5 | 5/5 | 5/5 |
| 全文本精确恢复 | 17/19 | 19/19 | 19/19 |
| 已配置任务/结构 validator 状态恢复 | 13/14 | 14/14 | 14/14 |
| runtime clean/health、同一 worker | PASS | PASS | PASS |

其余 5 个开放长输出没有任务正确性 validator，记录为不适用，不当成语义恢复 PASS。全量恢复 Gate 因 cycle 1 的 `0008` 失败；局部 15/15 仍成立，但不能替代该 Gate。

GSM8K 同轮 B/F/R 分别为 `10/10 → 9/10 → 9/10`、`9/10 → 9/10 → 9/10`、`9/10 → 9/10 → 9/10`。后两轮 `0002` 配对损伤与 `0008` 配对改善抵消总分，不能报告为没有配对损伤；损伤类型是未交付答案，不直接叫数学错误。

## Dashboard 与验证

Dashboard 同时显示展示摘录是否截短、完整字符数、实际 token 数与预算、finish reason、答案片段及位置。可以展开完整响应、重复片段与位置；无法可靠映射字符位置时保留 token-piece 证据，不猜测地址。完整原文保留 HTML 转义。

- 全部离线测试：91/91 PASS，包括预算无答案、正确答案后循环、合法重复 JSON/列表、答案位于摘录之外、全量恢复异常与完整周期读取。
- 浏览器：1365/390 两种宽度，无页面错误/水平溢出，57 行矩阵筛选通过。
- 同一原始 ZIP 再次运行两次评分：结果均与最终 `evaluation_result.json` 完全一致。
- 评分工具不占 NPU；旧数据与冻结故障配置不变。

## 正在运行的 V2：独立证据，禁止热修改

本次只读检查：两次 16 用例 clean 校准完成；IDENTITY 响应精确一致、512 次 identity hit、changed elements=0、移除后 clean。cycle 1 baseline 已完成 11/16 请求；尚无完整 `cycle.json`，不能报告完成一个 cycle。

校准中 `structure_json` 的结构检查两次均 `BASELINE_FAILED`，应保留该不合格范围，不能把之后同类失败全部归因给故障。其他适用规则的校准结果见原始 `calibration.json`，不代表开放回答语义全对。

已审计采集流程：本次解释修正不控制后续 arm、选样或重试；远端继续用冻结版本 `557d6aa`，未热修改。已有后续检查任务在完整周期出现后只读同步检查点，用本分支重新评分并标明新旧 scorer/hash；没有完整周期就只报告进度。

继续步骤：完整 cycle → baseline eligibility / 请求级命中 / 异常全文 / 四种恢复范围摘要 → 三个完整周期后判断同一正常输入是否稳定复现。当前不能宣布 `ROBUST_VISIBLE_DISTORTION_PASS` 或正常聊天用例库已经验收。
