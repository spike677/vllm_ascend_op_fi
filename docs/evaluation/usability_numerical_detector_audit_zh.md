# 数值派生与检测器离线校验 V1

本页是评价扩展的接口与验收说明，不是新的故障算法。FakeRequant 输入变换、原始 `EffectStats`、检测器阈值和旧结果均不修改。新增内容可放入既有 `EvaluationResult` 的扩展字段；不另建效果评分体系。

## 1. 数值项来自同一采样能量

入口：[numerical_effect.py](../../src/ascend_fi/evaluation/numerical_effect.py)。输入是同一个请求、每个选中 worker 的一份最终 `effect_summary`。不能把同一累计 query 重复汇总，也不能跨 request generation 拼接。

```python
from ascend_fi.evaluation.numerical_effect import sampled_numerical_effect

numeric = sampled_numerical_effect(
    worker_queries,
    expected_ranks=[4, 9],       # 明确 global rank，不是物理卡号
    request_generation=12,
)
```

每份 query 需要显式 `rank`、`effect_summary.schema=BOUNDED_EFFECT_SUMMARY_V1`、连续性与错误信息，以及 `per_module`。模块行的 `rank/module` 必须与外层身份一致。相同模块名在不同 rank 分开保留；漏 rank、重复 rank、损坏的统计不能算作零剂量。

设同一采样集合的输入平方和为 S、误差平方和为 E：

```text
relative_L2 = sqrt(E/S)
NMSE = E/S
SQNR_dB = 10*log10(S/E)
```

这是一个信息的三种表达，不是三份独立退化证据，不改变病态判定。PyTorch 2.9.0 Numeric Suite 的 normalized L2 与 SQNR 采用相应范数公式；本轮只按该固定版本核对定义，不依赖其当前发行状态。[官方源码](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/ao/ns/fx/utils.py#L408)

精度与汇总约束：

- 输入使用未舍入的 `sampled_error_sq_sum`、`sampled_input_sq_sum`，不是 Dashboard 中的小数 L2。
- 源采集保持 FP32 差值与每调用平方和、host float 累计；旧 reference 绝对误差仍是原 tensor dtype 的合同。
- 先加所有目标的 E/S，再推导总体指标。不能平均各模块 dB、NMSE 或 L2。
- 采样保持每模块第 1、65、129…次调用，arm 后重新计数；不是全量 tensor 测量。
- 分母是 rank-local activation 元素事件，不是唯一参数、唯一内存元素或独立统计样本。
- 跨模块总和使用 `math.fsum`；SQNR 在 log 域相减，避免 E/S 先溢出。合法零值与非有限状态分开。

| 情形 | 数值与状态 |
| --- | --- |
| S>0、E>0 | `MEASURED`，有限可表示的指标 |
| S>0、E=0 | NMSE/L2 为 0；SQNR 为 null，`POSITIVE_INFINITY` |
| S=0、E=0 | `ZERO_REFERENCE_ENERGY`，`UNDEFINED_0_OVER_0`；不能填正常零 |
| S=0、E>0 | 同上；NMSE/L2 正无穷、SQNR 负无穷用 null+状态表达 |
| 无采样 | `NO_SAMPLES`；采样数量与能量矛盾则 `INCONSISTENT_COUNTS` |
| 缺字段/只有旧展示 L2 | `MISSING`，总体 `INCOMPLETE_INPUT`，不反推 |
| 上游 NaN/Inf | `NONFINITE_UPSTREAM`，原非法值不进入输出 JSON |
| 比例溢出/下溢 | 对应值 null+`OVERFLOW/UNDERFLOW`，可表示的 log SQNR 仍保留 |
| 模块统计坏或必要 rank 缺失 | 总体 `INCOMPLETE_INPUT`，可读子项保留但不冒充完整总体 |

所有结果可以 `json.dumps(..., allow_nan=False)` 序列化。没有为指标追加 tensor dump、hook 或额外模型请求。

## 2. 完整原文检测器审计

入口：[detector_audit.py](../../src/ascend_fi/evaluation/detector_audit.py)、[CLI](../../tools/audit_detector_quality.py)。直接复用冻结 `compare()` 和 detector；不另写检测规则，不调整阈值。suite 文件由既有 `load_suite()` 校验 hash。

在仓库根目录用已有 Python 离线执行：

```bash
python tools/audit_detector_quality.py \
  --cases tests/evaluation/detector_audit_cases_v1.json \
  --out /your/new-output/detector_regression.json
```

`--out` 必须为新文件，父目录需已存在；不会覆盖历史报告。Windows 编辑端也能运行这条离线工具，但这不表示模型 runtime 已支持 Windows。

输入 `cases` 每条包含：

- `case_id`、`case_family`、原 `probe.prompt`、`cycle_id`、`protocol`、`split`。
- `source_kind=synthetic_regression` 或 `recorded_model_output`。
- 完整 `baseline/response.response_text`、`text_is_complete=true`。这里完整是指完整记录的响应，仍可能 `finish_reason=length`，并不要求自然停止。
- 实际原文 hash/source_ref；真实记录必须有 source_ref。token pieces 可缺失，但不得伪造 token IDs。
- 按 detector 分开的标签、baseline eligibility、标签来源、审核状态、审核方法、证据说明/区间。

新模块只输出 JSON，不自行渲染 raw HTML。既有 Dashboard 的 HTML 转义不变；不能直接将完整原文拼为未转义 HTML。

### 回归与质量分开

| 标签 | 允许用途 |
| --- | --- |
| `fixture_spec`，合成确定性文本 | 回归断言；不是实测或人工金标准 |
| `historical_audit_expectation`，历史审计结论 | 旧用例完整原文回归；不是独立复核 |
| `agent_candidate`、未确认、来源不完整 | 待复核；不能进入 TP/FP/FN/TN |
| `independent_human_review`、confirmed、有 reviewer/method/evidence_note、independent=true、该 detector baseline_eligible=true | 进入声明的可信标签子集；工具不能认证审核人的真实身份 |

每种 detector/协议分别报告 TP、FP、FN、TN、precision、recall、FPR 及分母。分母为 0 的比例为 null；不产生漂亮的空集 100%。`split=check` 且 `not_used_for_tuning=true` 才进入独立检查子集；仍不宣称泛化认证。

同一原 prompt 或同一 case family 不得跨 development/check 等集合；更改 cycle、case 名称或 completion/chat 标签不能规避检查。同 prompt 的三周期同一完整 B/F 内容在质量统计中去重，仍保留全部原文和周期记录。协议分层统计，另报告 family 数，不能把周期数当独立问题数。

### 异常区间是有界证据，不是完整病态占比

区间零基、右开。字符单位为 Python Unicode code point，token 单位仅在原证据确为 token pieces 时提供。重叠与相接区间先做并集，不重复计算。

当前 detector 只保留至多 12 个重复片段位置、有限异常字符位置或最长标点段。因此本次 `locations` 明确为 `DETECTOR_RETAINED_EVIDENCE_LOWER_BOUND`，不是整段文本全部异常覆盖。`first_observed_offset` 是保留证据中最早位置，不保证实际第一次异常；没有可靠对齐时不猜字符位置或整数 token ID。

## 3. 历史完整输出的特别回归

旧 `case_explanation_audit.json` 已包含 0001/0002/0004 的完整文本、字符数和 SHA256。本工具导入前全部复验，拒绝仅 600 字符摘录。

```bash
python tools/audit_detector_quality.py \
  --historical-explanation-audit artifacts/distortion/legacy_replay_scoring_v2_final/case_explanation_audit.json \
  --raw-source /your/existing/legacy_raw.zip \
  --out /your/new-output/legacy_detector_audit.json
```

`--raw-source` 只读取既有 ZIP/目录，由原 `RawSource` 校验成员 hash，不解压覆盖。它补回原 prompt 和 token pieces，且核对 ZIP 文本与审计全文相同。没有 ZIP 时仍可只读文本，但原 prompt 缺失时按原 sample key 分组并明确标记；不会新造 prompt。

这一边界在本轮实际暴露：仅全文、没有原 token pieces 的 lexical fallback，对 0004 的三次尾循环未触发；9 个旧断言只有 6 个一致。补回原 ZIP 的 token pieces 后，9/9 一致。阈值与检测实现没有改变。不能将两种 token basis 的检查结果无说明地混合，也不能只复制旧标签就声称回归通过。

完整 ZIP 回归保留：0001/0004 是重复候选；0002 预算内未交付答案不单独证明病态。旧 completion 输出不是新 chat 或 Qwen0.6B 的认证，旧标签仍是历史回归期待，独立质量验证状态保持 `INDEPENDENT_LABELS_PENDING`。

## 4. 本轮验收与范围

新增单测覆盖公式、零/缺失/非有限、聚合次序、不同 rank 同名模块、旧报告兼容、合法 JSON/表格/列表/代码、答案后循环、尾循环、普通改写、预算无答案、baseline 也重复、标点/字符、完整原文、并集区间、标签门禁、分组泄漏、质量分母、旧文本缺 token basis 的真实差异。

本机 Python 未安装 PyTorch：解析公式与标准库 CPU 测试实际运行；与 CPU torch 张量的直接测试显式 skip，不下载新依赖、不冒称张量数值实测通过。已核对官方固定版本定义；若将来运行环境已有 torch，同一测试会自动执行。

本轮不占 NPU，不改 reference、模型权重、阈值或历史证据。没有独立人工标签集，检测器泛化质量尚未验收；没有新的多 rank 现场数值数据，不借历史 rank0 注入报告声明跨 rank 实测成功。
