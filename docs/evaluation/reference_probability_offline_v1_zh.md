# 可选 D：固定参考 token 概率离线适配 V1

状态：`IMPLEMENTED_OFFLINE / LIVE_UNVERIFIED`。本包没有启动模型，没有向服务发送请求，也没有实现新的 TargetPlan 概率事务。它不阻塞选点、数值、检测器和单次使用流程收束。

## 1. 先核对旧能力

现有 [live.py](../../src/ascend_fi/evaluation/live.py) 的 `prompt_loglikelihood()` 已会调用 `/v1/completions`：`echo=true`、`max_tokens=1`、固定 prompt 和 logprobs。它按 `usage.prompt_tokens` 截掉生成部分，但其返回投影只有 `token_logprobs/prompt_tokens/http_status/error`，没有完整回传 token IDs、原请求、tokenizer、mask 和窗口身份。

现有 [suite.py](../../src/ascend_fi/evaluation/suite.py) 的 `_run_ppl()` 过滤数值 logprob，并核对总计数；`word_perplexity()` 按 **word_count** 归一化。旧报告有其历史口径，本次不更改这些函数或重算旧证据。

因此旧 API 有采集基础，但旧投影不能自动证明新的严格 token 对齐。不能将已有 189-token、旧 QMM 或旧 R1 概率结果挪到新配置。

## 2. 新增的是小型离线适配器

代码：[reference_probability.py](../../src/ascend_fi/evaluation/reference_probability.py)。复用原 OpenAI-compatible echo 响应形态以及原 `EvaluationResult.extra`，没有第二套结果类、控制器或数据下载器。

```python
from ascend_fi.evaluation.reference_probability import (
    parse_echo_reference,
    compare_echo_reference,
    attach_reference_probability,
)

one_window = parse_echo_reference(reference, recorded_echo)
comparison = compare_echo_reference(reference, baseline_echo, fault_echo)
updated_result = attach_reference_probability(
    existing_result, reference, baseline_echo, fault_echo,
)
```

`reference` 必须含：完整 prefix+reference 的整数 `token_ids`、等长布尔 `score_mask`、`model_identity`、`tokenizer_identity`、`template_identity`、`reference_sha256`、`window_identity`、`context_policy`、`logprob_semantics`、`cache_policy`。单窗口第一 token 没有前文，不能在本合同中设为 scored。

每份 recorded echo 必须保留原 `request`、`raw_response`、`http_status`、独立 `request_id` 和采集时的相同 reference 身份。原 request 以 token IDs 作为 prompt 且 echo 为 true。API 回传的 prompt token IDs 必须匹配，prompt token count、mask、logprob 位置都必须完整。只给 token pieces 或旧压缩投影时，本版明确不可评分，不猜整数 ID。

只评分 mask=true 的 prompt 位置，忽略最后生成 token 的 logprob。有限 top-k 中没有参考 token，不能推导其概率为零。`None` 在已评分位置会让整个窗口 `UNSCORABLE`，不会静默少算 token 得到更好看的 PPL。

## 3. 指标与边界

```text
T = score_mask 中 true 的个数
sum_NLL  = -sum(reference_token_logprobs)
mean_NLL = sum_NLL / T
token_PPL = exp(mean_NLL)
mean_NLL_delta = fault_mean_NLL - baseline_mean_NLL
PPL_ratio = exp(mean_NLL_delta)
```

PPL 是参考序列平均负对数似然的指数；tokenizer 与上下文窗口影响其口径，比较时必须固定。这里输出 token PPL，不把旧 word PPL 更名为 token PPL。[Hugging Face 固定长度模型 PPL 文档](https://huggingface.co/docs/transformers/v4.57.1/en/perplexity)

所有非有限结果均通过 null+状态序列化：

- 合法 `logprob=-Inf`：对应 NLL/PPL 正无穷；不当作缺失概率。
- NaN、正 logprob、+Inf、非数值：不可评分。
- T=0、BOS mask 错误、回传 IDs 错位或缺失：不可评分。
- PPL `exp()` 溢出：值 null、`OVERFLOW`，不截成最大有限值。
- B/F 的 PPL 都溢出但 mean NLL 有限时：在 log 域求差，ratio 仍可能精确可表示。
- B/F 的 mean NLL 都是真正正无穷时：`Inf-Inf` 和 `Inf/Inf` 不定义，不能报告 ratio=1。
- B/F 原请求不同，或复用相同 request ID / response cache key：拒绝配对。

本版只处理一个明确窗口。多窗口 sliding/overlap 的去重采集尚未扩展；不能把同一参考位置在多个窗口重复当作更多独立 token。没有全词表 dump、KL/JS 或 Top-1 翻转；后者显式 `ARGMAX_EVIDENCE_NOT_IMPLEMENTED`。

## 4. 不将数学解析当成注入实测

即使离线数值完整，`probability_transaction_gate` 仍是 `NOT_IMPLEMENTED`；`model_effect_attribution` 仍要求 **这条 echo 请求自己的** TargetPlan、同 epoch 绑定、IDENTITY、实际 hit/数值变化和实例结束证据。聊天请求的 hit 不足以替代 teacher-forced 请求，因为执行形态可能改变。

`attach_reference_probability()` 只增加 `EvaluationResult.extra.reference_probability_offline`，不改主 `status`、既有 `quality_performance` 或严重度门禁，也不覆盖已有同名扩展。调用者提供的 identity/cache-policy 元数据不是本工具独立检测到的线上事实。

后续现场适配应复用旧 completion client，但需要保留完整 echo 响应与对齐依据、接上当前 owner 的一次有损后停止流程，并验证当前 API 能给出所需参考 token IDs/logprobs。本轮未进行这些操作，不要求为 D 放宽单次请求或请求隔离约束。

## 5. 已运行的离线验收

14 项标准库 CPU 单测通过：固定 ID/mask 配对、只计参考而非生成 token、缺失不可丢弃、top-k 缺失、非有限值、BOS/T=0、sum/PPL 溢出、log 域 ratio、身份漂移、缓存复用、旧投影拒绝与 `EvaluationResult` 不被升级。

```bash
python -m unittest tests.evaluation.test_reference_probability -v
```

测试数据均为显式 fixture，不是新模型概率实测。没有依赖安装、网络采集、NPU 使用或历史证据修改。
