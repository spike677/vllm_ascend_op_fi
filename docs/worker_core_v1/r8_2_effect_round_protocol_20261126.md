# r8.2 效果工具候选协议（未冻结）

基线 r8 `e38c4113`。变更限效果工具和隔离验收探针；公共机制、Host/Worker 运行时和登记准入未改。逐文件证明见 `r8_2_runtime_byte_comparison_20261126.json`。r8.1 仍只登记，不包含本改动。

## Batch OFF 对照

Batch adapter 提供 `infer_observed(prompts, sampling) -> {output, trace}`。output 是未经加工的 API 回包；trace 按 rank 排序，包含真实 runner 的每次 execute 对应的请求 ID、batch 顺序、prompt token SHA、prompt 长度、computed/scheduled token 数，以及是否截断、观测方法是否已恢复。不能从 HTTP prompt 数量反推内部批次，不能预填 missing 值。

隔离验收示例在 `experiments/worker_core_v11_20261118/effect_schedule_probe.py`。它只临时包装 execute；原调用完成后读取已存在的 input_batch、scheduler 和 execute_model_state.logits。它不包装产品请求所需的 `_preprocess`/compute_logits 实例入口，不修改调度或张量。所有观测属性必须在下一次 status/disable 前恢复原对象。该示例只用于可信隔离验收服务，不暴露匿名生产 RPC。

1. baseline、fault、每次 off 的原始 API 回包和逐 rank trace 均写入原件的 attempts，含尝试序号。
2. 比较时保留顺序、prompt 身份、computed/scheduled token 数，去除每次重发自然变化的内部请求 ID。各 rank 组成必须一致。
3. off 与 baseline 组成不同，记录 mismatch 并重发 off；默认最多4次，程序接口硬上限8次。不中途改 baseline，不重发 fault，不按数值是否相等挑选结果。
4. 第一次组成相同即停止重发，然后要求 API 精确恢复。组成相同但数值不同仍是 OFF_API_NOT_EXACT；达到预算仍不匹配是 BATCH_COMPOSITION_RETRY_EXHAUSTED，不产生通过页。
5. summarize 会重新检查 attempts、所有 rank trace、首次匹配即停止规则和所选 output；缺失、截断或 owner 恢复未证实都拒绝。
6. fault 的组成也必须完整记录，但故障输出的批次可能与 baseline 不同。因此 API 变化是服务层现象，不单凭它声称全部变化都由共享 scale 导致；真正注入仍需 scale/Q 参考、QMM 消费、覆盖和恢复证据。

A2 用 `--effect-probe --effect-force-split-off` 把首个 off 的八条 prompt 分成两次真实 HTTP 请求，制造到达组合负例，之后恢复通常发送方式。该标志只在隔离验收脚本使用；保留两次 HTTP 的真实返回，不制造调度记录。它不改变产品的故障或恢复行为。实测结果另行记录，CPU 通过不替代 A2。

## under4 诊断

API_TOP5_REQUIRED 现在附 choice、token 位置和实际条目数，仍要求每个位置至少5条、所有 logprob 有限。诊断同时记录对应 execute 的全 logits 非有限元素数、真实 top-k token ID 和值；不能把 API top5 当完整 logits。

先区分：实际 logits 非有限；不同 token ID 解码成相同字串导致字典合并；服务端漏字段；请求关联不足。没有同次原件前不放宽门禁，不把错误强制转成通过。baseline/fault/off 回包及 trace 在校验前落盘，失败后仍保留。

11月26日协商窗口 `r82effects1126a3` 已复现：故障轮 choice0 position0 的 API 字典只有一个空字符串键；两个 rank 在 baseline/fault/off 各12步的完整 logits 均无非有限值。首步 top5 ID 为236、250、104、255、248，对应原始 logits 为21.375、20.75、20.5、19.875、19.75；实际模型 tokenizer 将这五个不完整字节 token 单独解码成同一个替代字符。vLLM completion serving 在构造 top_logprobs 时用解码字符串作字典键，因此不是“实际只算出了一个候选”。原件保留，不将本次失败改写为通过。

修复限效果工具请求：所有轮次显式设置 `return_tokens_as_token_ids=true`，使用该0.26镜像已有的无歧义键格式。temperature、seed、logprobs、生成长度及故障参数均不变；仍要求每位置至少5条且数值有限，仍精确比较OFF。CPU覆盖多token同字串碰撞；修复后的NPU回归必须使用新的源码清单，不能沿用上述失败轮次作为通过证据。不修改模型、tokenizer、采样算法或服务端。

最终源码 `9464df0c` 已在A2同组Worker两个owner完成under4和Batch回归，见 `r8_2_a2_completion_20261126.md`。Batch实际组成不匹配后重发成功，全部原件保留；under4使用token-ID表示后完整五项和恢复门禁均通过。旧失败记录仍为失败，不补改原件。

## 17 计数

没有改 changed 或 changed_elements 的定义。相同事件集必须在 session evidence 和 product review 中得到相同总数。已增加两路径对照测试，并对蓝区真实17原件复核；不同原件先关联 owner/request/image，不能把同名 scope0 当同一次执行。
