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

## 17 计数

没有改 changed 或 changed_elements 的定义。相同事件集必须在 session evidence 和 product review 中得到相同总数。已增加两路径对照测试，并对蓝区真实17原件复核；不同原件先关联 owner/request/image，不能把同名 scope0 当同一次执行。
