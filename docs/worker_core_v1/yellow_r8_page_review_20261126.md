# r8 黄区 11 页审查（2026-11-26）

组合：A3 / Ascend910_9382 / vLLM 0.26.0、vllm-ascend 0.26.0rc1 / CANN 9.1.0 / Qwen3.8 W8A8 / TP2。证据为黄区文本证明，不是蓝区原件审查。原文保存在仓库外，72928 字节，SHA256 `ccea0e2fe53bb1600af044686145d41ce537dd1fea5b3859e73197e3a4dd7632`。逐项 SHA、两个 owner 和四组 rank 计数见同名 JSON。

11 份效果页全部重新通过 `verify`；11 份产品提取段的规范化 SHA 也全部通过。Native 文本中 Markdown 转义的 `https\:` 和 `\_C_ascend` 恢复后，SHA 与原提取段相符；未更改原文。工具 SHA `de2c9bac…`、审查器 `536cf416…`、源码清单 `8568f8a4…` 均与 r8 对应。PAGE_CMP_OK 是黄区本地重提取对照的文字证明；蓝区没有原件，不能声称本地重跑 cmp。

| 能力 | 页面结论 | 登记与补交 |
|---|---|---|
| 28.fia.denominator_replay | 页面通过 | 等负例、逐 rank 完整身份与关联文本 |
| 18.api_half | 页面通过 | 同上 |
| 39.output_proxy | 页面通过 | 同上；本页两个 owner 无 OWNER_CHANGED |
| 39.matmul_range | 页面通过 | 同上；changed 小于 applied 合法 |
| 41.tail_column | 页面通过 | 同上；Native 运行时证明完整 |
| 26.fia_destination | 页面通过 | 同上；Native 运行时证明完整 |
| 20.buffer_substitute | 页面通过 | 同上；Native 运行时证明完整 |
| 17.spatial_padding | 页面通过，跨原件关联需补交 | 元素数差异见下文，暂不能视为整体证据闭合 |
| external.module_input_gain | 页面通过 | 仅受控许可路径，不纳入公共登记 |
| external.dispatch_input_gain | 页面通过 | 仅受控许可路径，不纳入公共登记 |
| external.conv3d_output_gain | 页面通过，视觉身份需补交 | 仅受控许可；核对真实视觉实例与身份文件关联 |

全部满足两个不同 owner、每 rank `eligible=attempted=applied=consumed=restored`、`failed=0`、`changed<=applied`、OFF 精确与三项恢复。Native 三项提取文本均有 `native_runtime_complete=true` 和库 SHA `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。这些是页面层结论，不替代登记所需的 Native 构建元数据。

## 登记门禁

当前 **0 项可登记**。效果页 `registration_ready=true` 只表示该页的部署字段子集齐全，不等于 `product_registration.validate_record` 通过。

- 补全同一精确组合的负例原件摘要：拒绝码、CLI exit、引擎存活、清理零资源、随后正对照；包括通用、Native、视觉及未登记拒绝。连接拒绝不是有效负例。
- 每个 rank 原样提供 `observe_identity` 的完整 `IDENTITY_FIELDS`；不能把单份 runtime-identity 的部署字段复制到另一个 rank。特别包括 Python、EP、源码指纹、映射运行库、implementation SHA、请求范围及 Native 构建身份。
- 补标签、许可 SHA、owner、原件 SHA 的对应关系和耗时记录。耗时不作门禁，image_digest 为 null 不阻止登记。
- 22、15、18.variance_over4 仍缺页面原文，只记“需补交”。over4 的 CLI 映射由黄区处理。
- under4 的 API_TOP5_REQUIRED、Batch 的 OFF_API_NOT_EXACT 仍保留失败，进入独立 r8.2 诊断，不填入 r8.1。

## 17：先关联原件，不能直接改计数

效果页 scope0/rank0 为 55152；产品提取 scope0/rank0 为 55154。两个数字分别来自 effect 原件和 `product_http_17.json`（SHA `453e926d…`），不是同一回执的两个序列化副本。产品提取只给 scope/rank，不能据此认定是效果页里的同一个 group owner。

源码中 `ProductSession.evidence()` 对 disable.requests 的每 request `changed_elements` 求和；`review_product_mode` 保留每 request 的同一字段，`extract_product_record.pages` 再求和。两条路径均包含负例行；没有发现常数 +2、减2或按调用数代替元素数的逻辑。17 的元素数来自真实 Conv3d 输出与同输入正常 padding 的逐元素比较，随图片内容变化，不能从调用数推定。

因此先补这两份原件中的 owner、内部 request ID、prompt SHA、图片像素/输入 SHA、每 request changed_elements。相同事件集必须相等；不同请求或图片允许不同，但必须明确标成两次执行，不能拼成同一组恢复证据。当前不能凭这两段摘要断言公共计数缺陷，也不能把 55154 改为 55152。r8.2 将用同一原件做两路径核对。

外部视觉提取引用 `b630836d…`（文本项也使用），17 引用 `c09d46ff…`。要求黄区核对实际图像范围、processor 配置、进程与身份采集来源；SHA 相同本身不证明运行错实例，但缺关联时不能视为已核实视觉部署。

## Batch 定性边界

异步到达导致内部 batch composition 不同，是有依据的候选原因。串行金丝雀一致不能单独排除 B8 的状态或数值问题。r8.2 记录每轮各 rank 的实际内部请求、prompt 身份、顺序、computed/scheduled token 数；组成不同才有限重发，组成相同后仍必须 API 精确恢复。所有尝试保留，不按输出好坏选择重试。
