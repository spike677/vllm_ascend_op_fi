# RC3 预览 r3 交付补充单

r3 可以交负责人转发。范围是六项公共持续能力：28 denominator torch、Batch torch、39 output_proxy、18、22、39 matmul_range。33 因完整模型传播未达标不纳入；这不是全部 12 项交付完成，也不是 A3 已验收。未向黄区发送内容。r2 标签和全部发行字节保持不变。

## 包身份

- 标签：`demoncatpfi-rc3-preview-20261122-r3`。
- 源码提交：`112376eeb97f8fd6523851496424814aab45209a`，源包清单 1014 个文件。
- 发行提交：`0a10cb2e7aef838402446bb50dd107718183e82e`；标签固定指向该提交。本交付补充单为冻结后的文档提交，不改包。
- 发行目录：`releases/demoncatpfi-rc3-preview-20261122-r3/`；含源码 tar.gz、三个 wheel、source-review.json、release-manifest.json、SHA256SUMS 和安装回执。
- wheel 版本：demoncatpfi `0.4.0rc2`；demoncat-worker-extension / demoncat-fault39-core 均 `0.3.0rc2`。
- 许可的 `source_manifest_sha256` 填 **source-review.json 原始字节 SHA256**：`09abc833731780cf72db312443ef9512cf9cfd628430c72ef5b70247ec5e16fb`。不是 release-manifest.json 的 SHA。
- 源码包 SHA256：`926b90ddfea82bd2f57aac98d0668ef381d8e3a325a8d4e0b0138961a38c41dd`。其他文件逐项核对 SHA256SUMS。

124 的无设备隔离容器安装通过：358 个已安装 Python/JSON 文件与源清单字节一致，无 Worker、无 NPU 初始化。镜像及 UTC 时间在 install-execution.json；无设备安装不能代替推理验收。Git 对象中的所有发行 SHA 另已复核，见 `docs/worker_core_v1/rc3_r3_git_byte_review_20261122.json`。

## 验收步骤与部署差异

执行完整步骤：[r3 A3 TP1/TP2 验收](../docs/worker_core_v1/rc3_preview_r3_acceptance_20261122.md)。只使用现役模型和镜像的专用实例，不接对外流量。现役对外服务原样图模式、异步调度、MTP 不在支持范围内。

|项目|TP1|TP2|
|---|---|---|
|环境|A3 / 0.26 / 现役 Qwen3.8 W8A8，单卡|同一目标部署，两卡|
|执行|eager、mp executor、串行、异步调度关闭|相同|
|缓存和旁路|prefix cache 关闭、无 MTP、无 KV connector|相同|
|输入|纯文本 image=0、video=0；seed42；4 token|相同|
|prefill|chunked 开启、max_num_batched_tokens=1024、max_model_len=1024；实际证明单请求单 chunk|相同|
|并发|Batch max_num_seqs=8；其他能力1|相同|
|验收|每能力同组 Worker 两个作用域，独立记录 TP1|每能力同组 Worker 两个作用域，独立记录 TP2|

可信宿主安装 `install_continuous_binding`，组客户端 `MultiRankTaskClient(..., continuous=True, world_size=实际TP)`，许可来源 `YELLOW_DEDICATED_INSTANCE`。绑定自身不授予资格。宿主使用 `request_plan` 和 `next_phase`：enqueue → advance(实际内部 ID) → 等待真实完成 → query，直到全部阶段及恢复 complete。最多六次无注入预热，连续两次完整 hidden/logits 有限且精确一致才进入验收。

无许可实际 prepare 必须拒绝；阶段用尽再次 advance 必须拒绝。全 rank 每次 eligible 调用持续触发，统计八项计数；实际消费者、同 stream 完成、区间字节恢复、资源清零、新 OFF/health 完整 hidden/logits 精确恢复均须通过。不要求 token 改变。源码指纹不匹配时原样回传拒绝码，禁止修改指纹绕过。

社区版 linear_attn.out_proj 量化，黄区现役为 BF16。39/Batch 候选和命中计数可能不同，以实际覆盖为准，不要求与蓝区计数相同。39 matmul_range 当前只验收独立 W8A8 QMM 本地列分片；非量化累加路径未 NPU 验收并拒绝，不能外推为 BF16 已通过。

## 蓝区证据与限制

组合为 A2 / Ascend910B4 / CANN9.1 / vLLM0.26.0 / vllm-ascend0.26.0rc1 / 社区 Qwen3.8 W8A8 / TP2。以下均为公共入口、同组 Worker 两作用域：

|能力|每 rank 每作用域应用、消费、恢复次数|故障请求耗时倍数（两作用域）|
|---|---:|---|
|28 denominator torch|64|1.4265 / 1.3416|
|Batch immediate|1216|6.3440 / 6.2005|
|39 output_proxy|1216|2.4235 / 2.4336|
|18|644|1.8885 / 1.8466|
|22|64|1.1611 / 1.0945|
|39 matmul_range|704；512非列分片调用不适用|2.6538 / 2.2636|

28 是分母重放效果变体，不宣称复刻历史 kernel。39 output_proxy 区分新增变化、上游非有限与原值已为 Inf。22 summary 输出 rotary 行/QK 效果。Batch 只覆盖独立 DynamicQuant→QMM，不覆盖融合量化；scale 精确、Q 确定性样本仅半整数边界允许 ±1，不把抽样当全量证明。39 matmul_range 改变真实 QMM 操作数的重算 K 范围，只有一个目标输出单元/调用，不宣称原 kernel 溢出。

相关测试门禁 971 项（0 失败、0 错误、2项已有弃用 skip），补充33项通过，两组存在重叠。历史197项中85个失败出现次数已复测解决；112项Native研究路线欠账保留，不把全仓说成全绿。具体清单见 `rc3_r3_regression_review_20261122.md`。

## 黄区文本回传与登记

每个能力、每个 TP 组合回传：安装检查全文；独立 summary 全文；原回执文件名、字节数、SHA256；各 rank 的 observed/eligible/attempted/applied/changed/consumed/restored/failed；耗时倍数；异常和完整错误码。原始回执留黄区，不回传私有张量。

源包 `tools/review_continuous_public.py --require-negative-creation` 直接输出总结；生成命令和摘要完整性要求见验收步骤。未完成、无命中、恢复失败、缺字段或矛盾均不登记。审查通过后只登记 `YELLOW_DEDICATED_INSTANCE`（A3、0.26、现役 Qwen3.8 权重、TP1 或 TP2），证据等级“黄区文本证明”。BLUE_COMBINATION 不自动转为 A3，product_closed 仍取决于 openchamber 的实测与回传审查。
