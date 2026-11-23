# 公共17项的交付范围与5项不交付结论

更新：2026-11-23；依据负责人最新决定、黄区纠正文本，以及开发提交 `5deb8c50`。本页是全项目“公共17项 + 黄区自有15项 = 32条”状态表的公共部分，不新增故障实现或部署资格。

## 数量与验收口径

- 公共历史清单共17项，本轮交付范围12项、明确不交付5项。
- 冻结 r6 的 [continuous_contract.py](../../fault_injection/continuous_contract.py) `MECHANISMS` 为12项，`PRODUCT_CANDIDATES` 为8项。
- r7开发线 `5deb8c50` 的 [product_contract.py](../../fault_injection/product_contract.py) 有11个产品候选：r6八项加41、26fia、20。新增三项只通过A2组合，不能把“11”称为r6数量，也不能因更正r6数字而撤销真实r7结果。
- “列入交付”不等于“所有环境产品验收通过”。Qwen3.8 TP1、Qwen3.8 TP2、DeepSeek V4-0731分别记录；蓝区A2不代表黄区A3，验收模式不代表产品模式。
- 黄区最终状态采用“产品验收通过 / 不适用 / 不支持 / 与某项合并 / 诊断工具”。尚未回传的产品结果明确标为待验收，不能为消除“未知”而虚构通过。本页5项不交付已结案，不安排注入；其余逐环境状态见[进度表](product_progress_20261123.md)。

## 17项逐项结论

|公共机制ID|交付决定|当前证据与范围|
|---|---|---|
|8.restore_offset|交付范围内，产品仍关闭|MoE专用；Qwen dense不适用。DeepSeek V4的实际combine行归属、全rank观测和噪声标定待回传，V2-Lite不代替V4资格|
|15.alias_detach|不交付：当前部署不适用|缺少自然的私有格式base→view原地写入→原base消费链，见下表|
|17.spatial_padding|交付范围内|仅已适配视觉请求范围；r7 A2 TP2真实HTTP内置金丝雀图及用户图回归通过|
|18.api_half|交付范围内|r7 A2 TP2产品回归通过；不把其结论外推所有norm/融合分支|
|20.buffer_substitute|交付范围内|r7 A2 TP2真实活跃输入内容覆盖、消费者、恢复通过；A3 Native仍关闭|
|22.wrong_row|交付范围内|r7 A2 TP2融合RoPE产品回归通过；新模型/算子仍须匹配|
|25.reduce_scatter|不交付：不支持|没有已验收的真实集合通信注入实现，不用其他扰动替代|
|26.residual_stomp|不交付：本轮不支持（范围排除）|负责人明确退出本轮交付；不是否认历史机制证据|
|26.fia_destination|交付范围内|r7 A2 TP2写当前活跃residual目的区、真实消费及恢复通过；不声称原kernel指针重定向|
|quant.batch_shared_scale|交付范围内|r7 A2 TP2产品回归通过，真实decode合批2/3/4/7/8，B1不注入；仅独立量化调用范围|
|28.fa.legal_tiling|不交付：本轮不支持（仅历史研究）|固定FA合法策略实验保留，不扩展产品资格|
|28.fia.legal_tiling|不交付：本轮不支持（仅历史研究）|固定FIA合法切分证据保留，不用denominator变体的通过结论替代|
|28.fia.denominator_replay|交付范围内|交付torch分母重复贡献的命名变体；r7 A2 TP2产品回归通过，不是历史默认kernel复刻|
|33.paged_decode|交付范围内|r7 A2 TP2产品回归通过，仅已绑定2304-token prompt/4-token生成；A3 TP2/TP1分别待取证与控制审查|
|39.matmul_range|交付范围内|r7 A2 TP2产品回归通过；按受支持投影和重算规则，不是任意矩阵通用资格|
|39.output_proxy|交付范围内|r7 A2 TP2真实投影BF16输出及消费者回归通过；输出效果代理，不声称累加器根因|
|41.tail_column|交付范围内|r7 A2 TP2真实KV区域、FIA消费者和恢复通过；无对应paged写入的prefill不适用，A3 Native仍关闭|

上述A2产品证据统一为 Ascend910B4 / vLLM 0.26 A2栈 / CANN9.1.0 / 社区Qwen3.8 W8A8 / TP2；每项同组Worker两个owner、真实HTTP、全rank合格调用、消费者、字节恢复/资源清零/关闭后金丝雀恢复。完整证据见[r7 A2报告](r7_native_a2_20261123/README.md)。黄区r5.1登记的是验收模式组合，不能据此填产品通过。DeepSeek产品资格没有由本页新增。

## 5项不交付：原因、门禁及保留证据

|机制|不交付的具体原因|当前普通入口拒绝原因|证据位置及不能外推的结论|
|---|---|---|---|
|15.alias_detach|在当前现役适配范围未证明自然视图写入链。仅有权重view被读取不满足条件，不能给模型人为增加写入后声称自然故障|`FAULT15_CURRENT_DEPLOYMENT_NO_PRIVATE_VIEW_WRITE_CHAIN`|[能力卡](../overnight_fault_cards/15.alias_detach.md)、[实现](../../fault_injection/alias.py)：格式29/BF16、base16×16、view8×8、stride16/1、offset17的小型NPU机制证据保留；不覆盖内部native视图或现役完整模型|
|25.reduce_scatter|实际group/backend/调用顺序、同输入归约差异及恢复未形成可交付Native实现；CPU合同或浮点次序模拟不等于通信实测|`FAULT25_UNSUPPORTED_NO_SUBSTITUTE`|[能力卡](../overnight_fault_cards/25.reduce_scatter.md)、[清单](../../fault_injection/catalog.py)：`not_implemented` / `not accepted`；禁止改成通信输出随机噪声、丢块或重复贡献来顶替原故障|
|28.fa.legal_tiling|负责人选择大误差denominator_replay作为28本轮交付，合法FA切分保留固定版本实验；未转成当前产品范围|`FAULT28_LEGAL_HISTORICAL_ONLY`|[28状态决策](fault28_status_decision_20261119.md)、[FA能力卡](../overnight_fault_cards/28.fa.legal_tiling.md)：历史FA S1S2策略和固定QKV/模型传播证据，不是FIA或现役产品证明|
|28.fia.legal_tiling|同样仅留历史研究；合法M变化可以产生微小真实差异，但本轮输入/策略搜索未满足后来固定容差首差及0.1%统计可观测性目标。不能把“差异小”泛化成策略永远无效|`FAULT28_LEGAL_HISTORICAL_ONLY`|[28状态决策](fault28_status_decision_20261119.md)、[FIA能力卡](../overnight_fault_cards/28.fia.legal_tiling.md)：历史图/eager/统计证据保留；torch分母重复贡献和Native denominator研究均是另一个变体|
|26.residual_stomp|负责人明确“只保留26 fia_destination”，严格邻接和residual定向污染均退出本轮；属于范围决定，不是技术不可能|`EXCLUDED_CURRENT_DELIVERY:本轮排除`|[最新范围决定](strategy_memory_discussion_20261121.md)、[能力卡](../overnight_fault_cards/26.residual_stomp.md)、[历史总结](../../experiments/provider_adaptation_20261116/validation_summary.json)：旧32B W8A8 TP2推理写入/恢复证据保留，旧固定CANN/分配器链不授新产品资格|

拒绝原因来自 [ordinary_admission.py](../../src/demoncat/runtime/vllm_ascend/worker_extension/ordinary_admission.py)；这5项均不在连续12项清单和产品候选清单。历史代码仍可存在于研究目录，不因此允许公共配置开启。`catalog.py` / `delivery_scope.py` 的历史条目不能替代当前组合准入。

## 黄区分工与r7边界

黄区自有15项由黄区自行探测、适配与产品验收，Codex不开发；W4A8到W8A8按模型/量化路径适配，不继承历史通过。重复项由黄区依据机制及真实注入点决定合并，同一故障仅一个CLI入口；只观测工具不计故障。外部接口契约暂缓，不纳入r7。

黄区CLI负责全局同时只允许一个故障生效，覆盖公共和自有两条路径；单模块互斥不能代替这条更强的规则。状态字段来源的核对见[本次接收记录](yellow_intake_20261123_corrections.md)，不为此修改r6公共代码。r7 N/D/P安排不变：N批A2已通过，等待黄区Native完整提取审查；D/P仍按各自实际证据推进，冻结标签和发行包保持不变。
