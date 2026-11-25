# 公共17项：13项交付范围（15进入r7.1候选）、4项不交付

2026-11-25夜间状态以[r8优先执行单](codex_plan_20261125_r8_first.md)、[阶段1收尾更正](yellow_phase1_y4_y5_resolution_20261125.md)及[控制证据更正](yellow_intake_20261125_control_evidence_correction.md)为准：r8未冻结，黄区已声明通过的项目仍须正式文本审查，不能沿用最初“六项”的统计。41此前为harness漏映射；26fia补配置后通过的声明待同次运行身份/效果页；39op第二owner异常和Batch enable exit5尚未定性。2026-11-26凌晨当前r8候选已通过16项[全项A2回归](r8_full_regression_20261125/README.md)，进度由逐项回执生成；A2通过不代表A3异常已修复。33与D包为第二批增量，未验收组合继续关闭。

2026-11-24修订：故障库与公共编号不相同。唯一当前口径为[41行覆盖台账](fault_library_coverage_20261124.md)：库8为CP padding，库17为转置卷积padding；库27→公共15、库41→公共20、库39→公共39mr。第3、37行均为注入能力并保留配置说明；第30行新增18方差误差命名代理，尚未通过实机验收。原12行映射被替代，历史发行物不改。

负责人已确认：第37行配置项说明加舍入变体；第13行纳入D包与公共8一起测；第7行仅在Qwen做贡献丢失现象复现。按已有能力＋变体登记，不增加公共机制数量。D包依赖和黄区适配边界见[d_package_scope_20261124.md](d_package_scope_20261124.md)；范围批准不代表实测或产品登记通过。

更新：2026-11-23；依据负责人最新决定、黄区纠正文本，以及开发提交 `5deb8c50`。本页记录公共17项；黄区自有14项、诊断工具1项、配置说明2项分别记录，不新增故障实现或部署资格。

## 数量与验收口径

- 公共历史清单共17项，本轮交付范围13项（含15的r7.1候选）、明确不交付4项。
- 冻结 r6 的 [continuous_contract.py](../../fault_injection/continuous_contract.py) `MECHANISMS` 为12项，`PRODUCT_CANDIDATES` 为8项。
- r7开发线 `5deb8c50` 的 [product_contract.py](../../fault_injection/product_contract.py) 有11个产品候选：r6八项加41、26fia、20。新增三项只通过A2组合，不能把“11”称为r6数量，也不能因更正r6数字而撤销真实r7结果。
- “列入交付”不等于“所有环境产品验收通过”。Qwen3.8 TP1、Qwen3.8 TP2、DeepSeek V4-0731分别记录；蓝区A2不代表黄区A3，验收模式不代表产品模式。
- 黄区最终状态采用“产品验收通过 / 不适用 / 不支持 / 与某项合并 / 诊断工具”。尚未回传的产品结果明确标为待验收，不能为消除“未知”而虚构通过。本页4项不交付保持关闭，不安排注入；其余逐环境状态见[进度表](product_progress_20261123.md)。

## 17项逐项结论

|公共机制ID|交付决定|当前证据与范围|
|---|---|---|
|8.restore_offset|交付范围内，产品仍关闭|MoE专用；Qwen dense不适用。DeepSeek V4的实际combine行归属、全rank观测和噪声标定待回传，V2-Lite不代替V4资格|
|15.alias_detach|r7.1候选；A3 TP2待验收|B档ND视图别名断开，真实impl写临时存储、gate读旧base、o_proj消费；A2 TP2产品已通过。无私有格式触发条件；黄区CLI需补15映射。见[执行单](alias15_a3_execution_20261123.md)|
|17.spatial_padding|交付范围内|仅已适配视觉请求范围；r7 A2 TP2真实HTTP内置金丝雀图及用户图回归通过|
|18.api_half|交付范围内|r7 A2 TP2产品回归通过；不把其结论外推所有norm/融合分支|
|20.buffer_substitute|交付范围内|r7 A2 TP2真实活跃输入内容覆盖、消费者、恢复通过；A3受信验收候选，产品资格未登记|
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
|41.tail_column|交付范围内|r7 A2 TP2真实KV区域、FIA消费者和恢复通过；无对应paged写入的prefill不适用，A3受信验收候选，产品资格未登记|

上述A2产品证据统一为 Ascend910B4 / vLLM 0.26 A2栈 / CANN9.1.0 / 社区Qwen3.8 W8A8 / TP2；每项同组Worker两个owner、真实HTTP、全rank合格调用、消费者、字节恢复/资源清零/关闭后金丝雀恢复。完整证据见[r7 A2报告](r7_native_a2_20261123/README.md)。黄区r5.1登记的是验收模式组合，不能据此填产品通过。DeepSeek产品资格没有由本页新增。

## 4项不交付：原因、门禁及保留证据

|机制|不交付的具体原因|当前普通入口拒绝原因|证据位置及不能外推的结论|
|---|---|---|---|
|25.reduce_scatter|实际group/backend/调用顺序、同输入归约差异及恢复未形成可交付Native实现；CPU合同或浮点次序模拟不等于通信实测|`FAULT25_UNSUPPORTED_NO_SUBSTITUTE`|[能力卡](../overnight_fault_cards/25.reduce_scatter.md)、[清单](../../fault_injection/catalog.py)：`not_implemented` / `not accepted`；禁止改成通信输出随机噪声、丢块或重复贡献来顶替原故障|
|28.fa.legal_tiling|负责人选择大误差denominator_replay作为28本轮交付，合法FA切分保留固定版本实验；未转成当前产品范围|`FAULT28_LEGAL_HISTORICAL_ONLY`|[28状态决策](fault28_status_decision_20261119.md)、[FA能力卡](../overnight_fault_cards/28.fa.legal_tiling.md)：历史FA S1S2策略和固定QKV/模型传播证据，不是FIA或现役产品证明|
|28.fia.legal_tiling|同样仅留历史研究；合法M变化可以产生微小真实差异，但本轮输入/策略搜索未满足后来固定容差首差及0.1%统计可观测性目标。不能把“差异小”泛化成策略永远无效|`FAULT28_LEGAL_HISTORICAL_ONLY`|[28状态决策](fault28_status_decision_20261119.md)、[FIA能力卡](../overnight_fault_cards/28.fia.legal_tiling.md)：历史图/eager/统计证据保留；torch分母重复贡献和Native denominator研究均是另一个变体|
|26.residual_stomp|负责人明确“只保留26 fia_destination”，严格邻接和residual定向污染均退出本轮；属于范围决定，不是技术不可能|`EXCLUDED_CURRENT_DELIVERY:本轮排除`|[最新范围决定](strategy_memory_discussion_20261121.md)、[能力卡](../overnight_fault_cards/26.residual_stomp.md)、[历史总结](../../experiments/provider_adaptation_20261116/validation_summary.json)：旧32B W8A8 TP2推理写入/恢复证据保留，旧固定CANN/分配器链不授新产品资格|

拒绝原因来自 [ordinary_admission.py](../../src/demoncat/runtime/vllm_ascend/worker_extension/ordinary_admission.py)；这4项不在交付运行清单；r7.1另增15候选（连续13项、产品12项，8仍关闭）。历史代码仍可存在于研究目录，不因此允许公共配置开启。`catalog.py` / `delivery_scope.py` 的历史条目不能替代当前组合准入。

## 黄区分工与r7边界

黄区自有14项由黄区自行探测、适配与产品验收，Codex不开发；W4A8到W8A8按模型/量化路径适配，不继承历史通过。重复项由黄区依据机制及真实注入点决定合并，同一故障仅一个CLI入口；只观测工具不计故障。外部接口契约暂缓，不纳入r7。

黄区CLI负责全局同时只允许一个故障生效，覆盖公共和自有两条路径；单模块互斥不能代替这条更强的规则。状态字段来源的核对见[本次接收记录](yellow_intake_20261123_corrections.md)，不为此修改r6公共代码。r7 N/D/P安排不变：N批A2已通过，黄区Native构建文本已审查通过；A3实际运行验收待执行；D/P仍按各自实际证据推进，冻结标签和发行包保持不变。

## 11-23最新映射与15边界

故障库20→公共20（A3产品等待r7 N批，r6 fast_validation仅历史证据）；28→公共28 denominator（黄区自有tiling与公共项合并，黄区方法不适用）；38→黄区P0-15 HCCL；3/37为黄区配置说明，Codex仅引用，不另建注入能力。

15历史NZ29/BF16最小AliasMechanism：6次匹配，每次64目标、192非目标元素；11-16 B/F/OFF机制复测通过，msProbe严格首差INCOMPLETE_COMPARISON，写后ND格式转换为观测点而非故障发生点。10-28错误stride权重方案已废弃，曾误编号27。最新[阶段0/1报告](fault15_tier_decision_20261123.md)不授产品资格。


## r7.1今晚A3 TP2产品状态（不继承A2结论）

|交付项|A3 TP2当前状态|
|---|---|
|15|r7.1受信候选，待CLI映射与本轮实测文本|
|17、18、22、39op、39mr、28 denominator、Batch|r7/r7.1本轮产品回传待审；r5验收登记不能代替产品结果|
|41、26fia、20|r7受信Native候选；运行时CANN9.1及库身份已设门禁，A3产品回传待审|
|33|A3关闭，等两rank launch/控制≥99%取证，满足后另发r7.2候选|
|8|Qwen3.8不适用；DeepSeek EP combine映射另列，不制造MoE调用|

按能力逐份给“通过 / 不通过 / 需退补”。没有本轮回传的项保持待审，不登记。今晚不做TP1；历史TP1证据保留。r7.1发行和main快进不修改任何旧标签/包。


## 2026-11-24 r7.4.1 定向补测

新增 `18.variance_over4`（均方gain代理，非精确方差kernel）、外部Conv3d图片输出示例已完成A2/0.26/CANN9.1/Qwen3.8 W8A8/TP2真实HTTP两个作用域；原18/17产品回归通过。首次图片类型识别失败经修复后复测，原件保留。全rank匹配调用、消费者、字节恢复、资源及金丝雀通过。见[实机报告](r741_npu_validation_20261124.md)。这是开发线定向验收，不是新增独立公共编号、A3登记或发行冻结。

## B1—B6蓝区补齐

两个18固定gain档、module-input和DynamicQuant参数例已通过A2/0.26/CANN9.1/Qwen3.8 W8A8/TP2两个owner。状态诊断与15风险边界、排空复测、41行台账及33只读取证补充件已整理；见[r7.4.1蓝区收口](r741_b1_b6_closure_20261124.md)。A3资格和r8/D包不由本轮自动完成。
