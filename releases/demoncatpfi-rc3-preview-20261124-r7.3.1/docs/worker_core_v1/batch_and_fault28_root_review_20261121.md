# Batch阻塞与故障28原始根因复核（2026-11-21）

本轮为本地源码/既有回执审查及3项CPU递推测试；没有新增NPU调用，不修改冻结包或公共准入。

## 当前结论

- M1：A2/CANN8.5.1/vLLM-Ascend0.18/Qwen3-0.6B W8A8/TP2的Batch两作用域已经通过，保留原结果。
- M2：A2/CANN9.1.0/vLLM0.26+Ascend0.26rc1/社区Qwen3.8-27B W8A8/TP2的无注入B8仍失败，未完成Batch故障两作用域。GDN是线性注意力；该异常不是FIA切分证据。
- 28：历史合法切分机制、实验模型传播与较新的TND分母错误研究闭环均有证据，但不是同一机理，也不代表普通Worker入口全部完成。

## Batch定位是否完成

已经定位数值异常边界和错误累加指纹，尚未完成底层责任定位与修复验收。

1. 首个NaN在Layer0 GDN WY输出u；KKT更早已有有限值异常。只查NaN会错过更早的错误。
2. 有限哨兵BF16=1.25实验中，23个输入完全可匹配head的w/u各16,192个字逐位符合“把两个BF16组成FP32，与旧字相加”。正常覆盖匹配为0。早先“整块漏写”的解释已修正。
3. KKT异常task集中于同一core的后续工作，累加此前Gram矩阵可解释异常结果。仍需证明控制状态由哪一段代码设置/遗留，以及当前编译和运行边界为何未复位。
4. 最新a9于07:20:11 UTC结束，隔离解除。首轮B8仍为request1=[0,0]、request5=[321,1179]；重复B8均为[271,760]。两rank均没有inline_atomic_recompile记录，因此不能宣称原kernel复位修正已实际生效，更不能判为通过。不能仅据新kernel名称推断修正装入。
5. 零初始化不是交付修复；基线未通过，不继续故障注入掩盖它。

依据：[定位报告](m2_b8_nan_localization_20261121.md)、[逐位分析器](../../experiments/worker_core_v11_20261118/analyze_gdn_atomic.py)、[派生证据](../../experiments/worker_core_v11_20261118/gdn_atomic_store_evidence_20261121.json)。a9原件仍在124，未修改。

## 故障28与原始根因的对应关系

原始链：Batch改变 → M/query轴切分改变 → causal下核内Reduce分块/尾块粒度改变 → 同一query完整Q/K/V下FIA输出改变 → 后续logits可能改变。Reduce不跨核，数学上的有效key集合不改变。原材料没有证明“分块重复/遗漏”或“分母多算”。

|当前变体|源码检查|判定|
|---|---|---|
|FA合法M/nRatio|M-only、ratio-only、joint控制分开|机制相邻，FA不能替代FIA历史路径；联合变化不能单独归因M|
|FIA合法切分|FiaBackend.native_plan为M/S2 512→128耦合；causal helper按tile末query裁剪并32对齐|方向接近原因，但同时改M/S2不足以独立证明M→核内Reduce；受控裁剪不能冒充历史默认代码|
|FIA denominator_replay|apply_tnd28_denominator.py在第二S2块之前对旧inSumTensor乘2；不相应重复分子|数学错误扩展，不是合法切分的浮点非结合性；0.7968%统计幅度不能当原故障28复现证据|
|真实Worker TND两作用域|研究B3/B4及双research owner的恢复通过；普通Runtime B5未完成|保留研究通过，不等于普通入口可交付，也未闭合历史Batch→M因果链|

TND最新回执baseline key为5000000000000200103，F为119000000030400003；候选还调整M/S2及调度核数。健康M128控制此前未通过。因而该组模型B/F不能将全部差异唯一归因到分母乘2，尽管源码/CPU参考明确证明新增错误的算术含义。

关键源码：[FIA选择](../../fault_injection/fia.py)、[BNSD causal宽度](../../fault_injection/native_adapters/attention28/fia/fault28_causal_tile.h)、[TND causal宽度](../../fault_injection/native_adapters/attention28/fia/tnd28_causal_tile.h)、[分母变体](../../fault_injection/native_adapters/attention28/fia/apply_tnd28_denominator.py)。

## 仍可采用的方式，按原根因接近程度排序

1. **原生Batch触发（首选根因验证，非强制注入）**：固定目标query的完整Q/K/V，在原生FIA B1/B8观测实际M、S2循环和causal尾块；不改算子。如果没有策略切换或无数值差异，如实记录。
2. **仅替换M选择规则（优先注入方案）**：在匹配版本真实host策略边界，将同次调用的M改为另一已验证合法M；其他独立策略输入、dtype、累加精度和softmax公式不变。循环、尾块、workspace由原host重算；必须证明核内Reduce实际随M改变。不能仅改raw/POD或用日志声明命中。
3. **由M派生核内分组（备用机制模拟）**：原生实现不暴露上述联动时，在真实kernel中让M决定S2分块或归约合并树；每个有效key恰好一次，分子/分母在线max重标定一致，不跨核。不保证大误差；明确标为受控归约分组扩展，不声称历史默认实现复刻。

分母重复、丢块、重复key、改mask或降累加精度均改变故障机制；不能作为严格原根因的替代通过路径。

## 关闭因果缺口的最小矩阵

先确定真实FIA布局/版本支持，再做B1/B8 × M策略A/B交叉对照。目标请求Q/K/V逐元素一致、完整有效key范围一致；记录实际query tile、每轮S2起止及尾块宽度、core归属。若同M消除跨Batch差异、固定Batch切M重现差异，才支持M因果；若仍不同，应继续查其他路径，不强归因。

先在真实算子中完成M-only对照；随后验证完整模型传播、重复稳定和OFF恢复。不能要求合法浮点重排必然超过0.1%；已有合法切分统计差异小与该机理相容。不得为过阈值而混入分母错误。真实图模式历史路径和当前eager/TND普通Worker入口分别验收。

CPU检查：test_recurrence.py三项通过。均匀scores=0、values=1、256 keys时，正确softmax输出1；分母replay输出2/3。这直接说明分母变体改变数学结果，不是仅改变舍入次序。该检查不是NPU验收。

当前建议：将28原根因主线明确回归“原生Batch触发＋M-only反事实对照”；denominator作为独立研究错误保留。此次没有扩展28预算，也未恢复此前因同因失败止损的M128实测。
