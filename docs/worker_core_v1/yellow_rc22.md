# RC2.2 交付草案：请求计划与研究入口

**未冻结、未交付、未发送黄区**。RC2.1只登记审查通过的黄区RC2组合；本文件与代码改动进入RC2.2，不能回写RC2标签或包。当前RC2.2范围为41、request_plan、28研究入口标注、prerequisites缺项汇总，以及18/39/Batch同Worker双作用域补验。20/26/33/28保持低投入关闭项，未通过不收录为可执行；RC2.2冻结后测试部署线仅维护。

## 请求计划唯一来源

`fault_injection.request_plan`保存阶段元组，provider实际推进和能力查询/任务创建读同一来源。公开capabilities的每行、`describe_delivery`与Runtime prepare/query的`provider_state.request_plan`相同。查询不运行模型、不collect、不推进阶段。无普通任务者plan为null；28未通过前没有计划。

|机制|有序阶段|内部上限|外部健康|
|---|---|---|---|
|8、17、18、22、39 output_proxy|normal→fault→off→health|4请求|无需额外请求|
|41|normal→normal_repeat→lossless→fault→off→health|6请求|无需额外请求|
|Batch|normal→reference→lossless→calibration_off→fault→off|6个B8批/48请求|开始前和清理后各1个B8健康批，总计64请求|

Batch的外部health不是decode token；这些数据反映现有验收器实际行为。`max_iterations`是成功计划的最大推进轮数；`complete`表示阶段完成，不表示资源已释放。仍必须finish、unregister、detach及读取资源为0。失败恢复的OFF/health由机制专用recover契约限定，不重试F、不靠补满循环隐藏失败。

宿主参考流程（伪代码中的宿主函数由接线实现；实际Client API是`advance`或`advance_batch`）：

```python
state = require_action(client.prepare())
plan = state['request_plan']
assert plan == capability['request_plan']
# 按已确认输入/阶段准备；先完成plan.external_health里的before_plan。
for action in ('register', 'check', 'enable'):
    state = require_action(getattr(client, action)())
try:
    for iteration in range(plan['max_iterations']):
        state = require_query(client.query())
        if state['next_phase'] == 'complete':
            break
        spec = plan['phases'][iteration]
        assert state['next_phase'] == spec['phase']
        ids = host_enqueue(spec)           # 真实enqueue返回内部ID；不伪造ID
        assert len(ids) == spec['request_count']
        reply = (client.advance_batch(ids) if plan['unit'] == 'batch'
                 else client.advance(ids[0]))
        require_action(reply)
        host_wait_for_actual_completion(ids)  # 有界超时、核对实际完成ID
        state = require_query(client.query())
        assert not current_failure(state)
    state = require_query(client.query())
    if state['next_phase'] != 'complete' or not state['complete']:
        raise RuntimeError('REQUEST_PLAN_NOT_COMPLETE_WITHIN_BUDGET')
    require_action(client.finish())
    require_action(client.unregister())
    require_detached(client.detach())
    verify_zero_resources_and_completed_recovery()
    # 再完成plan.external_health里的after_cleanup。
except Exception:
    # 只取消本作用域的在途请求；按机制recover契约完成必要OFF/health。
    # 不允许直接detach掩盖所有权；无法恢复则保留owner并报告。
    recover_owned_task_without_repeating_fault()
    raise
```

各provider保持其已有`*_PHASES_EXHAUSTED`错误；计划不足时`complete=false`，F前abort可清理；F已发生必须按recover完成OFF/health。RPC大小上限触发时仍返回只读request_plan，状态失败、不授complete；这不放大证据上限。

## 缺失字段报告

当前开发线`serial_eager_assessment`汇总全部不满足项，缺少pcp_size返回expected=1、actual={missing:true}、pending；同时保留async/multimodal等不匹配，不抛AttributeError、不补默认值。这只改善诊断；0.26等新部署默认关闭，语义适配属于RC3。冻结RC1.1的首项拒绝不被改写。

## 28 显式研究入口

三项历史机制保留原源码、kernel与证据。公共`attach`、普通Worker registry不执行；FaultConfig可描述目录条目但不能授执行，不提供用户research开关。显式研究调用仅经`worker_extension.registry.create_research_provider`或直接研究类构造；前者是研究协议对象，不能把历史HF/model_adapter结果转换成Worker资格。

历史HF/实验路径可直接使用原`fault_injection.attention.AttentionSession`及对应`FiaBackend`，FA路径使用原native_execution_strategy研究实现；必须满足各自旧launcher、环境、精确几何约束。接口存在不表示任意模型/栈可运行。限定路径、提交、证据和几何在只读`research_entry`及[28决策说明](fault28_status_decision_20261119.md)中。

`research_entry`不改变HISTORICAL_ONLY/TND_FAULT_IMPLEMENTATION_INCOMPLETE，不授delivery_capability/qualification，不生成已验收登记。发行包可以保留源码，发行说明必须把这些单列为“研究保留，不属于交付”。

## TP1 实机范围与历史记录

18、39、Batch 同 Worker 两作用域补验已在 124 A2、CANN 8.5.1、vLLM-Ascend 0.18 完成。18/39 为 Qwen3-0.6B BF16 TP1；Batch 为同模型 W8A8 TP1、每批8请求。Batch 每作用域六批和前后健康，共8批64请求，两个作用域复用同一 Worker。完整 hidden/logits 外部健康核验见 ordinary_batch_full_health_20261120_summary.json；较早 token-only 记录原样保留且不替代完整核验。

这些是已保存证据的收录，本发行打包阶段没有再次运行 NPU。41 沿用 A2/0.18/Qwen3-0.6B BF16/TP1 两作用域证据；文件名中的 a3 是 attempt 编号，不是 A3 硬件验收。41 尚未登记黄区。

## 2026-11-20 晚发行候选

以 d07c7215 为基础按 releases/RC22-SOURCE-SELECTION-20261120.json 选入单卡改动，未带入 M1/TP2。
18、39、Batch 的 TP1 同 Worker 双作用域已完成；Batch 后补的完整 hidden/logits health 证据见 ordinary_batch_full_health_20261120_summary.json。41仍仅为蓝区A2/0.18/Qwen3-0.6B BF16/TP1验收，黄区默认关闭。

继承RC2.1的三项文本登记及限制；不把18黄区的不同Worker两作用域改写为同Worker。
22总结器现在保留 actual_qk_before/after、正确/替代rotary行、独立reader输出哈希及非目标/缓存恢复字段；旧回执缺失项不会被补造，仍受64KiB限额。

新发行宿主的可信binding包装使用 fault_injection.rc22_profile.prepare_binding，七项候选包含41；既有rc2_profile保持六项不变以免改变冻结接口的含义。该包装只限定发行候选，不授任何部署准入。

普通请求计划沿用上文；研究入口仍拒绝。软件包版本为demoncatpfi 0.3.2 / worker-extension 0.2.2 / fault39-core 0.2.2。候选 CPU 回归137项通过（其中1项跳过）；无设备安装及制品冻结结果以发行目录清单为准。
