# M4 动态 EP 调用归属：当前设计与CPU实现

复用既有MultiRankTaskClient/owner/generation/四阶段生命周期；不新增任务状态机或条件命中collective。实际TP8+EP未验收，公共组合保持关闭。

`fault_injection/sites/ep_context.py` 在已有共同控制边界校验实际回执。输入由部署适配器从运行时提取：

- 每rank的物理专家slot→逻辑expert映射，支持同一逻辑专家有多个EPLB副本，不从config平均划分。
- 本次dispatch的源调用序号、实际request ID、token位置、top-k路由序号、实际物理slot。源调用序号需要跨rank一致，不可使用rank本地自增计数冒充。
- 每次故障调用用当时路由；不与normal调用清单比较。无本地调用rank返回空清单，不制造调用。
- 所有rank提供owner/generation和控制边界回执；漏rank、旧作用域、未知request、与实际placement不符及同一dispatch route被重复消费都拒绝。

CPU测试覆盖world_size=1同路径、8rank中7rank零调用、故障路由改变、重复dispatch、旧generation与错误placement。它不能证明现场MoeDistributeDispatchV2/CombineV2的映射正确，也不能单独证明覆盖或资源恢复；最终仍须合并`continuous_contract.review_rank_coverage`及实际消费者/恢复回执。

下一步运行适配需提供：实际executor边界、EP placement读取来源、dispatch/combination字段与token/request关联、同stream消费者，以及共同结束边界。蓝区没有0.32，按负责人要求不安排V4.1 NPU；这些缺口不得用V2-Lite稠密TP2或真实config的CPU夹具冒充通过。
