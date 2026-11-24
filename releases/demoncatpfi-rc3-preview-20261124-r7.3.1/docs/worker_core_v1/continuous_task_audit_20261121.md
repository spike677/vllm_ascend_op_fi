# 全 rank 持续任务逐项审视与执行顺序

依据负责人2026-11-21任务书及最新模型元数据补充。以下是新持续口径，不继承历史单次通过为持续通过。冻结发行包未改。

## 哪些尚未展开

|项目|已完成|缺口/下一步|
|---|---|---|
|交付范围|12项清单；26 residual排除查询/创建|其余未验收组合继续默认关闭|
|公共生命周期|原MultiRankTaskClient持续模式、四阶段计划、rank覆盖审查CPU验证|实际provider/query/finish与准入整合未闭环；实验RPC不是公共Runtime验收|
|逐调用租约与设备顺序|28真实对象强引用、同stream event、消费后恢复与逐调用SHA|尚未抽取为全方法共用的区间租约管理；Native活跃对象/合法原地更新边界未做|
|28|V2-Lite和Qwen3.8真实TP2，全层、prefill+3decode、两作用域通过|仅最后query行；公共Runtime、更多形状、V4.1稀疏MLA未完成|
|Batch|共享scale核心CPU通过；本轮新增真实DynamicQuant/QMM适配及CPU异常恢复测试|两模型B8 TP2真实site通过；Qwen3.8实际304投影每rank1216调用；公共Runtime待完成|
|39 output_proxy|旧单次/NPU实现保留|新栈BF16输出全rank持续、already-Inf区分尚未验收|
|18、22|旧普通任务和单次证据保留|新持续provider与Qwen3.8实测未完成|
|39 matmul_range|历史高保真实现保留|torch参考重算、新量化/非量化投影接口未展开|
|33|历史分页策略实验保留|新通用paged decode参考重算、合法key集合、持续调用未展开|
|8|历史MoE索引故障保留|V2-Lite新持续、动态路由归属和M4 EP实际接线未完成|
|17|原视觉模型范围保留|持续改造未展开；不扩大模型范围，优先级最低|
|41、26 fia、20|历史Native代码/证据保留|memory_fault_consolidation、活跃对象区间读写恢复、实际安全恢复点和持续TP2均未展开；按任务放最后|
|M4|8-rank CPU协调；V4.1真实元数据与少层dummy结构夹具|实际专家分布、动态token/request归属、稀疏索引及EP汇总仍缺，不能宣称完成|
|RC3|已有历史材料|新持续binding/许可/request_plan/负向检查/覆盖summary候选包未冻结；黄区未验收|
|性能|28短请求实测倍数及30s实验上限建议|其他方法预算待测；不对任意长度/批大小开放|

## 前轮执行安排（已被阶段1公共Runtime优先替代）

2026-11-21晚最新版：先提交同步、黄区接收和环境缺陷报告，再接公共Runtime持续准入。Batch新增NPU计算与仅舍入边界±1的抽样参考验收；下述CPU全量参考结果作为已完成的实验开发证据保留。只有公共入口两个作用域验收通过才登记组合与构建RC3预览包。

1. 按已审核0.26镜像接Batch的实际`AscendW8A8DynamicLinearMethod.apply`，自动发现模块，不让用户填层名。保留原DynamicQuant调用作备份，torch共享量化写入其真实Q/scale输出，原QMM消费后用同stream event等待，再恢复字节；CPU独立NumPy参考逐元素比较。
2. 124卡2+5用原资源锁、逐case门禁和有界清理器。先Qwen3-0.6B W8A8，再Qwen3.8 W8A8，B8、4生成token，最多6次无注入预热，连续两次全有限且完整hidden/logits一致才进入两个作用域。
3. 记录真实调用集；不把未观测到的调用当作命中，不把先验208乘次数当作实测。异常/未覆盖量化入口、padding映射不明、CPU参考不一致均拒绝验收。
4. 此后按原顺序39op、18/22、39mr、33、8、17；真实内存三项最后。每项实际site验收后还必须完成公共Runtime接线与准入，不用实验通过替代发行完成。

GDN基线不稳时只记录环境缺陷，不重新追查或修改第三方kernel。全rank条件命中点不加collective；所有汇总仍在请求结束的共同控制边界。

## 四环境总表

实时表维护在 [continuous_progress_20261121.md](continuous_progress_20261121.md)。A2结论不转为A3，V2-Lite稠密MLA不转为V4.1稀疏MLA，社区元数据与CPU结构不能授予黄区现役准入。
