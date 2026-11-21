# Batch 公共入口耗时：分解与有界延迟校验

基准是 A2 / 0.26 / Qwen3.8社区W8A8 / TP2 的 public-a6：两作用域 fault/normal 6.34397×、6.20054×。每rank每请求1216个真实QMM调用（304投影×4 forward）。不是单个量化kernel的6.3倍，也没有用CPU参考与NPU参考的接近耗时证明瓶颈归属。

## 源码计数与待测占比

|工作|当前每次fault调用|性质|
|---|---|---|
|Q与scale租约|各自backup/write/consume/restore，共8次完成事件|设备完成证明，不能直接删除|
|调用外层完成检查|另2次完成事件|共约10次显式同步，1216调用约12160次/rank|
|完整区间指纹|Q和scale写前/写后/恢复，6次回传|约7296次；不能用抽样替代恢复证明|
|CPU量化参考|输入样本、Q样本、全部scale及全量absmax回传|数次小传输与同步|
|变化计数/有限值等|标量回读|隐式同步，须计入|
|Python hooks、调度/租约和字典|每调用/每请求|需wall-clock分段测量|

上表是静态调用计数，不是时间百分比。下一窗口应分别计时reference、lease、fingerprint、consumer和控制边界；NPU异步段用事件，不能用Python返回时刻替代设备完成。

## CPU已实现、尚未启用的优化

`engines/deferred_quant_reference.py` 在每次真实调用同stream保存确定性样本、极值、Q和全部scale的独立快照；前向结束合并为三个传输，使用一次完成fence，再调用原 `compare_samples`。最大512调用、32MiB快照及拼接双份预算。预算不含量化算子的瞬时FP32 x/abs工作区，须另测峰值；超预算明确失败，不跳过eligible调用。

准则不变：scale逐元素精确；Q仅在原2-ULP半整数边界允许±1；没有额外epsilon。参考失败保留快照供诊断。重复调用ID、stream变化、布局变化和超预算拒绝。CPU测试覆盖live buffer复用后快照仍正确、一次fence覆盖多调用、错误scale/非边界Q拒绝、stream/budget拒绝及拼接峰值预算。

该队列尚未接入公共Batch默认路径，现有6.3倍验收原件和r2代码不变。下一轮NPU必须核对reference回执与原实现相同、完整恢复、两个作用域及实际耗时，才能称优化通过。

不能把所有恢复推迟至forward末尾：真实QMM完成后输入缓冲区可能被复用或合法改写。当前对象租约的backup→rewrite→device consumer→restore仍逐调用执行。后续若优化此处，须先证明设备同stream顺序和对象生命周期，再讨论合并fence，不能仅移除synchronize。
