# Batch 公共入口耗时：分解与有界延迟校验

基准是 A2 / 0.26 / Qwen3.8社区W8A8 / TP2 的 public-a6：两作用域 fault/normal 6.34397×、6.20054×。每rank每请求1216个真实QMM调用（304投影×4 forward）。不是单个量化kernel的6.3倍，也没有用CPU参考与NPU参考的接近耗时证明瓶颈归属。

## 源码计数与待测占比

|工作|当前每次fault调用|性质|
|---|---|---|
|Q与scale租约|各自backup/write/consume/restore，共8次完成事件|设备完成证明，不能直接删除|
|调用外层完成检查|另3次完成事件（消费者后、finally恢复前、finally恢复后）|共11次显式同步，1216调用13376次/rank|
|完整区间指纹|Q和scale写前/写后/恢复，6次回传|约7296次；不能用抽样替代恢复证明|
|CPU量化参考|输入样本、Q样本、全部scale及全量absmax回传|数次小传输与同步|
|变化计数/有限值等|标量回读|隐式同步，须计入|
|Python hooks、调度/租约和字典|每调用/每请求|需wall-clock分段测量|

上表是静态调用计数，不是时间百分比。下一窗口应分别计时reference、lease、fingerprint、consumer和控制边界；NPU异步段用事件，不能用Python返回时刻替代设备完成。

## 已完成 CPU 与 TP2 实测的延迟核对

`engines/deferred_quant_reference.py` 在每次真实调用同stream保存确定性样本、极值、Q和全部scale的独立快照；前向结束合并为三个传输，使用一次完成fence，再调用原 `compare_samples`。最大512调用、32MiB快照及拼接双份预算。预算不含量化算子的瞬时FP32 x/abs工作区，须另测峰值；超预算明确失败，不跳过eligible调用。

准则不变：scale逐元素精确；Q仅在原2-ULP半整数边界允许±1；没有额外epsilon。参考失败保留快照供诊断。重复调用ID、stream变化、布局变化和超预算拒绝。CPU测试覆盖live buffer复用后快照仍正确、一次fence覆盖多调用、错误scale/非边界Q拒绝、stream/budget拒绝及拼接峰值预算。

队列已接入可信部署参数 `reference_mode=forward`；默认仍为 immediate，r2 不变。public-a17 在相同 A2 / 0.26 / Qwen3.8 W8A8 / TP2 上通过公共入口两作用域：每 rank 每作用域 1216 次应用、消费和恢复，失败0。抽样数分别为1245170/1245168，与 immediate 相同，Q差值0、scale精确，OFF/health完整hidden/logits恢复。独立报告为 `experiments/worker_core_v11_20261118/continuous_public_batch_forward_review_20261121.json`。独立模式字段参与部署准入，不能继承 immediate 的验收身份。

|运行|作用域|normal秒|fault秒|fault/normal|
|---|---|---:|---:|---:|
|a6 immediate|1|1.480177348|9.390206204|6.343974×|
|a6 immediate|2|1.496098816|9.276616880|6.200538×|
|a17 forward|1|1.485617823|8.814571610|5.933270×|
|a17 forward|2|1.454316057|8.689359075|5.974877×|

本轮fault绝对时间约下降6.1%/6.3%，但两轮独立进程不是交错重复性能试验，不能认定稳定收益或给出置信区间。它仍接近6倍耗时，逐调用租约、字节指纹和消费者同步尚未优化；没有把费用归因百分比当作实测。下一步应增加分段计时和交错重复，保留严格精度和恢复标准。

a17 窗口：2026-11-21T13:11:46.137833Z 至 13:17:24.350358Z，338.213秒（含加载和退出），卡2/5有界隔离已解除，无外部容器停止事件。源码清单 `422a927b91c3481ae37a8a8c3c974ef5d8e13e625365f1e5f822b1f5ee3d2f14`。

不能把所有恢复推迟至forward末尾：真实QMM完成后输入缓冲区可能被复用或合法改写。当前对象租约的backup→rewrite→device consumer→restore仍逐调用执行。后续若优化此处，须先证明设备同stream顺序和对象生命周期，再讨论合并fence，不能仅移除synchronize。
