> 最新结论：v2 weight_scale 与 KV 示例的 A2 TP2 两owner实测已通过；KV真实HTTP取消与超范围负例通过。详见 [r74_v2_acceptance_20261124.md](r74_v2_acceptance_20261124.md)。以下为按时间追加的开发记录，旧“待完成”不覆盖最新结论。发行仍须以最终标签及安装清单为准。

# r7.4增量开发检查点

基线：r7.3.1发行提交 `d23cfc9f762bf71c5c3277788a1b6454cd1b1e66`。分支 `codex/rc3-r7-4-20261124`，没有冻结新标签，不修改既有发行包。

|项目|当前证据|不能宣称的结论|
|---|---|---|
|公共机制/共享Worker路径|414个原有src/fault_injection文件逐字节一致，重新核对见r74_v2_source_delta_20261124.json|没有修改v1或默认开放v2|
|v2区域原语/产品facade|36项针对性CPU通过，另4项独立回执审查测试通过；同代际租约、排空、全rank失败回滚、金丝雀|KV产品接线和真实取消路径仍未验收|
|weight_scale实际示例|A2/0.26/CANN9.1/Qwen3.8社区W8A8/TP2，同组Worker两个owner；逐rank48元素变化，4次真实QMM读取，三项恢复通过，缺许可exit4|仅专用network-none、唯一HTTP提交者的示例；不是A3或通用服务接线资格；没有同输入F hidden传播对照|
|33只读采集器v2|124 CPU假ACL四场景通过；9000次调用完整捕获；A2 CANN9.1公开头文件无设备编译通过|不代表A3 ABI/source匹配、serializer解码或≥99%控制重算通过|
|33资格|仍关闭；公开9.1.0候选已取得，候选解码移到codex/rc3-r7-5-candidate-20261124|不把A2 SDK布局或≥99%单项当作A3解码证明|
|发行与安装|当前是开发增量|没有r7.4 wheel或无设备发行安装验收，不授予部署资格|

开发验证：旧14项基线为r74_incremental_cpu_20261124.json；本轮36项覆盖区域、Runtime、入口、产品facade、真实示例构造器，4项覆盖独立审查器。采集器的CPU模拟和编译证据分别为r74_capture_cpu_20261124.json、r74_capture_build_20261124.json。CPU假ACL不冒充NPU验收。

## 实机与隔离记录

四个已结束窗口见[r74_v2_windows_20261124.json](r74_v2_windows_20261124.json)：两个只读普查252/244秒；weight示例首轮263秒，构造器identity参数遮蔽函数导致TypeError，未写入，已补回归；修复后第二轮259秒通过。每个窗口均确认有界清理器停止、隔离解除；只docker stop精确占卡2/5的外部实例，未改派发器/Docker配置，未删除或重启外部容器。JSON记录绝对UTC时间、容器名、镜像、停止结果及恢复。服务退出有vLLM multiprocessing残留semaphore/shared_memory警告，不能把进程退出当成provider恢复；本次provider恢复依据是退出前两rank回执。

第二次普查每rank观察到16层full attention、304个实际scale消费者、128次ScatterPaKvCache和128次FIA、192个K/V存储读取匹配。**存储匹配不是页独占、值读取或恢复证明**。

weight示例独立审查见[r74_weight_example_review_20261124.json](r74_weight_example_review_20261124.json)，原回执SHA `047d6118eeacd169ad9c90129c4134b71820039dacf44cd7626130764f48a265`。自动选中首次已观察的dense QMM `linear_attn.in_proj_ba.weight_scale`，不是用户填写层名。源码锁定的未使用FP32派生scale在故障阶段移除，OFF恢复同一对象；实际QMM读取BF16 scale。chunked和其他scheme拒绝；A3须重新核对派生副本，不继承A2结论。

## 持久计数口径

八项计数以持久区域的**写入事件**为单位。本例每owner/rank为observed/eligible/attempted/applied/changed/consumed/restored各1、failed=0，changed_elements=48；4次前向消费者读取另列reader_observations=4。consumed=1代表该写入已被声明消费者实际读取，不是4次写入。值恰好相同允许changed小于applied。部分rank开启失败的恢复保留consumed=0，restore_kind=FAILED_ENABLE_ROLLBACK_NO_CONSUMER，独立审查拒绝把它当验收成功。

下一项是KV产品接线：活跃请求的页独占、writer窗口和请求结束但页未复用的恢复边界；不能套全局inflight=0假证明。已读实际0.26源码：同步EngineCore.step在model future/sample完成后处理abort；Scheduler._free_request在下一次Worker收到finished_req_ids之前就可能free页。因此不能靠Worker看到finished_req_ids才恢复。需在实际页释放前加全rank恢复屏障，覆盖正常结束、abort和preemption，并拒绝未审异步/connector/cache共享路径。KV两个owner通过、实际HTTP取消与越界请求负例补齐后才能冻结r7.4。

33按负责人批准的五条标准在r7.5候选分支推进；r7.4只带采集器v2。故障库12行见fault_library_coverage_20261124.md，不新增机制。D包仍依赖D0/D1；没有用V2-Lite或A2证据登记V4。

## KV页面释放屏障增量

新增persistent_kv_release.py，只安装到专用EngineCore实例，源码锁定实际0.26的EngineCore、Scheduler和KVCacheManager。执行前通知所有rank实际请求代际与独占full-attention页；free/pop_blocks_for_free之前要求逐rank同序号确认、设备读取完成、字节恢复和该请求租约清零。正常结束、abort和preemption共用实际页释放边界；共享页、异步batch queue、connector、prefix cache和页归属改变明确拒绝。它是资源屏障，不是新的故障生命周期，也不默认启用。

external_kv_engine_plugin.py与external_kv_page_probe.py为隔离实例的只读探测入口，使用任务目录中的私有general-plugin元数据，不改vLLM源码或宿主安装。此探测不写KV；回执中的read_only_no_injection和page_injection_executed=false必须保留。review_kv_page_probe.py独立核对两rank的请求、序号、代际和释放通知，不能把零租约解释为真实注入恢复通过。

本轮定向CPU门禁52项全通过，见r74_kv_incremental_cpu_20261124.json，其中8项页面屏障、4项只读回执审查。重新核对原有414个源码文件无改动。KV真正的writer窗口、消费者值核验、两个owner及取消负例仍待实机完成；当前不冻结r7.4。

实机队列v2-kv-page-20261124-a1等待卡2/5共享锁900秒后以PROJECT_LOCK_WAIT_TIMEOUT退出；precision-liv/fla的独立租约持续续期。NPU用时0、隔离未启动、没有停止外部容器，控制器已退出。见r74_kv_window_blocked_20261124.json。已请求协调释放共享租约，不绕过同一把锁并发运行。
