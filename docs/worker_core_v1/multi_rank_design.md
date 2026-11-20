# 多 rank 普通任务设计：M1 → M4

2026-11-19负责人已批准立即开发，无需再次确认M1。M1只在开发线，成果进入RC3，不并入RC2.x。蓝区目标：124 Ascend910B4/A2双32GiB，既有CANN8.5.1、vLLM/Ascend0.18.0、本地Qwen3-0.6B BF16、TP2、PP1/DP1、串行eager。两卡同时空闲且锁/门禁通过才运行，不抢占。RC2维护优先。CPU双rank替身不算NPU资格。

## 执行器和身份

由vLLM标准TP2多进程/分布式executor拥有Worker和scheduler；组件不创建/重启Worker、不调用model.forward、不修改调度器。先通过现有`collective_rpc(demoncat_worker_identity)`发现全部rank，必须完整且无重复，不能把列表顺序当rank。固定每个rank的pid、process_start、worker_start、model_instance、逻辑设备和可见设备映射；另在实际RPC与模型回调记录thread身份。进程内对象身份不跨进程比较；物理设备由部署映射核对。

第一版宿主协调器复用单rank `WorkerFaultClient` 的票据/丢回复/清理状态机。每一阶段按rank发起受现有目标过滤保护的collective_rpc：目标rank处理，其他rank回`RPC_NOT_TARGETED`。同一阶段可能需要TP次广播，**不并发调用executor**；全部rank完成阶段后才向正常scheduler放行真实推理。这不同于所有rank同时注入。以后合并为单次带rank信封的RPC属于独立优化，不能改变票据校验。

真实0.18 TP2的executor类型、PID/thread和collective_rpc返回集合必须由实测记录，设计不将TP1 in-process executor的结果推广至多进程。RPC超时必须有界；某个进程丢失时保持全局失败/未确认所有权，不能强行认为已恢复。

## 生命周期、所有权与失败

一个group owner与group generation描述整个任务。各rank的实际Runtime票据仍由各自Runtime发放，保留`rank -> local_generation/attachment/lease`映射；不能把计数器强行重置以制造相同数字。对外group generation一致，证据同时显示所有local generation。所有rank共享group owner，独立attachment nonce、model身份和Native租约。

prepare→register→check→enable必须全rank成功；任意rank失败则任务整体失败，禁止advance/F及部分成功发布。已经接受动作而丢失回复的rank按不确定所有权处理，不能重试prepare/F。恢复遍历所有已触及rank，某个rank失败不阻止尝试其余rank清理；仍不能detach掩盖未恢复的rank。

每个作用域恰好一次F机会；第二作用域必须所有rankOFF/health/释放均确认后，以新owner/provider在**同一组Worker**创建。不能重启替代恢复。宿主执行成功请求计划后仍需finish、unregister、detach及全rank资源读回。

## 单rank自动选点与守卫

用户FaultConfig只选择机制/版本，不增加rank/card/layer参数。部署侧typed group binding拥有固定成员列表与角色状态，和普通TP1 binding分开准入。

- 18：副本RMSNorm正常实际返回及消费者在各rank确认后，默认rank0为目标。只允许其一次替换；其他rank对对应实际调用记录原返回身份/无写入，不能走F补丁。缺少rank0候选则拒绝，不悄悄改目标。
- 39：各rank正常请求形成实际分片投影候选；部署策略从已满足Native/消费者约束的实际候选选一个确定性目标rank（rank升序，再本地真实调用顺序）。记录分片范围、实际返回、消费者。其他rank在F阶段只守卫，不创建故障写入Native租约。正常校准阶段，各rank仍使用mode0无损Native租约核验实际边界；这不等同于其他rank完全没有Native调用。
- 22：按真实各rank的Q/K head与RoPE调用选点，禁止套TP1 head形状；只在一个目标rank施加行错配，其他rank保持原实现。
- Batch、41为第二批，不能因前三项通过自动加入。Batch需要B8同一请求集合和rank本地Q/scale映射；41需要rank本地KV/head范围、slot映射及恢复。

角色只能在normal实际证据齐全后、F尚未advance时由宿主一致确认；不能在rank内部各自按“第一个”独立决定。角色票据绑定group owner/generation、完整成员身份、目标调用和normal证据。未接通此合同的provider **不得启用TP2**。

## 效果、传播、恢复的分层证据

1. 目标rank本地：实际目标输入、合法性、命中一次、改前/改后、同一消费者，归属rank与真实内部request ID。
2. 非目标rank：对应边界无注入/无写入；其下游值允许因真实通信改变，不能要求整个F forward与baseline相同。
3. 跨rank：在实际通信调用的原生返回处记录操作、请求、rank与前后摘要，区分all-reduce前后变化。仅看到全rank最终值变化不唯一证明all-reduce归因；无法观测项保留缺失，不能以命中日志替代。
4. 驱动rank：完整hidden/logits精确哈希/差异摘要；非驱动rank没有logits时明确null，不伪造rank0副本。文本/Top1改变不是通过条件。
5. OFF与无故障机制health：每rank完整本地hidden及该rank实际存在的logits与自身baseline精确相等；各rank Native lease、hook、borrow、inflight归零。输出不同rank的分片不能横向误比。

故障后失败时各rank统一进入恢复协议：目标恢复自身可写资源，守卫rank保持同步请求阶段且不发生写入；所有rank共同执行必要OFF/health，避免一部分rank退出而另一部分进入通信。RPC清理与实际模型恢复分开，不能单rank触发模型。

## 请求映射与宿主循环

沿同一个request_plan：18/39/22为normal→fault→off→health。各rank接收同一真实enqueue内部ID，实际调度回调核验该ID、token/position、阶段和输入指纹。只有所有rank完成、next_phase一致才允许下一轮；缺rank、重复rank、漂移或phase分歧均整体失败。

调用仍是enqueue→advance(内部ID)→等待正常scheduler完成→query，不让宿主业务循环关心TP数。query/create的计划同源；group汇总单列rank结果，不以rank0成功代替全组。组complete表示全部rank阶段结束，资源清理仍单独核验。超过计划上限拒绝，失败不补跑F。

## 准入、测试与实施切片

组合登记新增TP2和每rank设备/源码/模型/Native构建身份；TP1 BLUE记录或黄区许可不扩权。真实新栈0.26、A3、Qwen3.8另行登记；负责人已确认最终A3环境，回传A2推断需澄清芯片名原值。M4 TP8+EP需新增专家路由/归属、跨rank恢复设计，黄区运行材料未到前不开验收。

|切片|开发与CPU检查|设备验收|
|---|---|---|
|M1.1 组宿主/票据|完整rank集合、共有owner映射、阶段屏障、任一rank失败/丢回复、全rank恢复、双作用域|0.18真实executor身份和调度映射；尚未通过|
|M1.2 18|真实副本候选、rank0一次F、其他rank守卫|TP2同Worker组两作用域，跨rank传播和全rank恢复|
|M1.3 39/22|分片/head语义、Native角色隔离、实际候选一致选择|逐机制两作用域；不继承18结果|
|M1.4 Batch/41|B8映射及KV分片范围|在前三项和资源门禁通过后单独验收|

M2仍依赖0.26流式空间与官方驱动门禁、社区W8A8权重及磁盘预算；M3编写A3 TP1/TP2步骤并审回传，正式范围另由负责人确认；M4蓝区不做DeepSeek实测。

当前状态：设计完成，M1.1开始开发；所有TP2机制公共执行仍关闭。最近124只读检查卡2/5无计算进程，但Kubernetes分配释放未证明，不能启动NPU。28降为P3，未启动SDK构建，NPU累计0秒；中断前只创建了本地SDK子检查脚本，未执行容器或设备调用。

## M1.1实施检查点（2026-11-19夜）

已增加开发线`MultiRankTaskClient`，复用原Runtime/Host票据。CPU使用真实Runtime加双rank替身，验证倒序RPC结果按身份归属、共同owner及各rank generation、同Worker两作用域、请求ID一致、角色缺失/双注入者/输入指纹不一致拒绝、丢回复不重放、rank漂移拒绝、恢复遍历所有rank。失败后仅显式recover允许共同OFF/health，原任务仍保留失败，不包装成通过。新增TP2只读前提报告要求实际TP group；旧TP1门禁和公共配置没有放宽。

**尚未接入真实TP2 provider角色协议，也未运行TP2 NPU**。目前group只是开发协调层；18/39/22继续因缺少真实group binding/部署准入保持关闭，不能把CPU替身的角色证明当实际命中。后续先补18的非目标rank观察、nullable logits/真实completion、全rank失败后对齐恢复，再补39/22。

已在现有0.18镜像内运行无设备、无网络、只读源码检查，未import torch/vLLM或创建Worker。安装源码SHA见[m1_executor_source_inspection_20261119.json](../../experiments/worker_core_v11_20261118/m1_executor_source_inspection_20261119.json)。MultiprocExecutor.collective_rpc在不指定unique_reply_rank时读取所有response队列；WorkerProc通过标准多进程启动，driver为rank % TP==0。实际多进程executor与线程身份仍需NPU启动后的读回。

安装的LogitsProcessor可以按平台选择gather或all-gather；gather在非driver rank可返回None。现有Rms18TaskRequest的TP1日志完成钩子要求非空完整logits，因此不能直接把TP门禁改为2。新适配器须核对实际gather策略，非driver只确认真实完成并保留null；不能制造logits副本，也不能只在root结束时提前宣称整个请求完成。

最新卡2/5只读检查无NPU计算进程，但Kubernetes设备分配仍保留，未证明释放；未抢占、未启动设备调用。M1源代码检查容器已退出。M1源码及接口不进入RC2.2冻结包，RC2.2需从其独立单卡提交集构建。


角色提交接口为受信任部署回调`commit_roles(decision)`：协调器依据所有rank同一normal请求、成员指纹、输入指纹和真实候选选择单rank，再提交角色并查询全部rank确认；回调丢回复不重试、不放行F。该接口还需真实provider接线，不能由用户配置提供rank或伪造normal证据。CPU同时覆盖RMS缺少rank0时拒绝，以及分片投影rank0不合格时选择实际合格rank1。


## 2026-11-20 实机状态更新

上面的11月19日记录为历史检查点。现已在BLUE组合（Ascend910B4/A2、CANN8.5.1、vLLM-Ascend0.18、Qwen3-0.6B BF16、TP2）完成18、22、39的真实标准多进程执行及同Worker组两个作用域。每项normal→fault→off→health共8请求；自动选rank0，仅目标rank施加故障，rank1核对原始边界，两个rank均完成数值恢复与资源清零。

具体原件哈希、源码清单与独立审查见[npu_progress_20261120.md](npu_progress_20261120.md)。本轮只接受实际`use_all_gather=True`路径，尚未适配非driver返回null logits的gather路径。39保存组合hidden/logits哈希及单独logits哈希，不能据此证明hidden单项变化；未直接观测collective边界，不能唯一归因到all-reduce。全部结果仅限BLUE组合，不变更公开部署登记或冻结发行包。

显式全rank recover补测发现：恢复health结束后，旧index仍指向off，导致本地complete与next_phase矛盾。开发线修正恢复完成时的计划索引；CPU新增终态回归，实机以新attempt重验，并保留旧失败原件。显式recover测试不代替真实Worker崩溃/通信故障验收。

Batch和41的TP2仍未接通：Batch当前身份和候选明确绑定TP1，B8请求行与rank本地Q/scale合同尚未扩展；41的writer/slot/consumer及health仍走TP1合同，缺少单rank注入与peer守卫接线。这些属于开发缺口，不是待重跑即可通过。20/26/33仍按普通入口适配预判与止损规则执行。28真实TND研究结果和失败限时另见专门记录，未授公共执行资格。

三项显式recover的实机终态已独立复核通过；源码核对见[TP2运行文件身份](../../experiments/worker_core_v11_20261118/m1_tp2_runtime_sources_20261120.json)，11份待提交运行/probe文件与实测快照逐项SHA一致。Batch的TP1完整health缺口随后补齐，未改变TP2待适配状态。
