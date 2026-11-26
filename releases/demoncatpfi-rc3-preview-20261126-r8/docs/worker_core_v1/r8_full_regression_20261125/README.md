# r8当前候选 A2 全项回归

组合：A2 Ascend910B4、vLLM0.26 / vllm-ascend0.26rc1 / CANN9.1、社区Qwen3.8-27B-W8A8、TP2（卡2/5）。实测源码3d4c9e39，清单`b87aced01c6bc4ad438f03118cb933e18d5ae0b02ab39962cdc5726318079c0e`。每项独立服务，但同项两个owner保持同组Worker；全rank持续触发。

结果仅为当前共享产品路径的BLUE回归，不是A3登记或黄区CLI验收。通过项均另审原始回执、实际消费者、覆盖计数、区域字节/资源/精确金丝雀恢复及范围负例。Batch按唯一调用计数，不把每个请求参与次数相加冒充实际调用。

当前完成16/16，独立审查通过16项。等待或运行中的项不借历史结果。耗时为模型启动、请求、恢复和退出总用时，不是kernel计时或性能倍数。

|用例|当前结果|窗口用时/秒|两个owner合计applied（rank0 / rank1）|独立审查|
|---|---|---:|---:|---|
|18api|通过|288.825|3864 / 3864|[记录](r8full1125a1-18api-review.json)|
|18over|通过|283.698|3864 / 3864|[记录](r8full1125a1-18over-review.json)|
|18under|通过|281.443|3864 / 3864|[记录](r8full1125a1-18under-review.json)|
|22|通过|262.034|384 / 384|[记录](r8full1125a1-22-review.json)|
|39op|通过|288.098|7296 / 7296|[记录](r8full1125a1-39op-review.json)|
|39mr|通过|288.024|4224 / 4224|[记录](r8full1125a1-39mr-review.json)|
|28|通过|261.568|384 / 384|[记录](r8full1125a1-28-review.json)|
|17|通过|325.483|6 / 6|[记录](r8full1125a1-17-review.json)|
|batch|通过|437.729|4864 / 4864|[记录](r8batch1125r1-review.json)|
|15|通过|257.267|384 / 384|[记录](r8full1125a1-15-review.json)|
|41|通过|255.970|288 / 288|[记录](r8full1125a1-41-review.json)|
|26fia|通过|255.807|384 / 384|[记录](r8full1125a1-26fia-review.json)|
|20|通过|287.630|7296 / 7296|[记录](r8full1125a1-20-review.json)|
|externalmod|通过|286.573|7296 / 7296|[记录](r8full1125a1-externalmod-review.json)|
|externaldq|通过|285.500|7296 / 7296|[记录](r8full1125a1-externaldq-review.json)|
|externalvision|通过|325.008|6 / 6|[记录](r8full1125a1-externalvision-review.json)|

Batch首轮在scope1 enable时API收到终止信号，失败原件保留；同源码完整复测通过，表内使用复测的完整双owner记录，不拼接两轮作用域。详见batch-first-attempt-analysis.json。

changed小于applied表示部分写入值相同，不自动判失败；failed必须为0，eligible必须与attempted/applied/consumed/restored一致。18方差档按各自命名变体验收，不能合并成一次18结果。

18.api_half额外检查缺许可、未知profile、未知机制的结构化exit4及拒绝后健康正对照。Batch额外检查非串行金丝雀拒绝后引擎存活，再完成串行金丝雀正例。Native三项保留真实库SHA、消费者与恢复证据。外部module/dispatch同时核对原始逐调用记录，不只看汇总计数。

抢占：本轮负责人允许卡2/5持续保护，包括改名任务。仅停止实际独占这两卡的非项目容器；共享其它卡的容器保留。每批结束停止监控，不改派发器配置、不删除外部容器或数据。完整隔离审计在窗口结束后补入，运行中不宣称已解除。

CPU首轮1360项/1358通过/2项带理由跳过。发行规范字节的全量CPU及最终安装验证另附；当前记录不代替最后一次冻结门禁。

安装后候选身份验收也已通过（[记录](r8canon1125a1-review.json)）；据其实际运行身份刷新BLUE 18.api_half登记。更新候选81643582的三个wheel无设备安装检查通过，406个文件逐字节一致（[安装检查](registered-candidate-install-check.json)）。与前候选唯一运行时差异为登记JSON，所有Python/机制字节不变（[对比](candidate-a2-runtime-comparison.json)）。免许可最终实测通过（[记录](r8canon1125r1-review.json)）：18.api_half同组Worker两owner、无临时许可；未登记22负例exit4、引擎健康；未知机制及未鉴权helper负例exit4；区域/资源/精确金丝雀恢复通过。

五个验证窗口累计占卡墙钟时间5693.097秒（含启动、失败、恢复和完整重跑，非kernel时间），成功停止仅占卡2/5的外部容器25次。所有窗口均已结束、监控已撤销；[最终检查](isolation-final-check.json)确认无本轮控制器或测试容器存活。窗口内隔离开始/结束绝对时间、被停容器名和镜像均保留在[审计](isolation-audit.json)。没有修改派发器配置、Docker入口、驱动或CANN，也未删除外部数据。
