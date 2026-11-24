# 20：真实输入内容覆盖，公共持续验收

native20-a2 在124 A2 Ascend910B4、CANN9.1、vLLM0.26/vllm-ascend0.26rc1、Qwen3.8社区W8A8、TP2通过公共入口两个作用域。仅登记开发线 BLUE_COMBINATION；冻结r3保持不变，A3、TP1和其他模型不继承资格。

## 真实链路

自动发现真实decoder下的304个W8A8投影，绑定当前请求的真实residual作为活跃源。在投影的实际 DynamicQuant 调用前，用 Native owned-object copy 把源最后一行的前64个BF16元素写入投影输入最后一行的前64个元素。

源与目的存储必须不同、行数一致、BF16连续基础格式、同stream且有强引用与独立租约。原生 DynamicQuant 真正读取覆盖后的输入，输出新的Q和scale；消费者完成事件之后立即恢复输入字节。随后原生 QMM 消费这同一组Q/scale对象，记录真实调用完成事件。没有重写QMM输出、没有替换执行器地址槽，也没有制造邻接。交付名称为“真实内存故障：输入内容覆盖”。

每次调用都新建区间租约，恢复前核对对象身份、版本、预期当前字节；出现正常原地修改、跨stream、共享未审查消费窗口时拒绝旧快照恢复。当前仅支持已审查单请求串行eager路径，不支持B8共享输入、MLA、其他量化方法或并发读写。

## 公共验收

|作用域|每rank observed/eligible/attempted/applied/changed/consumed/restored|failed|fault/normal|
|---|---:|---:|---:|
|0|均1216|0|2.368691×|
|1|均1216|0|2.395690×|

304个真实站点×4次forward，包含prefill和三次decode。两作用域复用同组Worker。hidden/logits均有变化；OFF/health新请求的完整四步hidden/logits精确恢复，租约、hook、owner、在途请求和许可清零。缺许可创建及阶段耗尽后的advance负向检查通过。不要求token改变。

首轮native20-a1因 `ORDINARY_EVIDENCE_LIMIT_EXCEEDED` 失败，未获得验收资格。诊断已有1216次调用，但没有完成全部四阶段和两个作用域。修复只调整证据表达：每次调用仍检查源值=写后值、原值=恢复值、消费者完成、QMM实际消费；任何失败直接拒绝。按实际站点汇总全部计数，附完整有序region列表哈希，不提高512KiB上限、不截断失败记录。公共回执不再保留每次调用的全部SHA对，因此哈希不能替代对未知原始列表的重新计算；独立复核依据是受审查采集器、站点完整计数和真实回执。该边界明确保留。

## 窗口与证据

卡2/5采用最多35分钟的有界按设备Docker stop清理器。两窗口均结束恢复派发能力，实际停止外部容器0个；未改Docker入口、驱动、CANN或调度器配置。

|attempt|UTC开始|UTC结束|窗口耗时|结果|
|---|---|---|---:|---|
|a1|07:46:25.880324|07:50:30.285339|244.405044秒|证据容量失败，隔离已解除|
|a2|07:53:23.279175|08:00:12.648051|409.368908秒|公共两作用域及独立复核通过，隔离已解除|

时间为2026-11-22验证窗口墙钟，含启动/加载/清理，不是纯kernel时间。第三方vLLM关闭时force-process和shared_memory警告保留，不把注入资源归零说成整个运行时无警告。

实测源码manifest `da8cf6e877cf8a60c2d603ba54fb3164e04b466efc5896ac7ab6f29f44fb02e5`；自有Native库SHA `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。原始回执留蓝区及本地私有暂存，不提交完整张量。

[独立回执复核](../../experiments/worker_core_v11_20261118/continuous_public_native20_review_20261122.json)、[源码审查](../../experiments/worker_core_v11_20261118/continuous_native20_source_review_20261122.json)、[CPU回归](native_cpu_review_20261122.md)和[A3独立构建准备](native_regions_a3_build_20261122.md)分别记录。公共安装函数支持该机制不等于任意部署获准；精确模型、软件、TP、Native SHA和许可仍需匹配。
