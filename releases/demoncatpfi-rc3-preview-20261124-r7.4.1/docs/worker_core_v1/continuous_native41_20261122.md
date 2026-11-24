# 41：Native KV 区域公共持续验收

2026-11-22，native41-a1 通过独立回执和源码审查。仅登记 BLUE_COMBINATION：124 A2 Ascend910B4、CANN9.1、vLLM0.26/vllm-ascend0.26rc1、Qwen3.8 社区 W8A8、TP2。r3 冻结物不改；此能力进入后续候选，A3/TP1 仍需独立验收。

## 真实调用与恢复

`install_continuous_binding` → 既有 MultiRankTaskClient/WorkerFaultRuntime → `NativeKV41Site` → 实际 `npu_scatter_pa_kv_cache` → 当前 decode V slot 尾两列/每个 KV head → `npu_fused_infer_attention_score`。Native ACL 在活跃原存储中写零，未替换 FIA 输出。属于真实内存区域改写，不宣称原 kernel 执行越界或存在分配器邻接。

site 核对实际 writer 的 cache 对象、slot 和写入值；consumer 核对同一 cache 存储、page table、请求、token 位置与真实 API 参数。对象层保留 Tensor/Storage 强引用、owner/generation、线程、设备、区间和 stream。Native 在同 stream 同步后备份/写入，FIA 返回后记录并等待同 stream 事件，再恢复原字节。恢复检查当前字节仍等于预期故障值，遇到合法原地更新或存储变化拒绝盲写回。

首次实现支持可证明的单请求 decode 窗口。每个 rank 的16个 full-attention层各执行一次 prefill 和三次 decode：64次 observed，16次直接 K/V prefill 为不适用，48次 decode eligible。线性/GDN层不伪造 FIA 调用。公开配置不接受裸地址、库文件、任意层名；Native路径和SHA由可信 binding提供并进入部署身份。

## 实测结果

|作用域|每rank observed|eligible/attempted/applied/changed/consumed/restored|failed|fault/normal耗时|
|---|---:|---:|---:|---:|
|0|64|均48|0|1.017253×|
|1|64|均48|0|1.043745×|

两作用域使用同组 Worker；两rank的 hidden/logits 都传播变化。每阶段四token，最多六次预热、连续两次有限且精确一致；OFF/health 完整四步 hidden/logits 与 normal 精确一致，区域恢复 SHA 一致，租约/hook/owner/在途/许可清零。无许可创建、阶段耗尽再 advance 的负向检查通过。没有要求生成 token 改变。

窗口 UTC 07:26:18.239595–07:32:21.588032，共363.348470秒（含加载、预热、关闭），卡2/5。使用既有35分钟有界精确设备映射清理器，本窗口停止外部容器0个；结束 `isolation_removed=true`，派发能力恢复。未改Docker入口/驱动/CANN。vLLM关闭时仍有一个强制终止进程和 shared_memory 清理警告，保留于私有process.log；不能把注入资源清零表述为第三方关闭完全无警告。

## 身份与复核

- NPU源码manifest：`9319113eda5ef0586cf5d72aa6aeab50610b827241c5e5eec00a67154a8dcede`。
- Native库：`d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`，29088字节；CANN9.1、g++11.4，自有构建收据。
- [独立回执复核](../../experiments/worker_core_v11_20261118/continuous_public_native41_review_20261122.json)、[源码审查](../../experiments/worker_core_v11_20261118/continuous_native41_source_review_20261122.json)、[构建记录](../../experiments/worker_core_v11_20261118/native_regions_build_20261122.json)。原始Runtime回执留在蓝区及本地私有暂存；不提交完整张量或原始统计数据。
- 本轮54项相关Python测试通过，另有3个真实SDK编译的CPU ACL替身测试通过。后者不是NPU证据；实机结论来自本窗口。
- [A3构建准备](native_regions_a3_build_20261122.md)提供独立源码构建工具、收据格式和审核要求，A3产物须另审查，不能沿用A2二进制资格。

限制：当前prompt/纯文本串行eager、BF16 base格式、单请求decode、当前V slot尾列；共享prefix、跨stream、MLA latent和其他KV布局未验收。26 FIA和20不继承本次41的消费者或恢复证明。
