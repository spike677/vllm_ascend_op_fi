# 2026-11-21 夜间推进汇总

此报告只登记蓝区原件审查。硬件均为124 A2 Ascend910B4；软件栈为CANN9.1.0、vLLM0.26.0、vllm-ascend0.26.0rc1、torch2.10+cpu、torch_npu2.10.post4；模型为社区Qwen3.8 W8A8或DeepSeek-V2-Lite-W8A8，TP2、eager。没有TP1或V4.1 NPU运行，不外推A3。

## 开发同步与发行

开发线 `codex/product-integration-20261119` 已同步e281ec32及其前序。已冻结r2不变，未冻结r3。新通过的能力仅写开发候选和BLUE_COMBINATION；r2仍只含当时审查的28/Batch。当前未收到新的r2黄区回传；没有自行联系或发送黄区任务。

## 公共入口实机

所有通过项均用同组Worker两个独立owner作用域、每请求prefill+3次decode、全rank持续匹配调用、真实内部ID、实际consumer、区域字节恢复、全新OFF/health完整hidden/logits精确恢复和资源清零。未把实验binding或旧单次结果提升为公共验收。

|模型/能力|窗口|每rank每作用域 applied / changed / failed|fault/normal 两作用域|范围说明|
|---|---|---|---|---|
|Qwen39 output_proxy|a12|1216 / 1216 / 0|2.42345× / 2.43357×|304真实BF16 ND投影；上游nonfinite1208与already-positive-Inf0分开|
|Qwen18|a14|644 / 644 / 0|1.88848× / 1.84664×|129模块+32融合Q/K输出；真实消费者|
|Qwen22|a15|64 / 64 / 0|1.16110× / 1.09448×|16融合mRoPE层；672 rotary行改变|
|V2-Lite28 MLA|a16|108 / 108 / 0|2.02685× / 1.82329×|27稠密MLA层；不是V4.1稀疏/EP|
|QwenBatch forward核对|a17|1216 / 1214或1213 / 0|5.93327× / 5.97488×|参考样本数不变，scale精确、Q差0；默认仍immediate|
|V2-Lite8|a20|104 / 26 / 0|1.58518× / 1.49855×|26MoE；78单组decode应用后值不变；当前路由→MLP→原生combine|
|V2-Lite18|a21|220 / 220 / 0|1.72329× / 1.65174×|55模块；27融合KV内部norm未覆盖|
|V2-Lite22|a22|108 / 108 / 0|1.06779× / 1.06119×|351 rotary行；真实Q和融合KV两消费者|
|V2-Lite39|a25|332 / 332 / 0|1.68125× / 1.75879×|observed463、eligible332；131输出不适用，52结构候选无forward|

性能是带观察器的fault/normal比，不是相对无观察器服务的生产开销。详细回执SHA、源码清单SHA和独立审查见 `experiments/worker_core_v11_20261118/continuous_public_*_review_20261121.json`；不提交完整张量或新statistics原件。

## 窗口与隔离

逐窗口绝对UTC时间、持续秒数和隔离开始/结束原记录见 `experiments/worker_core_v11_20261118/night_windows_20261121.json`。a7–a25共19个窗口全部结束，合计5101.944秒（85.032分钟），含加载、预热、退出，不伪称纯kernel执行时间。隔离采用最多2100秒的卡2/5精确设备清理器；没有改Docker入口、驱动、CANN或派发配置。19个窗口全部解除隔离，没有外部容器停止事件。失败窗口清理不冒充同进程恢复。

通过回执中的资源0指故障框架的租约、hook、owner和在途请求。vLLM关闭阶段仍出现Worker退出超时/共享内存resource_tracker警告，容器由controller收尾；不声称整个vLLM进程不存在退出问题，也不使用容器销毁代替此前已完成的同Worker OFF/health检查。

调试失败保留：a7/a8结构候选与真实调用计数区分；a9–a11补真实消费者复制语义；a13被patch后的融合forward身份识别；a18旧MoE接口漏选；a19 A2实际abs派生索引；a23投影维度限制；a24 MLA内部消费者被外层模式屏蔽。通过结果均来自后续独立窗口，不删除失败证据。

## CPU工作与限制

- DeepSeek残留静态分析完成，见 `dsv41_state_residual_analysis.md`。官方0.32rc1有-1/可见长度过滤、新块清理和Engram ready/stream同步，不能直接断言缺mask或缺清零。仍需黄区证据区分合法范围内旧索引、实际块清理范围、原生kernel和host offload顺序。
- D1/D2/D3仅设计：长prompt不能保证前面query超过top-k；位置对齐双进程需把fault导致的状态残留视为恢复失败；逐阶段新进程不证明同Worker恢复。待负责人结合黄区结果选择，未修改第三方代码。
- 历史210个失败事件（184个不同ID）逐项清单：8废弃、5环境、197待修复/复测。110条有具体拒绝代码，所有测试文件已定位。未把197条都认定为夹具问题，也未宣称全库已绿；完整RC3冻结前仍需修复相关欠账。
- 新持续框架及site相关37项CPU测试全部通过（含多维strided投影和嵌套消费者单次恢复）。小型回执 `night_cpu_review_20261121.json` 附代码SHA；它们不替代历史全库回归。
- Batch前向合并参考核对已通过NPU：fault绝对秒数由9.390/9.277变为8.815/8.689，观察到约6%下降；非交错重复性能试验，不给稳定收益结论。逐调用租约/指纹/设备消费仍保留，scale精确和Q边界±1标准不变。
- Native准备文档 `memory_fault_consolidation.md` 已写明41、26fia、20的活跃对象范围、同stream顺序和不得覆盖合法原地更新的恢复点，尚未做新Native持续验收。

## 未闭合项

39 matmul_range的新栈量化/非量化参考重算site、33的paged合法归约site尚未完成公共持续验收，不使用旧固定栈Native结果代替。17持续视觉范围、41/26fia/20 Native阶段、V4.1稀疏MLA/EP实际接线也未完成。12项总表见 `continuous_progress_20261121.md`，A3两列仍未验收。本报告不是12项全部完成或RC3可冻结的结论。
