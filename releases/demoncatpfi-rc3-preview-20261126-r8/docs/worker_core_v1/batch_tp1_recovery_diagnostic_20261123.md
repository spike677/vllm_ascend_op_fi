# A3 TP1 Batch关闭后金丝雀差异：诊断执行单

对象：r6、A3、v0.26.0rc1-a3/CANN9.1.0、现役Qwen3.8 W8A8、TP1。仅由负责人转交黄区，在专用实例执行；本页没有声称已在A3运行。保留失败进程和原回执，不在冻结r6内打补丁。

## 已知与静态结论

黄区报告开启前6次正常金丝雀相同，关闭后8个prompt第1步hidden全部变化，仍全有限。现有报告没有关闭阶段每条Q/scale恢复事件、完整cleanup、前后实际batch轨迹，不能定根因。

源码 [BatchQuantSite](../../fault_injection/sites/batch_quant.py) 针对独立DynamicQuant返回的Q与scale克隆备份，改写后交给原QMM；同stream事件完成后在finally恢复并核对before/restored SHA，close撤销quant/QMM包装与各scheme.apply覆盖。此torch变体没有使用Native共享scale状态。正常路径的权重/scale持久状态原则上不应被该代码写入，但必须核对实际安装的函数指纹和故障回执，不能凭静态代码替真实恢复证明。

[ProductClient.disable](../../src/demoncat/runtime/vllm_ascend/worker_extension/product_host.py) 依次drain、disable、收集请求、unregister/detach，再跑新金丝雀；故障后canary不相同则保留失败，不允许直接detach伪装完成。[AutomaticBatches](../../src/demoncat/runtime/vllm_ascend/worker_extension/product_batch.py) 按prompt归并金丝雀输出，但比较数据主要是hidden/logits哈希和有限性，没有把完整每步调度轨迹纳入金丝雀等价条件。多个HTTP请求是否在同一个decode批次，不能靠并发提交数量推断。

## 每一步的记录合同

不使用虚构的CLI参数。黄区在现有可信session/Worker扩展中调用下列已存在接口，诊断观察独立保存，不修改注入算子或准入。原件留黄区，文本按scope/rank分页≤4KB。

|步骤|动作|必须记录|
|---|---|---|
|D0 冻结原失败证据|先保存失败回执和 `client.status()`；不重新enable|包/工具/许可SHA、模型身份、Worker PID与rank在私有原件中、错误阶段；enabled、cleanup、recovery分别保留，不能把host enabled=false等同于Worker无残留|
|D1 关闭状态复核|读取已存在cleanup里的每rank disable/unregister/detach回复；检查quant/QMM和scheme.apply函数身份|registered/enabled、owned_hooks/方法补丁、Native句柄（torch变体应无此依赖）、leases/inflight/owners；每个字段实际值或null，缺字段不能填0|
|D2 字节恢复|审查全部合格调用的before/restored SHA、region/对象身份、consumed/restored和failed|按真实batch帧去重，eligible=attempted=applied=consumed=restored、failed=0；changed可更小。任一不匹配先定恢复缺陷，不进入“仅批次变化”结论|
|D3 原进程关闭后两次金丝雀|确认读取路径已关闭注入后，同一8个prompt、顺序、token前缀、seed/greedy、并发提交方式，各完整执行一次|每轮每step actual req_ids（原件）、prompt/token SHA、computed/scheduled tokens、输入行顺序、有效batch大小、prefill/decode组成、cache组/slot关系；host提交并发与实际批次分开|
|D4 无注入对照进程|另起同镜像/权重/参数专用实例，仅跑正常路径；保留原失败进程|同样预热条件下的一次金丝雀基线及完整调度轨迹；新进程结果仅辅助诊断，不能替代原进程OFF/health恢复|
|D5 比较|按同prompt、同已知前缀、同step对齐，逐一对比开启前、关闭后第1/2次、新进程|完整hidden/logits本地比较：shape/dtype、有限性、差异元素数、首个差异索引、B/F具体值、max_abs；同时比较对应实际输入/批次，而非只比token或整体hash|

金丝雀运行可复用部署配置里的prompt列表和现有真实HTTP发送函数；**不能用reset cache、清零GDN状态、重建模型或修改batch来让失败消失后计通过**。诊断时若为了区分批次影响另加固定同批对照，单独命名，不替换原失败样本。完整数组只在黄区临时诊断目录保存，文本回传索引/数值摘要与文件SHA。

## 定性规则

1. hook/函数身份/租约或字节恢复不一致：公共实现、黄区接线或其他包装冲突的候选；先根据加载源码和调用顺序归属。公共缺陷确认后从r6发行提交 `ea42a246` 开r6.1最小修复，新增标签/包/验证，再同步开发线。
2. D1/D2全部通过，但同输入前后batch轨迹不同：支持金丝雀调度不等价假设，尚不能单凭相关性结案；需无故障控制复现同一轨迹变化及数值差异，才修订金丝雀规范。
3. batch和输入一致、D1/D2通过而原进程关闭后仍漂移，新进程稳定：支持跨请求状态候选（KV/SSM等），不等于已定位GDN kernel。不重启代替恢复、不继续第三方kernel深挖。
4. 新进程无故障也不稳定：环境/基线问题，当前组合不授精确恢复资格；不降低容差或用DeepSeek噪声标准替代Qwen标准。

当前结论：**恢复未证明、根因未定；不开修复分支、不补PASS、不优化耗时。** 返回资料须包括D0–D5可用字段与缺项，以及缺许可/未知profile负向结果。


## r6.1维护后的执行决策

负责人已决定D4/D5由r6.1受控复测替代：旧r6缺调度轨迹，保留原因未定，不能补造历史调度。r6.1采用固定顺序串行B1金丝雀，注入前连续3轮、关闭后2轮精确核对，真实用户Batch仍并发。维护发行提交90b0f21b，详见GitHub同提交的 releases/RC3-PREVIEW-R6.1-HANDOFF-20261123.md。

缺许可测试里的ACTUAL_CALL_NOT_YET_OBSERVED是actual_call层的预期待观测项，prepare判断排除这一层；本身不污染缺许可负例。其他部署、源码、TP、profile和binding必须先匹配，否则分别记录，不能混称纯缺许可。
