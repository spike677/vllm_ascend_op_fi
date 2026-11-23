# 故障15：阶段0/1普查与档位决策（B档已批准）

## 结论

当前不能宣称找到A档的自然私有格式写入链；也不能把已观察到的调度索引缓冲区当作合格的模型数值写入点。负责人已批准B档及下述计数口径，进入阶段2：先验证预分配attention输出的真实写入边界，再实现注入；自然数值链不合格则拒绝该点。**截至本份阶段0/1报告，没有注入15、没有新增产品准入，也没有改动权重stride。**

A档要求私有格式、自然写入与消费链及格式转换独立存储的证据同时成立。B档必须标为“别名断开语义（ND视图），不含私有格式触发条件”。当前普查没有证明隐式格式副本，更没有证明A档不存在于整个模型/所有请求。

## 阶段0：读取实际源码后的候选清单

Qwen依据本次运行镜像内源码，四个文件SHA见[实际源码清单](alias15_census_20261123/actual_source_manifest.json)。DeepSeek依据已保存的官方vllm-ascend 0.32源码，提交`24487f4f48ee99cf45d86d1c504e645549b8c1ef`，仅静态研究，不代表A3运行路径。

|文件/函数|base与生命周期|视图/写入|后续消费者|格式和结论|
|---|---|---|---|---|
|vllm layers/attention/attention.py，Attention.forward，531–582|torch.empty分配的每forward输出|output.view；unified_attention_with_output带可写output参数|返回output.view后真实o_proj|逻辑ND候选；融合内写入不可由本次dispatch普查确认；优先补强此边界|
|vllm_ascend/attention/attention_v1.py，forward输出赋值，1665/1671|上层传入output|output[:num_tokens] = attn_output|返回给Attention/o_proj|源码成立，实际命中分支和写后消费者未连通|
|ops/gdn.py，forward 119–146|core_attn_out为本次forward零缓冲；可选output来自调用者|自定义qwen_gdn_attention_core写core缓冲；可选output切片赋值|norm、out_proj|Qwen3NextDecoderLayer实际不传可选output，不能把该切片写入当作已发生；kernel内部指针写不等于已证明视图写|
|ops/gdn.py，forward_core状态更新245/265、450/451|持久SSM/conv缓存|切片赋值、index_copy_；部分为speculative分支|后续token状态计算|持久状态恢复复杂；未选为第一目标，无NZ证据|
|attention_v1.py，reshape_and_cache；_nz_5d_view 1775|持久KV缓存|view5D + DeviceOperator.reshape_and_cache|后续FIA|名称含NZ或5D布局不等于实际format29；内核写入与区间关系仍不透明|
|DeepSeek v41/model.py，mask_output_buffer/lookup 1168、1222–1232|复用workspace/输出缓存|slice、view、copy_、尾区zero_|Engram/model下游；ready_event同步|静态ND候选，跨请求生命周期及实际消费者需A3普查|
|DeepSeek v41/compressor.py，pool_projected 55–68|持久_ring_pooled、state_cache.kv_cache|slice、squeeze交给compressor_from_projected|norm(pooled)、后续cache读|Triton写入不可见；实际存储格式未观测|
|attention/dsa_v41.py，scatter_cache_sk、索引/元数据更新|持久cache、topk和ring元数据|scatter、自定义算子、slice.copy_|SparseFlashMla、索引消费者|候选多为整数元数据，不能混同模型浮点输出链；无私有格式证据|

这是限时源码候选清单，不是覆盖所有内部native/Triton调用的证明。量化权重NZ主要被读取；本次未找到权重自然视图原地写入链。

## 阶段1：124 A2真实请求

组合：Ascend910B4、vLLM/vllm-ascend 0.26、CANN9.1、社区Qwen3.8 W8A8、TP2、eager、纯文本、4生成token。两次未注入基线、两次开启观察、一次移除观察后的请求；不创建替代模型。

|窗口|每rank dispatch|共享存储视图|视图写入|后续读取候选|耗时|
|---|---:|---:|---:|---:|---:|
|a1：严格base对象读取|19428|6284|558|16|255.365秒|
|a2：同存储连续区间重叠读取|19428|6284|558|48|246.159秒|

a2每rank有10个execute迭代，其中8个有调度token（两次prefill及各3步decode），2个为空的清理迭代，不能声称10个token步。视图base格式ND5822、不可观测462（包括CPU张量）。48条候选均为调度/索引数据，int32/int64或CPU数据；模型浮点输出的自然写入消费链未被本工具捕获。各rank hooks=0，execute包装器已移除；前/中/后token及文本一致，但**未核验完整hidden/logits无扰动**。

两窗口合计501.523秒（含启动/清理），不是纯kernel时间；卡2/5有界隔离已解除。日志及回执身份见[普查证据](alias15_census_20261123/alias15-readonly-a2.json)。控制器复用了旧产品窗口模板，`worker/recovery_plan`中的fault阶段字样仅是模板标签，真实执行入口为只读alias15_readonly_probe，没有任何注入。退出时Python resource_tracker有信号量/共享内存清理警告；这是进程退出警告，不能据此将其称为Fault15资源恢复证明。

## 推荐的B档验证边界与恢复语义

优先验证Attention.forward预分配输出→实际impl的output写入→o_proj输入：在自然调用边界确认实际输出对象/存储区间以及设备消费，保留每请求生命周期。**不通过人为增加copy_或把调度索引当激活来凑出链。** 若融合实现完全没有可重定向的真实视图写入，则该点不适用，另选自然候选；本次不绕过此条件。

阶段2应将命中写入改向独立存储，原base保留原值；不是事后篡改消费者结果。关闭后解除改向、释放临时存储/hook；每请求污染随请求结束释放，新请求金丝雀精确恢复。持久cache/状态若不能证明安全恢复点，拒绝注入，不能回写旧快照覆盖正常更新。

计数已获负责人确认：changed为发生数值变化的调用次数，满足changed≤applied；另报changed_elements为原base与应写值不同的元素数。保留现有八项调用计数合同，不混用单位。

黄区只读工具见[执行说明](alias15_census_handoff_20261123.md)。社区out_proj量化与现役BF16不同，A2零命中不能替代A3普查。阶段2先验证Attention.forward预分配output→impl实际写入→o_proj消费；不人为增加copy_，不污染调度索引，不采用无法安全恢复的持久状态。A2 TP2产品验收通过后再出A3 TP1/TP2执行单；15不阻塞r7。
