# 26 FIA destination：真实 residual 写入验收

native26-a1 在124 A2 Ascend910B4、CANN9.1、vLLM0.26/vllm-ascend0.26rc1、Qwen3.8 社区 W8A8、TP2通过公共入口两个作用域。仅开发线 BLUE_COMBINATION；冻结r3不改，不代表A3或DeepSeek。

## 实际边界

按实际 decoder 结构自动找到16个 full-attention层；input_layernorm 返回当前 residual，真实 `npu_fused_infer_attention_score` 产生当前请求最后query的输出。保留实际输出存储，取前64个BF16元素；在同层 post_attention_layernorm 的实际 `_C_ascend::npu_add_rms_norm_bias` 读取前，用 Native owned-object copy 写入 residual 最后一行的前64元素。

源与目的均为真实活跃对象，库只接受由对象层持有的有界handle。未移动张量来制造邻接，没有换算子参数指针，也没有事后改 hidden/logits。此交付称“真实内存故障：FIA数据写入真实residual”，不声称原FIA kernel越界写。

真实消费参数与 residual 存储区间对应，schema不允许写该输入，实际返回的新 residual/输出存储不与旧 residual 别名。消费者后记录同stream事件并等待；Native再次检查故障字节未被正常写入改变，恢复原区域并逐字节核对。若源FIA值中途改变、消费者原地写回、返回值复用旧 residual、布局或存储改变，则拒绝，不用旧快照覆盖合法更新。

## 结果

|作用域|每rank observed/eligible/attempted/applied/changed/consumed/restored|failed|fault/normal|
|---|---:|---:|---:|
|0|均64|0|1.116588×|
|1|均64|0|1.124118×|

覆盖16层×4次forward（prefill加三次decode），两rank持续执行；两作用域同组Worker。实际consumer均为 `_C_ascend::npu_add_rms_norm_bias`。hidden/logits有变化；OFF/health新请求完整四步精确恢复；源复制值与目的故障值SHA相同，目的恢复SHA与原值相同；资源与许可清零，负向检查通过。token改变不是验收条件。

窗口UTC07:38:49.219030–07:44:38.617334，349.398334秒，卡2/5；有界隔离结束恢复，外部停止0个。vLLM关闭阶段的强制清理与shared_memory警告保留；注入资源清零不代表第三方关闭无警告。

源码manifest `7ad3f4c43285157949a0dce1f8119df9a5551c9dd1059854b718abd73b9cacf8`；Native库SHA `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`，自有CANN9.1构建。原始回执留蓝区和本地私有暂存，不上传完整数据。

[独立回执复核](../../experiments/worker_core_v11_20261118/continuous_public_native26_review_20261122.json)与[源码审查](../../experiments/worker_core_v11_20261118/continuous_native26_source_review_20261122.json)共同限定资格。A3按[独立构建步骤](native_regions_a3_build_20261122.md)另审产物，并分别验收TP1/TP2。当前不支持MLA、sequence-parallel、并发共享residual、跨stream或不同add-rms实现；26 residual_stomp仍排除。
