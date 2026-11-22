# 33 方向 A：原生执行取证与边界

负责人选择 A，限时尝试；身份字段是登记前提，耗时暂不优化。本轮采用两小时工作窗口、最多30分钟 NPU 上限，不自动延长。A 尚未闭合，不自动改选 B/C/D，不扩大非原生等值单元的替换范围。

## 组合与结果

蓝区 A2 / Ascend910B4 / CANN 9.1.0 / vLLM 0.26.0+empty / vllm-ascend 0.26.0rc1 / 社区 Qwen3.8-27B-W8A8 / TP2。eager、单请求、单 chunk、2304 token prefill、生成4 token；max_model_len=max_num_batched_tokens=4096。实验没有开启故障，只记录真实调用；不属于四阶段、两作用域注入验收。控制器沿用的 recovery_plan/expected_hits 元数据不能当作本次恢复或命中证明。

每个 rank 实际观察到64次 FIA：16次 prefill、48次 decode。实际 API 为 `aclnnFusedInferAttentionScoreV3`；Python调用为 `torch_npu.npu_fused_infer_attention_score`。decode 的 query=[1,12,256]、KV缓存=[2244,128,512]、BF16、TND、block_size=128、GQA头数12/2、scale=0.0625、sparse_mode=3，最后一步 actual_seq_lengths_kv=2307。**缓存页面128不等于核内S2归约块128**，不能据此认定旧torch控制分组是原生顺序。

第二轮从进程启动前开启 msprof/runtime debug，实际 launch 日志确认：

|阶段|实际 tiling key|numBlocks|scheMode|
|---|---|---:|---:|
|prefill|5000000000000200203|20|1|
|paged decode|5100000000010200203|10|1|

kernel族为 `FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da`。现存日志有82条decode launch记录；日志发生轮转，不能将82当作完整调用覆盖分母。原始API调用元数据64次/rank另行完整保存。第一轮torch_npu profiler的设备表缺少混合核；第二轮op_summary出现prefill混合核，但decode设备表仍不完整，不能把缺行判成没有执行。

## 来源指纹

源码只在远端分析，保留指纹和定位，不导出完整CANN源码。

|文件/记录|SHA256|
|---|---|
|attention/attention_v1.py|4ba903ee8afb23012e78870034b1280039ccf81edb41b8e16f00e6e5bbc4c076|
|fused_infer_attention_score_tilingkey.h|ca5dd685ba91365d0ec9cdc22a4026a71789d54cb3dcc473f4e5c9e987b3e974|
|flash_attention_regular_decode.h|6e63888f47d8ee0b57d77c268a87dbfea6c3257e26bff7cc4ef08f4f02073f92|
|fused_block_epilogue_online_softmax.hpp|2f2932fe722e4bae523ba9c8df14e7ecad9415fa4f72d2d2cc461dfa42f8a664|
|fused_block_epilogue_rescale_o.hpp|71abeda56eb2cf4de6c026bfbfdc097b542f59efa107a9d134313d9d67ec5614|
|kernel relocatable JSON|f01fd8f4b205e7f02863ff11fa258a59a11094d80a670c2b9dd2337216fc9aa4|

tilingkey头文件第287/359行分别含上述两个key。kernel JSON声明的对象SHA为64df8f59554e4bee46a7a286554db4ef7c7e6a793156de02d1d4c628ef8c8bc2；这是声明值，不能当作独立读取对象文件所得的校验。key匹配只能证明候选模板定位，不能替代本次launch动态POD和数值指令次序。

## 尚缺的证据

1. 每次decode实际launch的动态tiling数据：S2切分、循环/尾块、核间部分和合并参数。现存日志未提供这些字段。
2. 与二进制一致的QK/PV累加、online-softmax和输出重标定顺序，以及每处转换/舍入行为。源码包含Cast及CAST_RINT等路径，不能用普通torch FP32公式直接宣称逐位等价。
3. 固定真实Q/K/V下的原生等值控制重算；再对同一合法key集合仅改变归约分组；最后完整模型两作用域传播和恢复。

当前结论为 `NATIVE_LAUNCH_IDENTIFIED_REDUCTION_ORDER_UNPROVEN`。没有新的故障资格，没有将旧split8/16失败改写为通过。旧结果见[paged33_redesign_20261122.md](paged33_redesign_20261122.md)：两者rank0均无模型传播，rank1分别4/3个单元变化，每rank48次eligible消费及恢复；不能归因为原生分组影响已闭合。

## 窗口与隔离

|attempt|UTC开始|UTC结束|窗口秒|源码清单SHA|
|---|---|---|---:|---|
|paged33-native-a1|2026-11-22 11:59:47.381528|2026-11-22 12:03:40.211235|232.830|3f77b3ccea974a3cc813e6d8cf7f5975f4dc0843592d50b80ba1a3070c15e6c6|
|paged33-native-a2|2026-11-22 12:12:57.216565|2026-11-22 12:17:53.824548|296.608|f3ab408cdca067c9088e4d0e0bfd4ebf9156471ac06803844f7031aacbc7aa70|

总计529.438秒（包括启动/预热/采集，不是纯kernel时间）。两轮退出0，卡2/5有界隔离均已解除。设备冲突不算机制失败；容器退出也不算同进程OFF恢复证据。完整profile/日志仅在实验存档保留，不提交张量或完整第三方源码。
