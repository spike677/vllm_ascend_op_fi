# 39 output_proxy：公共入口持续验收

组合：124 A2 Ascend910B4，CANN 9.1.0，vLLM 0.26.0 / vllm-ascend 0.26.0rc1，torch 2.10.0+cpu / torch_npu 2.10.0.post4，社区 Qwen3.8-27B-W8A8，TP2，eager 纯文本。不外推 A3 或其他模型。

`continuous-public-a12` 经独立 `tools/review_continuous_public.py --require-negative-creation` 通过。原回执留在124；仓库保留 `experiments/worker_core_v11_20261118/continuous_public_39_review_20261121.json`（含原件与比较器SHA）。源码清单SHA为 `ac15e212cb619355248a8cd1e61b0f512a080ddcff2feb203830b03c62dc31e7`。

同组 Worker 两作用域均经公共 Runtime、相同 request_plan：normal/fault/off/health，各实际 prefill + 3 decode，生成4 token。无许可的实际创建拒绝，结束后许可/租约/钩子/在途资源清零。OFF/health 的完整 hidden/logits 哈希与 normal 精确一致。

每个作用域、每个rank：304个实际投影，1216次 observed/eligible/attempted/applied/changed/consumed/restored，failed=0。352个结构候选中48个 `linear_attn.conv1d` 仅使用权重，没有发生模块forward，逐项保留为“结构存在但无实际调用”；视觉分支被纯文本观察器禁止。没有为这些对象制造调用。

故障写入每次 BF16 ND 投影返回值最后行的一个元素。实际读消费者：`aten::contiguous`（确有独立存储复制）、`npu::npu_swiglu`、`vllm::maybe_chunk_residual`、`vllm::qwen_gdn_attention_core`、`vllm::triton_split_qkv_rmsnorm_mrope`。视图本身、空张量分配及不复制的 contiguous 均不算消费；会原地修改受保护区的消费者拒绝。设备同stream完成事件之后恢复对象区间，恢复字节哈希逐次核验。

每rank每作用域 upstream_nonfinite=1208，already_positive_inf=0。NaN改为Inf是新增字节变化，不能归类为“上游已是Inf”。故障输出非有限是该机制的预期；normal/OFF/health仍必须全部有限。hidden/logits发生变化，不以生成文本是否改变作为通过条件。

两个作用域 fault/normal 耗时分别2.4234475×、2.4335716×；请求保持现有120秒预算，不宣称适用于更长请求。

## 调试窗口与隔离

|窗口|UTC开始|UTC结束|窗口秒数|结果|
|---|---|---|---:|---|
|a7|12:02:11.602679|12:06:01.212930|229.610|结构计数错误包含视觉/未调用对象|
|a8|12:07:50.126202|12:11:40.840674|230.714|352结构候选与304实际调用不等|
|a9|12:13:06.292969|12:17:08.902680|242.610|真实消费者未证明，拒绝验收|
|a10|12:18:21.085400|12:22:04.900777|223.815|将消费检查前移至不注入预热，定位48个in_proj_ba|
|a11|12:23:54.230631|12:27:38.194114|223.963|确认别名schema的contiguous实际复制，旧观察器漏记|
|a12|12:28:40.800372|12:36:28.519318|467.719|公共两作用域及独立审查通过|

均为2026-11-21；这是含加载/预热的窗口墙钟时间，不是kernel活跃时间。只使用卡2/5的有界精确设备容器清理器，每窗最大2100秒；所有窗口已停止清理器并恢复派发能力，未修改调度器/Docker配置，日志没有外部容器stop事件。没有修改GDN或其他第三方kernel。

此次只追加开发线 BLUE_COMBINATION 及候选，不修改冻结发行物，不发布r3。下一项为18/22公共持续接线。
