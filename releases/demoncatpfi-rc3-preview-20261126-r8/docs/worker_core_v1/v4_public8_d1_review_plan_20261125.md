# 公共8：V4 0.30 D1行归属审查方案

状态：审查方案就绪，未收到本轮D1原始证据，公共8保持 COMBINE_OUTPUT_ROW_JOIN_UNPROVEN。目标组合是黄区实际A3、V4 W8A8、0.30、CANN9.2.1、TP8及运行时EP组；不能借用A2/0.26/V2-Lite的资格。本文不修改r4.1冻结工具。

## 源码依据与身份

[源码候选清单](v4_moe_v030_review_sources_20261125.json)记录官方vllm-ascend v0.30.0 tag的六份文件URL、字节数和SHA。它们是审查候选，不是黄区安装源码指纹。D0回来后先对照实际安装版本、文件SHA、调用位置和算子schema；不同则读实际差异，不能按tag补全未知字段。还需实际vLLM调度/InputBatch源码、torch_npu/op-plugin/CANN对应算子契约，尤其MC2辅助张量语义。

源码定位：

- [token_dispatcher.py](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/ops/fused_moe/token_dispatcher.py)：dispatch/combine共享元数据，实际MC2通信组，量化分支。
- [prepare_finalize.py](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/ops/fused_moe/prepare_finalize.py)：padding、TP切分、all-gather与去padding。
- [moe_comm_method.py](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/ops/fused_moe/moe_comm_method.py)：dispatch → 专家MLP → combine → finalize的调用关系。
- [sequence_parallel.py](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/common/ops/sequence_parallel.py)：SP补齐与分片。is_padding=True与MC2 x_active_mask=True含义相反，不能直接复用mask。

## D1到达后的顺序

1. 核对run、包/源码/模型身份、prompt-manifest、每rank进程代次、D0绑定目标与组。错误、截断、整数预算耗尽逐项标缺证。
2. 按真实调用分为MC2 dispatch/combine、AllGather或其它实现。以下MC2审查不能移用于AllGather。
3. 关联键是运行代次、全局rank、请求集合、forward步、层、该层本步调用序号、prefill/decode。journal的sequence只是本地序号，不能要求不同rank相等；同层重入不能按最近一条猜测。保存实际dispatch与combine引用链。
4. 建立调度token域：InputBatch.req_ids、query_start_loc、num_computed_tokens及实际调度数得到每个输入token的请求/位置。再按实际SP与prepare分支传播行号，得到dispatch输入的本地有效行与padding。未经这一步，runner行号不能直接用于combine输出。
5. 用下表核对路由元数据、生产者/消费者和算子契约，证明raw combine输出行对应dispatch输入的哪个token域。整数匹配只能证明候选关联，不能替代行语义。
6. 跟踪finalize是否all-gather、去padding或保留本地分片，以及真实下游算子读取的对象区间。最终输出行→调度token→请求映射必须在所选注入边界成立。
7. 输出逐rank审查结论、缺失字段、原件SHA；通过映射审查后才进入D8适配和两个owner产品验收。观察通过不等于注入通过。

## 路由证据表

|阶段|需要的实际值|审查内容|
|---|---|---|
|prepare/SP|replace_allreduce、TP成员顺序/本地rank、原始行数、padded_num_tokens、切片区间、mask|原始token如何成为本rank输入；不能由总TP数猜偏移|
|dispatch输入|expert_ids、x_active_mask（若提供）、x元数据、group_ep、ep_world_size/ep_rank_id、global_bs、quant_mode、可选group_tp参数|有效行、专家范围及副本/分片域；运行时MC2组映射到全局rank|
|dispatch输出|expand_x；dynamic_scale；assist_info_for_combine；expert_token_nums；ep_recv_counts；tp_recv_counts；expand_scales|按安装schema命名，不仅凭tuple位置；验证整数边界、计数与该版本契约|
|MLP|expand_x到MLP输入，再到combine.expand_x的生产链|combine输入是专家计算结果，不能误认为原token输入或dispatch原输出|
|combine输入|expert_ids、expert_scales、assist_info_for_combine、ep_send_counts、可选tp_send_counts、mask、通信参数|与同一步同一层dispatch保存元数据对应；ep_send_counts在候选源码来自dispatch.ep_recv_counts|
|combine输出/finalize/consumer|行数、有效行、padding、输出对象区间、重组规则、真实下游输入区间及设备读取完成证据|每行归属唯一且可追溯；证明实际消费者，不以Python返回作为设备读取完成|

候选源码dispatch前七项输出依次为expand_x、dynamic_scale、assist_info_for_combine、expert_token_nums、ep_recv_counts、tp_recv_counts、expand_scales；安装schema不同须重审。V2的assist_info是辅助编码，不能当成旧版expand_idx或按旧公式解码。若契约不能证明raw输出顺序，就停在缺证；不通过“shape相等”认定归属。

量化分支可把可选tp_recv_counts换成torch.empty(1, int32)再传combine。必须核对该分支参数是否被算子使用；不能要求它与dispatch逐位相等，也不能把未初始化值解释为有效路由计数。此说明不是允许忽略真正使用的计数。

get_mc2_group的成员必须单独确认；generic EP组不自动等于MC2组。算子可选tp_world_size=1不表示部署TP=1。跨rank地址没有可比性；同进程指针复用也不是同一张量的证据，需时间、对象生命周期和内容SHA共同约束。

## 行映射判定

对每个输出有效行r建立M(rank, step, layer, invocation, r)=(request_id, token_position)。padding映射为无请求，禁止注入。验证有效行覆盖、重复数及缺口时，以实际通信域为单位：TP分片可能互斥，副本可能重复，EP没有本次调用的rank不能制造调用凑齐。

现有sites/dsv4_request_rows.py的combine_row_owners要求完整调度行的双射；它不能直接接受只含部分调度行的raw TP分片。选择已证明的完整重组边界，或另做明确的分片适配，不能复制/补造行使检查通过。只有同一请求至少两条有效行时，才可做请求内行错位；单行decode不得跨请求借行制造效果。

通过条件：实际分支与源码/算子契约相符；映射无歧义；非目标/无效行不被修改；真实消费者可验证；同stream读取与恢复边界安全。随后产品验收还需全rank持续覆盖、两个owner、三项恢复、噪声金丝雀和负例，不能拿D1计数替代。

## r2现有证据的上限

r2有schema、调用栈/层、调度行、整数值/SHA、张量元数据、标量参数及候选关联。padding和valid_output_rows当前明确为null；不记录prepare上下文、完整MC2成员映射、浮点router/scale值或设备消费者完成证据。缺失可由同身份的可靠源码契约和运行时证据补齐；无法闭合时按[定点补采单](v4_d1_targeted_capture_20261125.md)处理，不能把null补成默认值。

审查输出标签仅为文档结论（不是新增RPC拒绝码）：SOURCE_MISMATCH、TARGETED_RECAPTURE_REQUIRED、ROW_MAPPING_PROVEN_PENDING_ACCEPTANCE。没有正式产品验收时门禁始终关闭。

