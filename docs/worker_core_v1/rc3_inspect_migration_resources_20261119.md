# 现役 inspect、软件迁移与蓝区资源（2026-11-19）

本页是4.7准备结果，不是4.8/RC3执行许可或通过结论。4.8已获项目负责人批准，仅按新增容量/驱动门禁执行0.26 A2迁移；黄区现役inspect到达后再形成4.9适用性结论。不等待这些材料来阻塞RC2子集交付。

## 给 openchamber 的只读 inspect 清单

仅在已批准的专用实例中读取真实部署；不把历史启动脚本当实值。按字段回传文字，未知写null/原因，不传权重、密钥或完整张量。

1. 模型完整标识、config/权重manifest SHA、量化方式/description SHA；实际模型类与配置一致性。
2. A2/A3及具体芯片，驱动/固件；CANN版本、SDK/opapi/NNOPBASE文件身份。
3. 镜像完整digest、Python、torch/torch_npu/vLLM/vLLM-Ascend版本及构建提交；安装路径与关键源码SHA。
4. TP/PP/DP/EP、实际Worker/rank/模型实例数、每卡可用资源。
5. enforce_eager、async scheduling、chunked prefill、prefix cache、graph/compilation、speculative/DBO/context parallel。
6. max-model-len、max-num-seqs、max-num-batched-tokens、block-size、KV dtype/布局与cache分组。
7. 实际Attention backend、prefill/decode API及输入布局；FIA/TND/MLA路径，不能只报后端配置名。
8. 实际量化method类/源码、DynamicQuant/QMM或融合API、W4A8/W8A8的Q/scale/zero-point约定。
9. KV writer→cache→consumer，KV Transfer/AscendStoreConnector/Mooncake、Engram/LMCache实际启用情况。
10. 是否能建立串行eager隔离实例，关闭异步/图/speculative及非必需组件；若关闭Engram/LMCache后不能正常加载，保留实际错误，不强行替换链路。
11. 正常推理请求的prefill/decode、首token与资源结果；只读inspect本身不触发故障。
12. 三项主项目合入确认、接线源提交和部署负责人确认。13项旧部署字段加hardware_family与以上细项分别回传。

## 最新模型基准（用户材料，仍需actual inspect）

- 现役DeepSeek：V4.1-Flash-1017-W8A8，v0.32.0rc1-a3，A3，TP8/DP2/EP，8实例16NPU，maxseq16、maxlen1048576、batched16384、block128、async/prefix、dspark5、FULL_DECODE_ONLY、Engram cpu_offload+dp_shared_memory、CPU binding/shared expert DP；无LMCache或KV Transfer。
- **后续DeepSeek测试优先 V4.1-Flash-1017-W4A8-Ascend**。历史镜像v0.30.0-a3-lmcache、TP8/DP2/A3只是启动参考；先确认权重完整性、镜像/CANN、EP与LMCache必要性、可控eager。不得以W8A8代替或共享未经验证的Q/scale假设。
- Qwen现役：Qwen3.8-27B-W8A8，参考v0.26.0rc1、TP1/8实例，maxseq4、len262144、batched4096、prefix、qwen3_5_mtp3、FULL_DECODE_ONLY、CPU binding、AscendStoreConnector/Mooncake。实际栈和接口需确认。
- 顺序：Qwen家族→DeepSeekV4.1 W4A8→DeepSeekV4.1 W8A8。MoE、视觉、量化、Attention/KV、collective各选满足机制的真实模型，不强制一个模型涵盖全部能力。

## 逐能力迁移清单

共性：当前 `deployment_acceptance.py` 精确组合绑定0.18/CANN8.5.1/A2；新增配置只能增加**已审查验收记录**。`serving_context.py`、`call_binding.py`、`prerequisites.py` 的源码签名、实际内部request ID、runner/execute/logits边界和TP1串行eager检查是代码契约。新栈若改变调用语义，必须新增适配并验收，不能只更新hash放行。

|能力|源码/版本与Native边界|迁移处理|
|---|---|---|
|8|route8/MoE restore与真实unpermute、expert method签名|新MoE/routing源码适配；同语义经源码审核可增配置，EP需改代码|
|15|私有格式/别名+真实写读链|当前不适用；新模型先发现自然链，不用改支持表造目标|
|17|vision17 task、Torch Conv3d与processor/cache|新视觉路径需代码适配；非视觉不可用，无需套Native ABI|
|18|rms18任务、RMS return与消费者、root/logits源码|同语义可审新binding，融合/RMSNorm/调度变化改代码|
|20|WeightNz/QMM实际slot、serving20/serving20_qmm、Native opapi|重新编译Native并核formals与consumer；W4A8融合差异须代码|
|22|rope22、fused RoPE真实reader与index语义|新rotary分支/缓存布局改代码，不能复用旧源码SHA|
|25|真实ReduceScatter，当前不支持|维持不支持；多卡不能通过配置补出机制|
|26 residual|writer26/residual26、allocator块邻接、typed QMM ABI|新allocator/ABI代码与构建，逐调用证明邻接，TP/EP不能继承TP1|
|26 FIA|typed FIA writer→当前请求residual→consumer|新FIA/MLA布局和存储生命周期需代码，保留真实目的存储约束|
|Batch|serving_quant_projection与quant_group Native、DynamicQuant/QMM|新CANN独立构建ABI收据；W4A8新量化语义需代码，不能单改版本|
|28 FA/FIA legal|历史限定源码/tiling/key|本次不扩展，不迁移为默认可用|
|28 denominator|真实TND prepared input、executor、POD/ELF|当前未闭环；新SDK/NNOPBASE行为/launch须代码与重新构建|
|33|Source105 arch22 S2 host/key/模板和logical KV reader|保留Source105机制，新CANN/template/MLA/page布局需Native及Python适配|
|39 matmul_range|独立Native MatMul算术/输入范围|不是RC2普通任务；按实际API适配，不做8.x/9.x一刀切|
|39 output_proxy|真实单请求输出消费者、postwrite符号/ABI/read_span|新CANN重编与实际DSO身份；现ABI不暴露编译器SDK，必须另附构建回执|
|41|postwrite、真实KV writer/cache slot/consumer|重编共享backend；MLA/Engram/KV-transfer布局与生命周期需代码|

多卡共性缺口：Host已支持显式选Worker而非默认rank0，但未证明TP/DP/EP协调、全rank恢复和跨rank效果；source组合登记不能替代这些代码/验收。异步和图模式同理。串行eager条件不满足时明确拒绝。

## 蓝区资源实查

124为aarch64，可见物理卡2/5，两张910B4、每卡32GiB。2026-11-19 12:44 UTC检查无外部NPU计算进程；本轮卡5随后用于Batch。共享主机不保证未来连续窗口，逐case用资源锁和门禁；kubelet残留分配须API/PodResources交叉核对，不改分配文件。

磁盘实查：/home约62GiB可用（94%已用），/约14GiB可用。不能假设足够保存两套完整新镜像、Native构建与W4A8权重；迁移前需按镜像层大小和模型制品实际大小预算，不清理其他任务数据。

只读registry检查（未pull）：

|镜像tag|manifest可达与平台|index digest|
|---|---|---|
|v0.26.0rc1|HTTP200，arm64/amd64|243bf20fad2f4f6de56aa1b041bcdfdbe5bb9d485f3190f40f1c5d0f418fc469|
|v0.32.0rc1|HTTP200，arm64/amd64|ff6d1ea57df08dd9382c51d202d0dd2d9acc35d53a65bcb0dfd906886e674912|

[0.26官方安装说明](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc/installation.html)把无a3后缀tag列为A2；实际层下载、CANN与可运行性尚未验证。0.32的manifest可达不等于A2兼容已验，仍待官方对应资料及隔离inspect。没有把现有CANN9.x ATK镜像当作目标vLLM镜像。

124到HuggingFace本次连接超时/网络不可达；不能承诺从124直接下载。官方公开的 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 和 [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)可作家族组件迁移参考，本机也有既有Qwen小模型及V2-Lite-W8A8。V2-Lite不是V4.1替代验收。[V4.1官方模型](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)的公开存在不证明用户指定1017-W4A8-Ascend制品已经取得；该精确W4A8权重、容量和A2可加载性待确认。当前两卡不具备用户历史TP8/DP2实例资源。

建议分工：蓝区先在批准后做0.26 A2无设备检查/新CANN构建与Qwen小模型机制迁移；DeepSeek专项先解决精确W4A8制品及资源，再运行真实该路线。黄区87 A3由openchamber按独立专用实例验收，每能力两作用域，通过许可回传后按来源登记。A2通过只登记A2，A3最终结论不合并。

## 负责人更新后的执行边界

阶段D仅0.26 A2；暂不拉取/实测0.32，不进行任何DeepSeek实测。先核对Docker实际数据目录、压缩/解压大小及驱动要求。根分区存储、拉取后不足15GiB或驱动不兼容任一条件出现即停止且不拉取；关键尺寸/兼容性无法证明时保留未通过，不猜测。自有重构建产物置/home且总量不超过10GiB；不改宿主驱动/CANN/Docker，不动0.18环境、不删外部数据。第一步通过后，只用本地Qwen3-0.6B BF16/W8A8验证18/39/22/Batch各两作用域。
