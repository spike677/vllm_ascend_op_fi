# RC3 第一批范围建议：Qwen 混合注意力专用实例

状态：**负责人已确认四项范围，按18/39→22→Batch推进；尚未开始RC3机制适配/发布**。依据[黄区接收记录](yellow_intake_20261119.md)，不是蓝区NPU证据。目标为A3、Qwen3.8-27B-W8A8、0.26系列、TP1专用实例；硬件代际尚待权威确认。当前只审查文本、串行eager、关闭MTP/KV connector/prefix/async的测试条件，不涵盖现役FULL_DECODE_ONLY和异步调度。

## 四项结论

|能力|结论|阶段与实际候选|还缺什么|
|---|---|---|---|
|18 api_half|需适配|先限fresh prefill；实际AscendGemmaRMSNorm返回与后续消费者|Gemma的weight偏置/融合residual语义、真实schema及完整返回值；模型属于多模态架构，旧门禁拒绝，需要证明文本请求分支，不能只删is_multimodal_model检查|
|39 output_proxy|需适配|先限fresh prefill；实际量化线性返回/消费者的有限值→Inf受控代理|0.26模型/logits回调、真实返回对象映射、9.1重构建及无损/OFF；不继承旧postwrite二进制或历史越界结论|
|22 wrong_row|需适配|full-attention层的prefill RoPE；暂不覆盖linear-attention|出现split_qkv_rmsnorm_mrope融合kernel；旧独立RoPE hook可能没有调用。需融合边界、Q/K切分与mRoPE/partial_rotary语义；无自然可接边界则本组合不适用，不能伪造独立调用|
|Batch shared_scale|需适配|动态量化线性的首decode（沿既有六批B8协议），当前仅有B1摘要|实际DynamicQuant→QuantBatchMatmulV3关联、scale形状、量化描述/权重SHA；B8对应部署和请求映射、两个owner。W8A8_DYNAMIC标签和计数不能授共享尺度修改资格|

报告的64层为每4层前三层linear、末层full：**48 linear + 16 full**，0-based full层3、7、…、63。这来自结构摘要，还需完整模块/实际调用映射。full路径两阶段都有FIA；linear prefill为chunk/GatedDeltaRule，decode为recurrent。18/39/Batch不应只按FIA选层；22也不能把linear层都当RoPE目标。模型为dense，无MoE；不纳入8。

MTP、KV connector关闭后短请求报告成功，只支持这个短请求的可运行性；chunked-prefill关闭告警保留。未来若必须开启chunked，需重审fresh-prefill映射，不以默认关闭强行放行。

## 0.26 字段差异的源码线索

官方 vLLM-Ascend tag `v0.26.0rc1` 指向commit `f2f74a16c3c50a76f4349d807918e83edec1e35c`。
[runner源码](https://github.com/vllm-project/vllm-ascend/blob/f2f74a16c3c50a76f4349d807918e83edec1e35c/vllm_ascend/worker/model_runner_v1.py)中DCP从实际group读取，暴露`dcp_size`和`use_dcp`；没有旧`pcp_size/use_cp`赋值。[vLLM 0.26 runner](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/worker/gpu_model_runner.py)读取`parallel_config.decode_context_parallel_size`为`dcp_world_size`。

这是**候选语义来源**，不是pcp的同义字段证明，也不是黄区镜像已安装代码的身份。迁移必须核对安装指纹、CP配置和实际group，说明新版本PCP分支是否存在；禁止`getattr(...,1)`或让openchamber附加伪字段。`is_multimodal_model`、mRoPE位置布局、混合KV/Mamba状态、fresh request映射及采样生命周期也需逐项审查，不仅处理一个属性。

## 三层迁移清单与范围确认

1. 部署身份：A3硬件型号、精确镜像/包/源码/模型/量化身份；字段新增属于配置变更，不能授实际调用资格。
2. 机制兼容：CP等新字段来源、混合模型文本分支、norm/rope/quant接口、Native9.1构建；上述语义变化必须改代码并重测。
3. 实际调用：真实enqueue内部ID→能力声明阶段（18/39/22先prefill，Batch首decode）→自动目标→消费者→效果→OFF/health，两个独立owner且同Worker、资源归零。蓝区A2只授对应可移植性组合。

建议先18/39，再22融合mRoPE，最后Batch B8。每项必须独立有证据；没有对应W8A8小模型时Batch不以BF16结果代替。已获准按上述顺序进入RC3开发，仍遵守部署门禁；A3步骤按最终适配的binding/许可/request_plan编写。DeepSeek未启动，第二批暂不定范围；不拉取0.32或做DeepSeek实测。

## 同架构公开小模型与传输建议

已核查官方模型配置和API（2026-11-19），均为`Qwen3_5ForConditionalGeneration/qwen3_5`、BF16、24层（18 linear+6 full），full_attention_interval=4、head_dim256、partial_rotary0.25/mRoPE。可用于同架构小规模可移植性测试，不能代替27B部署的量化或A3验收。

|候选|固定revision|原始safetensors字节|许可|选择|
|---|---|---:|---|---|
|[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)|2fc06364715b967f1860aea9cf38778875588b17|1,746,942,600（约1.63GiB）|Apache-2.0|优先，最小资源|
|[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)|15852e8c16360a2fea060d615a32b45270f8a8fc|4,548,221,488（约4.24GiB）|Apache-2.0|备选，不同时下载|

[0.8B配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)、[2B配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json)与对应LICENSE为选择依据。权重体积不等于运行显存；在32GiB A2 TP1、短文本/≤2048、无视觉输入/无MTP、受限KV预算下具有容量余量，实际0.26启动/峰值尚未验证。

124无法访问HF。建议由负责人在可访问官方源的机器获取上述固定revision的完整配置、tokenizer/processor、generation_config、safetensors、LICENSE和逐文件SHA清单，经批准传输到`/home/demoncat-model-intake/Qwen3.5-0.8B/`；禁止只传LFS指针或从Git提交权重。到达后先离线校验大小/SHA/索引完整性，再经过0.26门禁启动。当前未下载权重，未创建服务；公开BF16制品不是本任务所需W8A8，Batch须另有匹配量化制品。
