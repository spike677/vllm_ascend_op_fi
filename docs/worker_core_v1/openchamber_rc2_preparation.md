# 给 openchamber：RC1.1 回归与 RC2 测试部署准备

无需等待RC2发行即可准备。蓝区无法访问黄区，请只回传总结；不要求原文、权重或私有数据。
RC1.1标签：`demoncatpfi-yellow-rc1-1-20261119`，发行提交
`70db3e09465f1ca98f4a4cd0436a4096efd4b592`。请按[yellow_rc1.md](yellow_rc1.md)
回归18/39两个独立作用域，并返回阶段、OFF/health恢复、最终资源与清理状态；第五请求应明确拒绝。

|能力|所需蓝区参考测试部署|黄区准备事项|
|---|---|---|
|18/39|Qwen3-0.6B BF16，TP1|现有部署可回归RC1.1；RC2新增身份门禁需实际环境摘要|
|22|Qwen3-0.6B BF16，TP1|已有纯文本测试部署；只选首条实际fused RoPE，需匹配来源与最终reader|
|8|DeepSeek-V2-Lite-W8A8，TP1/BF16计算|需MoE权重及匹配vLLM-Ascend动态专家实现；纯文本dense Qwen不能替代|
|17|Qwen3-VL-2B-Instruct BF16，TP1|需视觉权重、tokenizer/processor和单图像入口；16×16 patch Conv3d|
|Batch|Qwen3-0.6B动态W8A8，TP1/B8|需动态W8A8权重与按实际环境构建的量化bundle；BF16部署不可替代|

公共串行测试参数：enforce_eager=True、async_scheduling=False、
enable_chunked_prefill=False、enable_prefix_caching=False、TP/PP/DP=1；
max_num_seqs=1（Batch协议为8）。不启用speculative、DBO、上下文并行。
视觉每阶段必须真实执行encoder，不能复用缓存结果。17新普通接口统一为
advance(internal_id)→真实等待完成→query，不要求collect_replay；新接口已在124的a5两个独立作用域通过，黄区仍需独立验证。

请回传：
1. 各模型能否获取、确切标识/来源、模型配置SHA、权重清单SHA与量化方式（动态/静态）。
2. 镜像完整ID，vLLM/vllm-ascend/torch/torch_npu/CANN实际版本、硬件型号。
3. 实际TP/PP/DP/EP，执行与调度参数；是否允许上述独立测试配置。
4. 视觉processor配置和图像输入是否可用；Batch是否能构建并提供bundle清单/ABI摘要。
5. RC1.1回归：每请求内部ID、实际阶段、效果摘要、matched_off_and_health_confirmed、
   finish/disable/unregister/detach/remove结果；不要把用户请求名称当阶段依据。

RC2按能力声明部署组合。入口实现或结构兼容不代表验收，不匹配的组合默认不可执行。
蓝区显式试验许可不是黄区用户配置；不能将另一镜像/模型的历史身份填到新构建上。
黄区主项目接线与product_closed继续由openchamber实际验收确认。

Batch普通入口已补验统一宿主动作：advance_batch(actual_ids)→等待完成→query，宿主无需手工
step/capture/collect；六个实际校准/故障/OFF批次仍保留。本轮一个蓝区NPU任务作用域通过，
黄区仍需独立验收，不能将该结果套用其他镜像或模型。
