# Batch 共享量化尺度：真实 DeepSeek Worker 与逐请求证据

124 物理卡5、910B4/A2、CANN8.5.1、vLLM-Ascend0.18.0，真实 `DeepseekV2ForCausalLM` / `deepseek_v2`，预训练 DeepSeek-V2-Lite-W8A8（MLA/MoE），TP1、BF16、eager。不是 Qwen 蒸馏模型，也不是构造模块。新采集原件、辅助 Q/scale 张量和 vendor 二进制留在124，公开文件只有实现、总结和哈希。

**公共 Worker 单点自动选点、实际量化命中、逐请求对应和 OFF 恢复通过。独立统计取得故障量化尺度的首个可观测变化，但完整基线自比及其他阶段仍有 NaN，不能宣称全段严格统计通过。**

## 实际入口与机制

安装后的三个 wheel 经标准 vLLM 临时 Worker 加载。共用 `prepare → register → check → execute → advance → finish → unregister` 管理生命周期；Extension 不创建 Worker，不替换 `execute_model` / `_prepare_inputs`。

用户不填写层名。实际调用按 `input_ids → VocabParallelEmbedding → RMSNorm → W8A8 projection` 的存储关系选择，得到 `model.layers.0.self_attn.kv_a_proj_with_mqa`。135个结构候选中只验收这一条真实输入链，不能把专家路由后的 row 当作请求，也没有开放所有候选注入。

每阶段8个真实 enqueue 请求，各生成2个 token。prefill 正常执行，首个 decode 的8个 row 通过实际 request ID、`req_id_to_index`、长度、位置、已生成前缀和设备输入字节核对；不是从 ID 后缀猜映射。四阶段为 reference、lossless、comparison、OFF，每阶段建立新请求。

故障仍是主动改变后端归约域：真实 `aclnnDynamicQuantV2` 的 X/Q 描述符由 `[8,2048]` 改为 `[1,16384]`，scale 为一个原生量化结果，再在同一 stream 上发布到8个 scale row；原生量化只执行一次。不修改输入、权重或后续模型输出。该机制是错误共享 batch 尺度的受控模拟，不等于异常文本自然诱发了后端错误。

本部署只导出匹配头文件的 V2，不把版本号当成 V1/V3/V4 的 ABI 证明。当前 torch_npu2.9 的全局 `GetOpApiFuncAddr` 不导出，因此加载器明确记录 `ACTUAL_TYPED_ENTRY_COUNTS_REQUIRED`；加载成功、符号存在和 armed 都不算命中。真实 GWS/execute 各1次、API版本2、错误0后才确认 native execution。原 TP2 加载器保持原合同。

## 真实逐请求结果与恢复

下表logical request和tensor row编号从0开始。

| 项目 | 观察结果 |
|---|---|
| B/F 量化输入 | 完整输入哈希、请求顺序、token位置和已知前缀一致 |
| logical request 2 / tensor row 2 | scale **0.13877953588962555 → 0.14271654188632965** |
| 该 row 的 Q | 2048个 INT8 元素中71个改变，最大整数差4 |
| 其余7个 row | scale原本即为0.14271654188632965；Q和scale均未变 |
| F scale | 8行相同，精确等于B各行scale的最大值 |
| F实际命中 | comparison阶段选定真实调用1次；lossless仅无损路由；reference/OFF不注入 |
| 模型传播 | 完整 hidden/logits 哈希变化；8个请求的2个生成token和文本均未变 |
| 恢复 | lossless/OFF的完整hidden、logits、Q、scale精确等于reference；unregister后正常生成恢复 |

逐请求证据来自实际量化返回的 Q/scale 和独立运行时映射，是辅助核验，**没有输入首差比较器**。详见 [完整生成范围摘要](summaries/quant-generation-per-request.json) 和 [首decode范围摘要](summaries/quant-decode-per-request.json)。两次独立 B/F 重启取得相同逐行现象。

## 只用原始 statistics/mix 的独立比较

原比较器 SHA-256 为 `28d9d87fa3bab5ed18c692b29e3ac880e86e46d31916397e0da040c4eb2e9bbe`，没有修改；固定 `abs(F-B) > 1e-8 + 1e-5*abs(B)`。只读取 dump/stack/construct、配置和采集器身份，不读取命中日志、Q/scale 辅助张量或模型传播报告。全部 API/module 采集，无算子过滤。null 保留，NaN→NaN 按不可比较处理。

先保留整个 prefill＋decode 的4阶段采集。其 step0 首个阻塞是 `Tensor.__setitem__.6.forward/input_args/0` Max NaN→NaN；step1–3 是 `Tensor.__setitem__.2.forward/input_args/0` Max NaN→NaN。B自比也相同，严格结果未通过。见 `summaries/quant-generation-statistics-step0..3.json`。

随后在新 attempt 中单独采完整首个 decode：Host 用公开 engine.step 完成正常 prefill，确认实际请求已产生首token，再在下一次模型调用之前开启采集。没有跳过该 decode 内的早期算子、删除 NaN 或改变原采集文件。这里的 `step0/1/2/3` 是 reference/lossless/comparison/OFF 四个独立请求阶段，各采一次首decode，不是连续4个生成token。

| 首decode采集阶段 | B/F严格输出 | B自比与限制 |
|---|---|---|
| step0/reference | `NPU.npu_kv_rmsnorm_rope_cache.0.forward/output/2` Max NaN→NaN，index112 | 同处不可比较；之前242项null |
| step1/lossless | `Tensor.__setitem__.7.forward/input_args/0` Max NaN→NaN，index456 | 同处不可比较；之前958项null |
| step2/comparison | **`NPU.npu_dynamic_quant.0.forward/output/1` Min 0.13877953588962555→0.14271654188632965**，index76 | 首差前163项null；B自比到index456仍遇到上述NaN |
| step3/OFF | `NPU.npu_kv_rmsnorm_rope_cache.0.forward/output/2` Max NaN→NaN，index100 | 同处不可比较；之前218项null |

step2 的首个可观测变化是量化 scale 输出，符合归约范围改变导致共享尺度的预期；不是通过运行时日志指定位置。它只证明可观测前缀中的首差，不能排除之前null字段的变化，也不能唯一证明归约域错误。全段健康自检未通过，不能用辅助恢复哈希覆盖这一事实。统计报告中的 `UNPROVEN_REQUEST_MAPPING` 保留；真实请求映射属于独立运行时证据。

逐阶段完整摘要见 `summaries/quant-decode-statistics-step0..3.json`，含原始三件套、配置、采集器及私有完整报告哈希。

## 实现、部署与边界

- [真实请求row绑定](../../src/demoncat/runtime/vllm_ascend/worker_extension/batch_context.py)、[Serving入口](../../src/demoncat/runtime/vllm_ascend/worker_extension/serving_quant.py)、[共用Provider](../../src/demoncat/runtime/vllm_ascend/worker_extension/providers/quant.py)。
- [V2原生边界](../../fault_injection/native_adapters/batch_scale_v2_only.cpp)、[原生控制器](../../fault_injection/native_adapters/batch_scale_control_v2.py)、[校验加载器](../../fault_injection/quantization.py)、[无设备构建](build_quant_v2.py)。
- [模型验收脚本](probe_quant_worker.py)、[原始统计比较脚本](compare_pair_124.py)、[单独逐请求核验](summarize_quant_pair.py)。

真实构建DSO SHA-256：`f05f3b689bac4d3a9d9fa3b9611893d9c44561c0639f31e7ee34afe0f1e3d5ed`。源码、V2头文件、real opapi、控制器和构建记录指纹见 [构建摘要](summaries/quant-build.json)；`COMPILED_NOT_NPU_VALIDATED` 是构建时状态，实际NPU资格在单独运行摘要中。

整段生成使用 `packages-quant-a5`，随后首decode采集使用 `packages-p4-final-a1`；两者共用同一native构建。旧probe按旧case冻结保存，当前公开probe与最终首decode实测字节一致；不能用当前probe的哈希覆盖旧attempt。具体镜像、安装wheel、冻结脚本和资源门禁见 [执行校验摘要](summaries/quant-execution-manifest.json)。

vLLM 的原有 custom opapi 通过进程级 LD_PRELOAD 保留，避免同名私有库遮蔽 AddRmsNormBias；没有修改共享安装。早期库遮蔽、factory、wheel缺少源码、原生状态字段不齐的失败均保留于 [失败记录](summaries/p4-prior-attempts.json)，未计通过。资源锁和每case空闲门禁均启用，未停止外部作业。

本次支持范围为这个具体 CANN8.5.1/V2/TP1/B8 模型部署和一个自动发现的自然投影，**不表示全部8.x/9.x、其他量化API、TP2、所有135个候选或任意模型已通过**。公共 `all_eligible` 仍关闭。CANN9.1历史TP2证据单独保留，不能替代本次部署验收。

124原件：`/home/p3p4-closure-20261117/quant-deepseek-a6` 与 `quant-deepseek-decode-a1`；独立报告为 `quant-deepseek-independent-step{0..3}-a1` 与 `quant-decode-independent-step{0..3}-a1`。新原始 statistics/mix 不提交 GitHub。
