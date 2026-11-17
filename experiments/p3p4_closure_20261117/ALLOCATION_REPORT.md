# Fault26 写前 NaN：分配来源调查

旧 residual 变体 rank1 的严格统计结果仍不可比较；没有删除、初始化或将 `NaN→NaN` 判为相等。新调查只观察正常推理，不启用故障。

在实际源码中定位 `Attention.forward` 的 `torch.empty`，以及 Ascend 后端赋值语句，按同一存储身份、完整逻辑字节哈希、shape、stride、offset 串联分配后、赋值前和赋值后。观察器使用已有模块 hook 和受限 Python trace，不替换 Worker 核心方法，不写张量。NPU同步和CPU读取会影响时序及后续分配复用，结果仅适用于这次观察运行。

| 新运行 | 两次真实 prefill 的分配 / 赋值 | 同存储关联 | 分配后 / 写前 NaN | 写后结果 |
|---|---:|---:|---:|---|
| SmolLM2-135M | 60 / 120 | 120 | 0 / 0 | 全量赋值与 RHS 精确一致且有限 |
| GLM-4-9B-0414 | 80 / 160 | 160 | 0 / 0 | 全量赋值与 RHS 精确一致且有限 |

每层先在后端执行真实拷贝，再由上层执行一次对应赋值；两次赋值不是两次分配。首次赋值前的完整目标字节与 `empty` 返回后相同；第二次已含第一次写入结果。观察前、两次观察及移除观察后的 token、文本和 logprob 一致。

分配源码：`vllm/model_executor/layers/attention/attention.py:419`，SHA256 `990ead07c828e26192ca46b4b8abae7a6792f67c27510705ad7b483035f3631c`。
实际 prefill 写入源码：`vllm_ascend/attention/attention_v1.py:839`，SHA256 `bbe69c3015ee15789dea0d6ec9795243458b68494294c4c023ed5a5d665c550b`。
从实际实现方法枚举并解析赋值位置，不依赖用户填写层名。

这证明了新进程中未经初始化的目标分配随后被有限 FIA 结果完整覆盖；**没有复现写前 NaN，不能证明旧 rank1 的 NaN 来源，也不能据此解除旧统计阻塞**。`torch.empty` 本身不证明 allocator 之前的内容及其写入者。另一轮 GLM FIA 统计中实际出现的 NaN 同样保留，不能借用本轮“没有 NaN”来消除。

实现：[只读观察器](trace_attention_allocation.py)、[真实模型入口](probe_allocation.py)。摘要：[SmolLM](summaries/allocation-smollm.json)、[GLM](summaries/allocation-glm.json)。完整存储身份和逐赋值报告保留在124，公开摘要仅提供计数、源码身份及原件哈希。

第一版只找名为 `forward_fia` 的方法，实际未找到拷贝，观察不完整并失败；改为枚举实际方法后以新 attempt 重测。没有把第一次的60次分配误记为完整写入链通过。
