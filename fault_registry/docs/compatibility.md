# 兼容范围与环境依赖

| 路径 | 本轮完成 | 仍需外部环境 |
| --- | --- | --- |
| 17 / 18 统一 Framework | PyTorch CPU 实际数值注入、触发、OFF、恢复 | 模型专属 Worker 和 NPU 准入 |
| 其余 15 项 | 配置、注册、结构化拒绝和旧入口保留 | 逐项桥接 Provider 参数、触发与租约 |
| Invocation Correlation | typed event、显式 parent ID、身份/generation 校验、乱序和 mock 测试 | 真正 runtime hook 给出的逐调用 Kernel 因果 ID |
| FIA Model Entry | 复用实际 BNSD prefill 准入检查、B/F/OFF mock 调用，原对象不改写 | Native launcher/lease 桥接与真实 NPU 执行 |
| FA 9.1.0 | 显式 SDK/source/policy/dependencies/work 的可移植配方，复用已有 fresh builder | 本机缺少 SDK，未运行新编译 |
| FA 8.5.1 | 版本注册并明确拒绝缺失的 FA 配方 | 新 SDK 适配实现 |
| Native CANN | 注册 8.5.1 和 9.1.0 既有 native_build 入口及 ABI 条件 | 精确部署 manifest 与实机 ABI 检查 |
| Qwen / GLM / DeepSeek | 相同配置/路由和缺少实际候选时的拒绝合同 | 各真实模型权重、Scheduler、vLLM/torch_npu |

模型标签没有选择算子或授予资格的权力。当前 CPU 路由检查实际模型结构、eval 状态和 Tensor device；NPU/CUDA 输入拒绝。compatibility 条目中的 DECLARED_RECIPE_ONLY 不是两版本实机通过。真实运行的 CANN 版本、SoC 和模型路径必须由环境、源文件与 Native 审计绑定。

FIAModelEntry 的 mock_executor 仅接收 CPU Tensor，事件明确为 MOCK。InvocationCorrelator 只接受显式作用域一致的父子 ID；时间相近和相同地址不能建立关联。source=runtime 表示采集器提供的声明，关联器本身不认证设备来源，因此输出始终不授予 NPU 资格。
