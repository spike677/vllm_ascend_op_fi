# Worker Extension 兼容性

按实际运行组合登记资格。Worker 扩展加载、MRO、RPC、Native preload、真实算子命中和生产宿主接线分别验收。

## vLLM-Ascend 接入

vLLM v0.18.0 的 Worker wrapper 解析一个扩展类全名，将其加入 Worker 的继承关系。加载器检查扩展的全部非双下划线属性，包括私有辅助方法和类属性。扩展只定义命名空间化 RPC 方法，辅助实现放在模块内。服务已有另一个扩展时，先核对该镜像的组合能力及方法冲突，保存原 serving 接口的回归结果。[vLLM v0.18.0 Worker 加载实现](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/worker/worker_base.py#L228-L254)

Ascend v0.18.0 的 Worker 初始化后可选择不同 runner 类。以实际 Worker 中的 rank、当前 NPU context、已加载模型、runner 类型和已打开功能判断支持。包版本属于部署记录，不能替代上下文验证。读取已初始化的模型对象与检查实际请求归属是不同的证据。[Ascend Worker runner 构造](https://github.com/vllm-project/vllm-ascend/blob/v0.18.0/vllm_ascend/worker/worker.py#L284-L295)

| 维度 | 需要核对 |
| --- | --- |
| Python 包与 MRO | 在实际镜像独立安装并导入，目标 Worker 正常启动，扩展和宿主方法无冲突。 |
| RPC | 通过现有 serving RPC 调用，按回复的 Worker/rank 身份对账。列序不能充当 rank。 |
| Worker 与设备 | 真实 PID、进程启动标识、rank、逻辑 NPU、visible device 和模型实例。无法观察的字段明确记录。 |
| TP 与 DP | 相同 rank 在不同数据并行组可对应不同进程和模型实例。以实际实例和进程标识联合匹配，分别验收组合。 |
| Native | 目标 Worker 的执行进程内可见指定映射、动态符号、构建指纹、原生能力和快照。 |
| 重启 | 真实宿主创建新的 Worker 后旧租户和 generation 失效，按新身份重新 attach。 |
| 隔离与恢复 | 缺证据或身份发生变化时拒绝启用，清理只作用于本租户。实际不可用状态保持未知。 |

## 证据与待验范围

| 路线 | 已有证明 | 本次适用范围 |
| --- | --- | --- |
| 旧 Fault39 独立扩展 | CANN 8.5.1 上 vLLM 0.18 的 TP1/TP2，CANN 9.1 上 vLLM 0.23 的 TP2。采用既有独立扩展、scheduler scope 和对应真实模型验证脚本。 | 旧试验回执作为部署基线。通用扩展的 MRO、RPC 与故障执行各自验收。 |
| 旧 26、共享量化扩展 | 对应固定镜像、runner、TP 与工作负载内完成机制实验。 | 保留旧路径的证据，新增通用扩展接入需确认实际请求及可用 scope。 |
| Fault20 MatmulWeightNz | 发布的 v17 与两模型验证路径配套。 | 新隔离回调是单独实现，安装、实际加载和单次命中均单独验收。 |
| 新通用扩展的 CPU 用例 | 14 项扩展测试通过，覆盖模拟 MRO、身份、租户隔离、Native 状态与拒绝路径。使用带 PyTorch 的本地环境，6 项 Provider 模拟测试全部通过，含 Fault20 回放及 OFF 重试。 | WSL 验证 C++ 与 ctypes 接口，124 在无设备环境重编匹配 CANN 的 Fault20 Native 状态接口。真实 Worker 的加载、状态与执行证据分别登记。 |
| 新通用扩展的真实 Worker | 124 的 vLLM-Ascend 0.18.0、Qwen3-0.6B BF16 eager。TP1 通过 Worker 加载、RPC、缺少预加载拒绝、原生空闲查询和重启失效。TP2 通过两个真实 Worker 的身份与健康输出复核。 | TP2 变更动作、其他模型／镜像版本和 DP 需要单独核验。设备上的 Fault20／39 故障命中与注入后 OFF 恢复需要实际排他回放。[实机报告](../../experiments/worker_extension_20261116/README.md) |
| 黄区宿主 | 现有蓝区有宿主接线契约。 | 以实际 `src/demoncat`、Controller、Registry 与目标 serving 镜像复核后验收。 |

[旧 Fault39 实机报告](../../fault39/reports/README.md) · [CANN 与 vLLM 构建范围](../../fault39/docs/cann_version_adaptation.md) · [既有 Provider 实测记录](../../experiments/provider_adaptation_20261116/README.md)
