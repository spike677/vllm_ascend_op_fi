# 蓝区 Worker Extension 验收

通用扩展以独立 wheel 部署。vLLM-Ascend 负责原有 Worker 生命周期，扩展在 Worker 内提供身份、能力、附着、Provider 控制、原生查询和退出六个 RPC。加载扩展和读取静态能力无需引入 Torch 或 Provider。身份根据当前进程、Worker rank、NPU context 和加载的模型对象获取。Worker 更换后必须重新附着。

## Provider 与门禁

| 路径 | 接入方式 | 执行前的宿主证据 |
| --- | --- | --- |
| Fault20 WeightNz／QMM | 复用现有 Runtime 注册表、隔离版 CANN 回调、无损资格流程和 Opt-in 单目标执行。新增只读 C++ 状态，读取已加载构建、回调、执行和租约。 | 实际加载的模型和 CANN profile 对应，部署绑定提供独占回放与完整模型输出，原始输入存储核验通过。 |
| Fault39 MatMul | 使用同一个扩展，由独立 ABI v1 Provider 读取预加载状态，复用单点资格后端。 | 真实 Worker 预约、实际调度请求 ID、相同输入、原始存储和完整模型输出回执。 |
| Fault26 两种变体、共享量化尺度、Fault33 | 继续使用各自当前发布的 Provider 和实验路径。 | 接入共用扩展仍需核验真实调度、目标识别和恢复能力。 |

Worker 预加载由启动环境管理。两个验证包分别代理 `libopapi.so` 和 `libopapi_nn.so`，每个实机案例只部署其中一套。扩展查询原生状态失败时返回 `NATIVE_STATE_UNAVAILABLE`。缺少可验证的独占回放时，Provider 拒绝注册和注入，解除附着仍要求原生空闲已经核实。

## 验收方法

在 124 的 vLLM-Ascend 0.18.0、CANN 8.5.1 镜像中，使用分开的 vLLM Worker 进程和 Qwen3-0.6B BF16 eager。通过 vLLM 的真实继承扩展入口和 `LLM.collective_rpc` 观察 PID、rank、逻辑 NPU 和模型身份。用固定文本生成两枚 token 核对附着、未激活的 Provider 和解除附着期间的健康输出。该检查属于真实 Worker 加载、原生状态查询和回退控制。Fault20／Fault39 的真实故障命中与注入后的 OFF 恢复需要各自的可信宿主回放。

实机控制器复用 124 的卡2／5资源锁和启动前设备门禁。独立构建使用无网络、无 NPU 设备的容器。原生包由构建清单和镜像环境校验。发布记录仅包含摘要和哈希，原始运行回执与租户票据留在验证主机。

同一套扩展连接实机的 `vllm_ascend.worker.worker.NPUWorker`。卡5映射为逻辑设备0。可见设备变量未设置时，回执显式保存 `null`。进程重建后，旧附着票据和旧 Fault39 owner 票据均返回 `WORKER_IDENTITY_MISMATCH`。

## 已验收范围

| 用例 | 结果 |
| --- | --- |
| TP1 无 Native 预加载 | 扩展进入真实 Worker 的继承关系，RPC 识别逻辑设备0和 rank0。两个 Provider 都返回 `NATIVE_PRELOAD_NOT_ATTACHED`，租户可以解除附着。 |
| TP1 Fault20 | 预加载的两套回调报告实际构建匹配。默认 `aclnnMatmulWeightNz` 路径的准备、空闲回读与解除附着通过。未提供可信独占回放的注册返回 `FAULT20_TRUSTED_EXCLUSIVE_REPLAY_REQUIRED`。 |
| TP1 Fault39 | 预加载 ABI v1 原生回读报告 `MATCHED_BUILD`，准备、空闲回读与解除附着通过。未提供可信调度回放的注册返回 `FAULT39_TRUSTED_SCHEDULER_REPLAY_REQUIRED`。 |
| 重建 Worker | 从另一真实 Worker 带来的旧附着和旧 owner 票据被拒绝。同一输入的两枚生成 token 保持 `[12095, 13]`。 |
| TP2 双卡 | 两个独立进程通过真实 collective RPC 分别返回 rank0／逻辑设备0 和 rank1／逻辑设备1。两次正常生成 token 一致。此用例只验证身份与读取。 |
| 本地回归与构建 | 402 项 Python 测试通过，其中 20 项覆盖新扩展及 Provider。C++／ctypes 回归及 124 无设备 Fault20 原生构建通过。 |

两种有预加载的用例分别运行于新容器，实际返回 `native_registered=false`、`enabled=false`、`inflight=0`、`leases=0`。无预加载、准备后和解除附着后的同一输入生成结果一致。最初启动用例因启动器丢失 CANN 的 Python 搜索路径失败。保留 toolkit 原路径后重试通过。[回执摘要及校验](validation_summary.json)

## 发布与校验

本次分支增量包含实现、测试、文档和验收摘要。已有原始统计及本轮原始运行回执留在 124。`validation_summary.json` 记录实机回执、构建和 wheel 的 SHA-256。发布文件可以从仓库根目录校验。[本次发布文件校验清单](SHA256SUMS)

```sh
sha256sum -c experiments/worker_extension_20261116/SHA256SUMS
```

## 故障执行缺口

目标版本的 `collective_rpc` 调用已有 Worker 方法，真实 `SchedulerOutput` 通过原有执行路径进入 Worker。普通控制 RPC 缺少真实请求范围。Fault20 还需要完整模型输出和原生输入保护回执，Fault39 需要在同一模型线程关联实际调度请求并独立读取原始 `self`／`mat2` 存储。缺少这些宿主能力时，当前接口验证拒绝注册。两条机制通过共用扩展的实际故障命中、注入后 OFF 恢复与重新取证均待验收。既有独立实验证据按原版本与调用链保留。[vLLM 调度执行顺序](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/engine/core.py#L345-L372) · [Worker RPC 接口](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/entrypoints/llm.py#L594-L632)

[Worker 附着契约](../../docs/contracts/worker_extension_contract.md) · [部署](../../docs/worker_extension/deployment.md) · [兼容性](../../docs/worker_extension/compatibility.md) · [此前独立 Provider 实机证据](../provider_adaptation_20261116/README.md)
