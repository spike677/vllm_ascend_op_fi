# Worker Extension 架构

Worker 由 vLLM/vLLM-Ascend 创建和运行。扩展类通过 `--worker-extension-cls` 加入宿主 Worker，读取已经初始化的上下文，通过宿主现有 RPC 执行本进程故障控制。部署环境在进程启动前提供 Native Runtime。宿主 Controller 负责跨 Worker 的选择、命令发送与结果汇总。

```text
Serving 启动配置 ── Worker 构造前 ──→ Python 扩展可导入 / 匹配的 Native preload
现有 Controller ── collective_rpc ──→ 已初始化 Worker / DemoncatWorkerExtension
                                           ├── 当前 rank、NPU 与模型实例
                                           ├── Worker 本地租户状态
                                           └── 现有 Provider → 实际 Native Runtime
```

## 现有实现审计

蓝区基线可审查的是既有 Fault39、26 和量化扩展及实验回执。黄区现有 `DemoncatWorkerExtension` 的实现与正式 Controller 映射仍待在实际宿主复核。

| 现有入口 | 实际行为 | 可复用的边界 |
| --- | --- | --- |
| `Fault39WorkerExtension.fault39_rpc` | 单一 namespaced 方法，经真实 serving RPC 接收动作。`prepare_worker` 从 `self.rank`、`self.device` 和 `torch.npu.current_device()` 核对身份，使用 `CtypesProvider.attach_preloaded` 和实际 Native 快照。 | 保留 Provider、进程内 rank 绑定、真实快照、owner/generation、disable/clean 机制。39 的 RPC 动作和 Native ABI 留在机制层。 |
| `fault39.tools.vllm_validate` | 用服务执行器的 collective RPC 调用旧扩展，按响应里的 rank 匹配 Worker。 | 可沿用真实 serving 端对照方法。执行器对象的可达路径需要按目标版本核验。 |
| `Fault39.install_runner_scope` | 旧验证入口在已初始化 runner 实例上包装 `execute_model`，检查 `scheduler_output` 签名，把请求 ID 送入 Provider context。只有单请求可直接归属，多个 ID 只表明 scheduler 请求集合。 | 请求和调用归属的证明条件可复用。新通用扩展的 MRO 和 attach 流程不安装该包装。 |
| `Residual26WorkerExtension` | 旧实验以 `dcat26_install/remove` 打开真实模型的 `NativeSession`。runner 包装提供捕获窗口，实际原生 Provider 检查实时分配、writer 和 consumer。 | 复用 Provider 证明及清理边界。试验专用 RPC 和捕获留在既有路径。 |
| `QuantWorkerExtension` | 旧实验按签名及源码 SHA 核验实际 runner，并由 `execute_model`、`_prepare_inputs` 和量化方法窗口建立请求行映射。 | 量化作用域需要独立适配与验收。Worker 的通用身份无需依赖量化 runner 内部接口。 |
| Fault20/33 Provider | `Buffer20Runtime` 与 `Paged33Runtime` 在匹配部署中执行机制专用入口。新 scoped MatMul 回调和工作负载回放另有资格校验。 | 通过已审查的 Provider/profile 绑定进程内模型、调用、证据与恢复。保留不同实现身份。 |

[Fault39 扩展和 runner 审计](../../fault39/src/demoncat_fault39/vllm_worker.py) · [真实 Worker 验证调用链](../../fault39/tools/vllm_validate.py) · [Residual26 旧实验扩展](../../fault_injection/deployments/residual26_worker.py) · [量化旧实验扩展](../../fault_injection/deployments/batch_scale.py) · [部署 Provider](../../fault_injection/deployments/README.md)

## 通用扩展

入口类为 `demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension`。该类由 serving 注入到真实 Worker 的 MRO。方法仅增加 `demoncat_*` 扩展入口。构造、设备初始化、模型加载、常规 `execute_model` 和退出沿用宿主，各动作需要时才解析真实 Worker 上下文。包独立安装，通用导入无需初始化 NPU。

扩展层只处理身份、附着、动作分派和结果。当前桥接按现有机制 catalog 解析 `20.buffer_substitute` 与 `39.matmul_range` 的版本 1，再把 Worker、实际模型及身份交给相应 Provider。两个 Provider 复用既有 Native 接口与验证逻辑。排他的 Worker 回放能力由部署提供，缺少可验证调度、scope、profile 或实时 Native 快照时按对应范围返回不可用。Fault26 residual、Fault26 FIA、共享量化和 Fault33 的既有机制与证据继续沿用原路径，通用扩展接线仍待完成。

| 已接线的 Provider | 准备及查询 | 允许执行前的条件 |
| --- | --- | --- |
| Fault20 WeightNz / QMM | 检查正确的进程 preload 符号、实际 eager Worker、算子子路径和匹配构建。 | 实际模型的 `QualifiedNativeBinding` 支持排他 Worker 回放，绑定与 Worker、算子和构建一致。Provider 校验可定位调用及无损性，执行结束用同一回放确认 OFF 输出。[Fault20 桥接](../../src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault20.py) |
| Fault39 MatMul | 从已加载 DSO 连接既有 ABI，核对实时快照、rank 和 profile。 | 部署提供 `Fault39WorkerReplay`，以实际 scheduler 请求 ID 限定原生 context，使用同一排他预约完成自动选点、无损回放及单次故障。[Fault39 桥接](../../src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault39.py) |

重新附着以当前 Worker、当前模型实例及本次租户为准。旧进程或旧模型的 owner/generation 不会自动恢复。OFF 与恢复核对需要真实 Native 状态和原机制证据。Worker 重启、资源回收及运行健康管理继续由 serving 负责。[附着合同](../contracts/worker_extension_contract.md) · [部署](deployment.md) · [兼容性](compatibility.md)
