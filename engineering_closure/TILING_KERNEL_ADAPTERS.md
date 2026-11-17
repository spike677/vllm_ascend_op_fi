# E4 Tiling 和 Kernel 适配器

[`kernel_adapters.py`](kernel_adapters.py) 在已有算子路径之上对账源码、构建和实际 Launch。`verify_native_build` 复用 `NativeBuildManager.verify` 和 `verify_loaded`，核对机制、精确算子及 recipe。加载检查绑定实际进程映射和构建键，调用身份、数值效果及 OFF 由相应执行证据提供。

| 故障类型 | 修改点及工程状态 |
| --- | --- |
| Host 参数 | Fault17 使用真实 Conv3d `padding` 参数并验证输出几何。见 [`module_faults.py`](../fault_injection/module_faults.py)。 |
| Tiling 参数 | 现有 FA/FIA/IFA 适配只接受随依赖字段一起生成、受完整构建和调用约束的合法策略。直接修改任意 Tiling 参数的通用机制仍待实现和 ABI 验证。 |
| Tiling 计算逻辑 | FA S1S2 使用原生 Host 控制器选择 M tile/N grouping。FIA Legal 使用 Host M/S2 loops 和 tails 的耦合改动，仍走健康 causal-tail 模板。Fault33 历史 IFA 使用 SInner。8.5.1 TND page16 改用实际 `FAInferTilingData` 专用 compiled entry，对照 workspace 和最终参数。 |
| Device Kernel | FIA Denominator Replay 在第二个 S2 块重复前一块 denominator 质量，精确键为 `119000000000400000`。源补丁和单独构建见 [`fia`](../fault_injection/native_adapters/attention28/fia/)。 |
| Kernel 调度和 Pipeline | launch verifier 对账 block、schedule、stream 及完成同步。主动注入调度与 Pipeline 的独立可执行路径仍待适配。 |
| Reduce 数值行为 | Fault25 同进程组同步 SUM 的原始两次实际输出来生成 measured delta replay。现有实现定位在输出阶段，后端 Reduce Kernel 内部故障需单独构建、入口和数值证明。 |

## 独立构建与 Launch 边界

**FA S1S2** 使用 [FlashAttentionAdapter](../native_execution_strategy/adapters/flash_attention.py) 核对证书、实际 Host/POD 和原始 Launch 资料。证书绑定 `V100R001C11SPC001B243`、`Ascend910B4` 和桥接 ABI。`verify_fa_launch` 只接受该适配器及其真实上下文。FA 当前构建仍依赖已审查的历史工程，面向新 SDK 的可移植 recipe 尚待开发。

**FIA Legal / Replay（9.1）** 分别使用 `legal_tiling` 和 `denominator_replay` 的 [新鲜构建入口](../fault_injection/native_adapters/attention28/fia/build_fresh.py)。`verify_fia_launch` 核对 completion SHA、完整 vendor inventory、public serializer SHA、该机制该阶段的 mode/key/M/S2、实际注册 Kernel 字节与 final argument 原始数据。底层 [Launch verifier](../fault_injection/native_adapters/attention28/launch.py) 要求独立 register/bind/launch 与同调用同步，按真实参数槽选择原始 tiling bytes。Legal/Replay 的 B/F/OFF 与统计首差各自核对。

**Fault33 历史 BNSD IFA（9.1）** 的模型本地入口见 [`deployments/paged33.py`](../fault_injection/deployments/paged33.py)，沿用已锁定的 vendor 校验器、固定 page128 与几何。

**Fault33 TND FIA（8.5.1）** 通过 [`paged33_tnd.py`](../src/demoncat/runtime/vllm_ascend/worker_extension/paged33_tnd.py) 核对实际 TND page16、cache Storage 和请求，及新构建的 `tnd_page16_fai_s2_256`。`verify_paged33_tnd_launch` 使用该变体的 completion、`FAInferTilingData` 序列化器、最终函数键和 workspace 回验。独立 BNSD 和 TND 路线各自保留证据与失败边界。

离线核验读取的是既有构建和观测文件，执行新的 NPU 调用次数为零。数值故障、独立统计首差、实际 Worker 生命周期和 C2/C3 资格仍由对应真实运行验证。
