# E5 多模型和多 CANN 兼容

公共 Runtime 复用目标身份、触发与恢复合同。适配器按实际算子、选中的 SDK、芯片和构建识别。模型族、版本字符串、同名符号和加载成功只提供上下文。私有 ABI、Host Tiling、Device Kernel 和真实调用链分别取证。

## 当前兼容基线

| CANN 安装版本 | 已有范围 | 当前进入条件 |
| --- | --- | --- |
| `8.5.1` | 910B4/A2 上的标准 Worker。Qwen3-0.6B 的 Fault18/39 Native、DeepSeek-V2-Lite-W8A8 的 Fault20 有限定 prefill C2/C3 回执。另有按原证据范围保留的历史结果。 | 选中 8.5.1，核对对应适配器、Wheel/自定义库、实际 Worker 加载、算子、请求和 Storage。 |
| `9.1.0` | 910B4/A2 上历史 Fault39 Qwen3 TP2 B/F/OFF。另有 FIA 独立构建、Launch、B/F 统计，以及双 Rank 通信校准。 | 为当前源码单独构建和核验实际 Worker。FIA、通信与 MatMul 分别验收。当前 Fault20 与 Fault26 的 Native Build Manager 配方仅接纳 8.5.1。 |
| 其他版本 | 9.0.0、9.2.0 有历史 Fault39 算子级回执。 | 本交付范围固定 8.5.1 与 9.1.0，其他版本按实际安装、适配器和模型单独建档。 |

版本事实和历史范围见 [8.5.1 环境档案](../profiles/environments/cann-8.5.1-prior-124.json)、[9.1.0 环境档案](../profiles/environments/cann-9.1.0-prior-124.json)、[C2/C3 矩阵](../experiments/c2c3_closure_20261117/C2_C3_MATRIX.md)和 [Fault39 版本验收](../fault39/docs/cann_version_adaptation.md)。

910B4 的 A2 归类来自已初始化 Worker 的实际设备名。其他 A2 型号与 A3 需分别建立设备、算子包、目标架构、构建和运行记录。代码遇到无设备观测、其他 A2 或 A3 时标注 `SEPARATE_SOC_PROOF_REQUIRED`。驱动版本、安装包和已映射 ELF 身份保留原值，GNU Build ID 缺失时保留 `null`。 [硬件与加载记录](../docs/cann_hardware_compatibility.md)

| 模型 | 已有范围 | 接入时保留的边界 |
| --- | --- | --- |
| Qwen3-0.6B | 标准 Worker RMSNorm 返回代理、Native MatMul。 | 读实际 Dispatcher、ACLNN 及请求。fused RoPE 需要专门调用链。 |
| DeepSeek-V2-Lite-W8A8 | 标准 Worker 动态 QMM。MoE Unpermute 有真实调用清点。 | 完整 permutation、无损验证、逐目标效果及同进程恢复待证。每次量化另核请求行。 |
| GLM-4-9B-0414 | CANN 8.5.1 TP1 单请求 fresh-prefill 的 FIA26 B/F/OFF。 | 仅原始回执限定的模型、源码、镜像、构建与场景。 |
| SmolLM2-135M | CANN 8.5.1 TP1 单请求 fresh-prefill 的 FIA26 B/F/OFF。 | 仅原始场景。新版本、Decode、全调用分母与批量恢复需分别验证。 |
| Qwen3-VL-30B-A3B-Instruct-w8a8 | 历史 TP2 `restore_offset` 与 patch16 原生 Conv3d 参数代理。 | 原有范围见证据索引。VAE k3 Conv3d 和 DeepSeek MoE 有独立合同。 |

模型依据见 [Worker 矩阵](../experiments/c2c3_closure_20261117/C2_C3_MATRIX.md)、[GLM FIA 回执](../experiments/p3p4_closure_20261117/summaries/fia26-glm-fault.json)、[SmolLM2 FIA 回执](../experiments/p3p4_closure_20261117/summaries/fia26-smollm-fault.json)和 [历史证据](historical_evidence/inherited_pass.csv)。

## 可执行检查

从项目根目录运行。PC 可盘点目录与检查门禁。Linux aarch64 中先选择准确的 `ASCEND_HOME_PATH` 或 SDK 环境，再检查对应的构建输入。

```bash
python -m engineering_closure.compatibility --cann 8.5.1
python -m engineering_closure.compatibility --cann 9.1.0 --fingerprint-libraries
python -m engineering_closure.compatibility --cann 8.5.1 \
  --mechanism 20.buffer_substitute --operator aclnnQuantMatmulWeightNz --inspect-build
python -m engineering_closure.compatibility --cann 9.1.0 \
  --mechanism 39.matmul_range --operator aclnnMatmul --inspect-build
```

`--ascend-root` 盘点指定目录，当前 SDK 的判断仍使用实际选中路径。`--fingerprint-libraries` 读取库 SHA-256 和 ELF Build ID，适合离线盘点。`--inspect-build` 调用现有 `NativeBuildManager.inspect`，绑定已选 SDK 的安装记录，核查头文件、ELF、版本专属导出、工具链和源码。该入口默认只读，不初始化设备或创建 Worker。成功退出码 `0` 表示请求的静态检查成立。退出码 `2` 保留 SDK 缺失、版本或映射不匹配、配方拒绝及工具链错误。

在允许构建的环境运行现有 `fault_injection.native_build` 的 `build`、`verify`、`loaded`，保存每个版本的构建身份、ABI 编译和实际 Worker PID 映射回执。Provider 的原生签名、实际调用、目标身份、效果、OFF 同进程恢复及健康请求由后续验收证明。`runtime_qualified` 在本只读检查始终为 `false`。 [Native Build Manager](../fault_injection/native_build/README.md)

历史证据 CSV 使用冻结基线的 GitHub 永久链接。原始 dump、研究工作区源码、端侧镜像、构建文件和远端原始回执的可用性按每条记录披露。现有摘要可用于继承已记录的原范围，原始数据及历史回执的全部哈希本轮未逐一验证。
