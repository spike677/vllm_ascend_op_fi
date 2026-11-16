# 生产接线准备与验收记录

## 范围

审计基线为上游发布 `9171f3bacd78b1550cc39cc74e126e3a13cfdb07`。执行分支为 `codex/production-provider-wiring-20261116`。按 Git blob 校验原发布快照。另一个工作树的未提交 Native 和覆盖代码按逐文件来源复用。已跟踪及未跟踪实验独立保留。[来源索引](source_manifest.json)

| 路径 | 本次提交内容 | 实机和生产等级 |
| --- | --- | --- |
| Fault20 v17 WeightNz | 发布版 `build20-a1`、旧部署和回执 | 既有 Qwen3-0.6B、SmolLM2-135M 同进程 OFF |
| Fault20 scoped WeightNz 与 QMM | 新 bundle、回调、Worker 范围识别、两次发现、lossless、pre-OFF，最多一次故障尝试 | CANN 8.5.1 aarch64 无设备 SDK 构建、主机 C++/ctypes 和 CPU 资格测试通过。Worker 加载、设备命中与 post-close OFF 待验 |
| Fault39 独立 ABI v1 | 保留 `c1f5726a` 独立包线路与 `9171f3b` 版本适配范围 | 既有四版本算子、部分 Qwen/SmolLM2 实机回执 |
| Fault39 可选宿主资格 | 复用独立 `NativeBackend`，同 Worker 发现两次、self/mat2/dual 无损、单次 F、OFF，无法恢复时保留 owner | CPU 合同测试，通过同一独立 binding 的可选工厂接入 |
| 17 项审计和覆盖 | 现有目录生成源码、合同、证据作用域与缺口。 `QualifiedTargetSet` 支持离线资格和 drain/OFF 对账 | 15 统一合同、1 独立合同、1 缺合同。多目标执行资格待实机 |

## Fault20 实现身份

| 维度 | `build20-a1` v17 | 新隔离回调 |
| --- | --- | --- |
| 构建与 Native | `build20.sh`、原 L4 Run 与 `legacy20_bridge.inc` | `build_qmm.sh`、`fault20_abi_probe.cpp`、`qmm_auto.inc` 和 `matmul_auto.h`。独立 build profile |
| 真实 API 和输入 | `aclnnMatmulWeightNz` self，发布试验保留 0 位尾数 | WeightNz self 保留 3 位，QMM `aclnnQuantMatmulWeightNz` 单独绑定 x1 |
| 回调与隔离 | `dcat20_*`、逐次装载进程 v17 SHM arm | 对应两个 API 的 `dcat20_weight_nz_*`、`dcat20_qmm_*`，持有 Worker 线程和唯一 token，无共享 arm |
| Storage 和代次 | 已锁定的 v17 形参和 live slot，依赖专用 Worker | typed GWS 与当前 Run 真实地址、无重叠、零 offset、cache 注册代次、模型实例与调用身份。故障前重验原输入哈希 |
| 无损与恢复 | 原 v17 无损 D2D 与已发布模型 OFF | 同目标 reference/D2D/OFF 门禁、一调用临时地址、实际 Run 及同步/释放证据。未知同步保留隔离。故障后的完整模型 OFF 在实际 runner 验收 |

两条 scoped API 都要求匹配 CANN 8.5.1 的隔离构建与 Worker。历史 v17 支持和默认部署按原范围工作。两种构建的源码/DSO 身份和回执分别对账。[身份对照](provider_identity.json)

## 宿主准备和运行门槛

黄区的现有 CLI/Catalog/RuntimeBinding 最终把已加载模型及 Worker 映射交给现有 Controller。新 Fault20 通过 `register_worker_matmul` 显式登记可用的构建、实际 Worker 和复跑回调，`ModelFaultBackend.inject` 消费该资格。Fault39 在独立 `fault39.native.matmul` 绑定上通过 `DocumentContractAdapter(..., qualified=True)` 选择资格工厂，需要由真实宿主提供独占 Worker 预约和实际 readback，保留原 native ABI。两条生产路径分别先验证断开、发现、无损、实机命中、OFF 和清理。参见[接线规划](../../docs/unified_model_injection.md#生产接线规划)。

## 验证与阻塞

382 项 Python 回归及主机 C++/ctypes 验证通过。124 在无设备容器完成新的 CANN 8.5.1 SDK 构建，profile 为 `5d5fc76b20b7c5162edf5f1e3677007065a08540f5eeb1cd616cacd10c27532e`，构建容器已移除，原始回执和二进制留在验证机。[验证矩阵](validation_matrix.json)

发布的 26 QMM rank1 首差仍受 NaN 阻断，33 的可观测首差仍在下游。Fault39 既有首差由独立原始统计复核。新采集原件保存在验证机，索引只记录路径、摘要与哈希。[当前阻塞](blockers.md)
