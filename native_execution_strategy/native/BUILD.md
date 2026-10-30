# 版本固定的 FA 私有桥接构建

本目录只包含 host 桥接、公开 POD 解码器、CPU 检查及私有构建工具。模型、SDK、NPU kernel 二进制均不随源码提交。新桥接不继承旧实验的实机验收，初始注册状态必须为 `experimental`。

## 固定输入

- 官方 ops-transformer v9.1.0 commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`，来源 `https://gitcode.com/cann/ops-transformer.git`。
- opbase `cd48c600c4b37146f504217eab2bf3956f7fee8b`；ops-tensor `b24535e30a6f7a3909c53c8b09713a9c0d08e54b`。
- CANN 9.1.0 / `V100R001C11SPC001B243`，固定构建镜像 `sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9`。
- JSON 单头 SHA `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6`；protobuf-25.1 archive SHA `9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42`；abseil-20230802.1 archive SHA `987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed`。
- 构建 Python 包版本见 `build_requirements.txt`。依赖复制到新任务私有目录，不安装到系统。

`build_cpu.py` 记录本次实际构建配方。它从既有的固定官方源码缓存创建独立新 checkout；旧任务路径仅用作只读材料缓存，没有链接旧实验行为，也不会在旧目录构建。另一台主机复建时应从上述官方 commit 和固定依赖取得同样材料，替换脚本的缓存路径及新的任务根目录；不能拿移动的 master 或未知 cache 替代。

## 源码修改位置

`apply_fa_bridge.py` 拒绝错误 commit、已有 tracked diff 或已应用补丁的 checkout；它修改两个 host CPP、加入 `fa_bridge.h` 和 `fa_plan.h`，并生成完整 patch 与每个文件的前后 SHA。

原始 `flash_attention_score_tiling_general.cpp` 的 S1S2 override 先计算 M/S2（3255–3266）和 nRatio（3268–3283），这两个 override 尚未生成 M 依赖的库 tiling。新 `SelectFactors(context,M&,R&)` 位于 CalcNRatio 末尾，核验完真实 context 和本次原生整对后一次选择整对；拒绝时两值都保留。随后原生 MatchTemplate、SetCore/MultiCore/TensorSize/Sparse 与 DoLibApiTiling 继续计算所有依赖字段。

该结论只适用于已固定的 S1S2 override。普通基类的 CalcS1S2 搜索会较早调用 Matmul tiling，不能把这个选择点无条件推广到其他模板。逐行顺序证据保存在 `build_evidence_a1/source_order_audit.json`。Kernel 源码未修改，输入张量与输出张量不经过桥接读取或改写。

## CPU 构建和私有安装

实际新根目录为 `/home/native-execution-strategy-20261030`，源文件放在其 `native/` 下，构建驱动拒绝已有 source generation。容器使用 4 CPU、16 GiB、512 PIDs、1800 秒上限，不传入任何 NPU device，也不启用 privileged。它先进行 header 语法、typed-plan/原子选择、scope/公开 shape 和 205 字段 POD roundtrip 检查，再构建官方 package。

关键 CMake 参数为：`ENABLE_BUILD_PKG=ON`、`ENABLE_BUILT_IN=OFF`、`ENABLE_OPS_HOST=ON`、`ENABLE_OPS_KERNEL=OFF`、`VENDOR_NAME=native_execution_strategy`、`ASCEND_OP_NAME=flash_attention_score`、`ASCEND_COMPUTE_UNIT=ascend910b`。完整 argv 在 `build_cpu.py` 和构建回执中。包包含原版 kernel 的 JIT 源材料；CPU 构建不证明设备端编译和运行已通过。

`stage_cpu.py` 仅在构建回执 PASS、源码未变、构建容器已移除后运行无网络、无设备安装容器。它拒绝覆盖已有 stage，不使用全局安装或 `--force`。最终位置：

```text
/home/native-execution-strategy-20261030/native_vendor_stage_a1/vendors/native_execution_strategy_transformer
/home/native-execution-strategy-20261030/native_vendor_stage_a1/vendor_manifest.json
```

清单 schema 为 `nes.fa.private_vendor.v1`，逐文件记录 SHA/bytes/链接目标；运行前由独立 launcher 重新核对全部材料。清单和证书真实 SHA 由 launcher/注册表绑定，C++ 不做自身 SHA 的循环嵌入。

## 检查边界

CPU fixture 中的证书 SHA 是明确标记的合成值，只检查 JSON 结构和原子选择逻辑，不构成真实证书。CPU tests 覆盖 native 正 batch 透传、B2/B4 controlled 拒绝、重复匹配全部选择、原生整对或 context 不符时完整透传，以及 PID/TID/phase/version/dtype/precision/shape/JSON 错误拒绝。

新 POD schema 名称为 `nes.fa.public_pod_layout.v1`，其 1464 字节尺寸、类型、对齐、205 字段和 sections 与旧公开 SDK 布局逐项一致，只有 schema 元信息更名。每次实际运行仍须核公共 context raw 与实际 launch raw 完全相同，并核模型/算子输入输出与统计透明性；不能由 CPU 检查、计划声明或命中记录代替实机证据。
