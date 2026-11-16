# Fault39 历史源码与证据来源审计

审计日期：2026-11-16。范围仅 Fault39；本阶段只读本地源码、Git 对象和用户附件，没有访问黄区、连接 NPU 或重跑历史实验。

## 1. 已找到的可复用基线

本地仓库 origin 为 `https://github.com/spike677/vllm_ascend_op_fi.git`。三个本地目录共享 Git common directory `C:/Users/admin/Documents/demoncatpfi/.git`，并不是三份互相独立的历史证据。

| 记录 | 本地可验证内容 |
|---|---|
| `f95f64ebf1d114f7bc30e0a7cd279feeb5f37deb` | 2026-08-07 初始 Fault20 原型，硬编码 CANN 8.3.RC2 构建配置 |
| `abf690117233062435df73bb66e829017a17c365` | 2026-08-14 引入 Fault39 浮点 MatMul、五种 cancellation、A2 报告和可配置 CANN 8.5.1 构建支持；control ABI v13 |
| `733c6882878a37e14f65d9c03ee2ef2e00829cce` | 2026-08-18 后续 Fault26 相关变更，说明共享运行时代码有其他故障历史 |
| `c8cd66f5364a7950b8831964b5b4e5c73af1cf0c` | 2026-08-27 共享 runtime 硬化，包含 metadata ABI v17 和 control v17 |
| `1eba3308c0cd274825fd777d0a4b6e703560f510` | 审计时原工作树 HEAD；本审计所列历史文件无工作区差异 |

可复用源码位置：

```text
fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/
  buffer_substitute_preload.cpp
  buffer_control.cpp
  fault20_buffer_api.h
  build.sh
fault20_fast_validation/source/l4_metadata_lifecycle_proxy/
  libopapi_metadata_proxy.cpp
  build.sh
FAULT20_CANN851_USAGE.md
```

`abf6901` 的真实 Git diff 同时加入以下适配，因此不必退回最早的 8.3.RC2 实现重新解决构建问题：

- `FAULT20_ASCEND_TOOLKIT` 参数化 toolkit 根目录。
- `FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT` 经编译宏进入 runtime 的真实库路径门禁。
- 两个 `build.sh` 都增加 `-I"$toolkit/pkg_inc"`。
- `FAULT20_REAL_LIBOPAPI` 参数化真实 `libopapi.so` 路径。
- 增加浮点 `aclnnMatmul` 的符号检查和加载检查。

历史使用文档中的 CANN 8.5.1 显式参数为：

```text
FAULT20_ASCEND_TOOLKIT=/usr/local/Ascend/cann-8.5.1/aarch64-linux
FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT=/cann-8.5.1/
FAULT20_REAL_LIBOPAPI=/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so
```

脚本的旧默认值仍是 8.3.RC2；默认值不代表缺少 8.5.1 适配。新交付应显式锁定发现并验证的 toolkit/库路径。`pkg_inc` 修正可由源码 diff 验证；`pkg_inc/base/dlog_pub.h` 在当时机器上的实际位置和 8 月 10 日运行结果来自用户历史说明，本机未持有该机器的头文件或原始运行记录。

逐文件 Git blob ID、原始 blob bytes 的 SHA-256 和附件 SHA-256 见 `../contracts/provenance.json`。当前所列 Git blob bytes 与原工作树相应文件 bytes 的 SHA-256 一致；不能凭平台假设换行格式后重新生成来源 hash。

## 2. 搜索范围与尚未找到的原件

已查：

- `C:/Users/admin/Documents/demoncatpfi`，包括 ignored/hidden 文件名及相关文本。
- `C:/Users/admin/.codex/worktrees/batch-outlier-msprobe-publish/demoncatpfi`。
- `C:/Users/admin/.codex/worktrees/fault28-native-fa-publish/demoncatpfi`。
- 共享仓库所有本地 refs、remote-tracking refs、reflog 可达对象及路径历史。
- `git fsck --connectivity-only --no-reflogs --unreachable` 的 13,258 个不可达对象中全部 1,617 个 tree 的文件名记录；不可达 commit 数为 0，目标文件名无命中。没有将所有孤立 blob 的内容扫描当作已经完成。
- 本地 attachments 中上述历史文件名、`aclnnMatmul_5015` 和 A3 adaptation 标识；只在本次附件及其索引中命中。

以下精确目录/文件名没有在上述来源中找到：

```text
source/l4_buffer_substitute_cann851_v1/
source/l4_metadata_lifecycle_proxy_cann851_v1/
runtime/cann851_validation_v1/
fault20_cann851_dynamic_path_audit.md
fault20_cann851_lossless_validation.md
fault39_a3_matmul_cancellation_validation.md
fault39_a3_matmul_cancellation_validation.json
```

这表示本次未取得这些原件，不表示历史工作没有发生。8.5.1 所需构建改动已经在可追溯的 `allhit_v1` 和 metadata proxy 源码中找到。A3 专门报告的结论、`A2 .v1 → A3 .v2 diff=0` 和相同二进制 hash 声明暂记为 `user_provided_history_claim`，不提升为本地可验证的源文件或原始运行证据。

本地 `fault39_a3_validation_handoff.md` 是 2026-08-14 提交的后续验证方案，明确写明当轮未连接/操作 A3；它不是附件所指的 8 月 17–18 日 A3 完成报告。二者不能合并。

## 3. 新附件的来源与修正

用户提供附件：`f166ff77-01c4-44b1-a547-c0fda9bfcf73/已粘贴的文本.txt`。

SHA-256：`58cf917386f0ec40622fd12854e798272cdc3a0f26acc6797ad26b1df4ab71ab`。

附件提供以下历史参考：A3 half 输出 finite=1120、zero=1120、+Inf=2208、-Inf=1792；其余四 pattern 为全零，并说明源代码与构建产物相同。保留其参考价值，来源状态见 `reference_profiles.json`。

两处必须修正：

1. **finite 包含 zero**。源码 `scan_qmm_output` 先排除 NaN/Inf，再累加 finite；其中为零的元素另累加 zero。`zero` 是 finite 的子集，不是额外互斥分类。总数应为 `finite + nan + positive_inf + negative_inf`，不能再加 zero。A2/A3 的 finite=zero 表示非零有限值均为 0。
2. 附件以新 ABI “例如 v14”为示意，但本地共享 ABI 已到 v17。新的 Fault39 控制面必须采用独立 magic、SHM name、ABI namespace/version 和布局校验，不复用旧 Fault20 v13/v14/v17 的名字或布局。

## 4. 历史 Fault39 的真实边界

真实注入路径是 typed `aclnnMatmulGetWorkspaceSize` metadata proxy，加私有 `AddParamToBuf`/`AddOpExecCache`/`op::internal::OpExecCache::Run` interposer。公开 `aclnnMatmul` execute wrapper 负责转发与日志，输入替换发生在当前 Run 的地址 vector 边界。

Proxy 通过 `LD_LIBRARY_PATH` 放入 `libopapi.so`，并用重命名 SONAME 的真实 `libopapi_real.so` 转发；cache interposer 通过 `LD_PRELOAD` 加载。它不是仅拦截一个公开 ACLNN 函数的通用实现。

关键源码定位（对应来源 HEAD）：

| 能力 | 文件与符号/原行号 |
|---|---|
| typed metadata | `libopapi_metadata_proxy.cpp:330`, `capture_tensor_metadata` |
| GWS/cache 捕获 | 同文件 `:963`, `aclnnMatmulGetWorkspaceSize` |
| 注册 metadata | `buffer_substitute_preload.cpp:5985`, `fault20_float_matmul_gws_commit_v17` |
| slot proof | 同文件 `:1600`, `prove_matmul_storage_order` |
| runtime gate/selector | 同文件 `:5159`, `run_and_maybe_substitute` |
| cancellation layout | 同文件 `:2864`, `float_matmul_cancellation_metadata_valid` |
| cancellation pattern | 同文件 `:2907`, `fill_bf16_cancellation_mat2` |
| temporary buffers | 同文件 `:3105`, `run_float_matmul_finite_dual` |
| observer | 同文件 `:2731`, `observe_float_matmul_output_after_run` |
| output scan | 同文件 `:2625`, `scan_qmm_output` |
| 私有 Run 导出 | 同文件 `:6062`, `OpExecCache::Run` |

历史 slot 0/1/2 是已验证 CANN 收集契约、typed formal 收集顺序、storage presence/uniqueness、cache 关联和当前 vector 长度共同证明的结果；不是可跨版本假定的公开 ABI。冷态 GWS device address 在报告中与目标热态 Run address 均不相同。未来124必须重证符号、库版本、metadata、slot 和 lossless。

历史代码强制 aarch64、C++ ABI0；库路径片段和符号存在性检查不等同于完整语义 ABI 验证。新版本不能将旧实验二进制 SHA-256 当作新构建的验证结果。

## 5. 五 pattern 与数值契约

历史已实现 mirrored 规则，而非仅名称相似：`self[k]=+2^58`，每列 mat2 的前后 K/2 项成对反号。block256/512/1024 在前半按块交替，后半逐点反号；half 前正后负；alternating 偶正奇负。

原实现固定 `M=1,K=12800,N=5120`，self stride `[12800,1]`、mat2 stride `[1,12800]`、offset=0、BF16、storage rank=1、完整 storage 覆盖。mat2 寻址是 `offset+k*stride0+n*stride1`。不能只删常量就声称任意 shape/stride 已支持。

泛化构造需要逐列精确平衡，拒绝奇数 K；禁止将全 K 普通重复 blocks 截断，因为截断可能破坏零和。设 `H=K/2`、`A=2^e`，受控选择整数 e，同时证明：

```text
A 为有限 BF16
A² = 2^(2e) 为有限参考 FP32
H × A² > FP32_MAX = (2^24 - 1) × 2^104
```

采用精确整数/幂比较，不用发生浮点溢出的计算结果作门禁。对于严格二次幂 A，`e<=63`，因此 `K=2/4/6` 均无解；最小可行偶 K 为 8。固定 `e=58` 在 `K<8192` 不满足该参考同号半和压力条件。同轮所有 pattern 必须锁定同一 K、A、dtype 和 target profile。该压力条件不保证真实 kernel non-finite，不证明硬件 accumulator 的 dtype。

如果小 K 使某些 block pattern 与 half 相同，应记录实际序列/等价关系；不能声称获得了五个不同排列。不同硬件得到全零或其他有限分布时，应保存实际结果，不能用 A2/A3 参考计数作为固定验收阈值。

## 6. 生命周期与泛化缺口

旧数值路径复制调用方地址 vector、pre-sync、D2H 到 pinned host、填充、分配 device temporary、同 stream H2D async、真实 Run、post-sync、释放，再从原 output slot 观察。原模型 storage 不被永久修改。

- post-sync 失败时保留可能仍在使用的 buffers，避免提前释放；旧代码没有完整异步回收队列。
- `disarm` 仅 `enabled=0; generation++`，没有等待在途调用；不能承诺返回即完全停止。
- arm-lifetime reservation 没有完整 generation fencing；新实现应冻结不可变调用配置，阻止 disable/re-arm 交叉使用状态。
- non-finite observer 会 disarm；有限成功 case 主要靠 single-hit cap 阻止后续注入，仍需显式 disarm。
- 原 selector 是 arm 后第 N 个固定 shape 候选，没有显式 model-layer 或 decode phase 证明。
- 动态 runtime 地址需检查别名/重叠，cache registry 需明确失效与地址复用策略。
- 泛化 layout 必须证明逻辑寻址边界、无覆盖冲突、算术无溢出、format/dtype、bytes，并保留 view 之外 storage 内容。
- `input_finite_after` 在旧日志中是已知构造推导的元素数，不是 H2D 后再次设备 readback；新证据应区别构造保证与回读实测。

## 7. 证据分层与交付边界

`locally_verified` 表示本机找到了可检查的 Git 源码或历史报告，并核对其 hash；不表示本轮重新运行了历史 NPU 实验。原始 A2 JSONL 和容器证据仍未本地取得。A3 参考值目前为 `user_provided_history_claim`，待原报告或独立新实验补强。

历史 A2 cancellation 支持 `MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED`，不支持 `ACCUMULATOR_OVERFLOW_PROVEN`。QMM lossless 不能替代浮点 MatMul lossless；output Inf/NaN proxy 也不能替代自然 cancellation 证据。

P1 交付的是独立 Fault39 core、接口文档驱动的 adapter 和 mock 契约测试。当前无法取得真实 `src/demoncat` 内源源码，因此不能称为已经完成该源码的 catalog/binding/controller/worker 集成。蓝区124的 P2–P6 原生/模型验证须独立形成证据；黄区后续只在真实框架可见时完成接线验收。本审计不包含黄区操作。
