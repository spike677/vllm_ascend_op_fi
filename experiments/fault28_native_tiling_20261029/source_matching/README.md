> 本页保留最初FIA源码匹配与CPU探索记录。最终使用真实HF原生FA路线；a2构建、实际ABI/分块验证及任务私有高精度统计结论见[最终FA策略说明](fa_policy/README.md)与[实验README](../README.md)。原始阶段性记录不改写为最终验收。

# 故障 28：安装包与官方源码匹配

2026-10-29。状态：**`KERNEL_SOURCE_MATCHED_HOST_ABI_PENDING`**。已经取得强匹配的官方 release 候选；原版私有 OFF 包已在无设备 CPU 容器中构建成功并安装到任务私有目录，尚未证明完整 stock host tiling ABI。本目录研究没有启动 NPU 计算，也没有安装/替换系统库。主实验另有原生 synthetic 校准，以下明确区分其运行证据与本目录的 CPU 证据。

FA 路线的新进展见 [FA 私有策略与解码证据](fa_policy/README.md)：同 binary 的 OFF/ON typed host 已完成 CPU 构建与私有安装，FA 官方 POD 为 1464 字节，真实 stock 日志 112 份候选 raw 已解码，SDK 原生 key 宏核对通过。此项与前述 FIA 原版 OFF 包分别记录；两者均不能仅凭构建成功宣称实机透明或故障复现。

## 已确认的版本与字节证据

- 官方仓库：<https://gitcode.com/cann/ops-transformer>。
- 标签 `v9.1.0` 对应提交 **`c77c0829f1f0f0b8f882db1d9b051a758b605472`**，提交时间 2026-07-30 16:25:10 +08:00。不是 master，也不是后续移动的 `9.1.0` 分支 HEAD。
- 冻结镜像 `sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9` 内 `ops_transformer/version.info` 为 `9.1.0`、`timestamp=20260730_231653901`。
- 已安装 PFA 32、FIA 81、IFA 13 个 kernel 源文件和 43 个可映射 attention 公共文件，**169/169 与该提交的 Git blob 逐字节相同，0 个不同**。另外 385 个已枚举的 common 文件未在这次按 attention/op_kernel 路径映射，不能计为匹配或不匹配。
- 三个前期候选 kernel 的 SHA 全部相同；PFA/FIA tiling key 头和 `fia_tiling_data_public.h` 也相同。Git blob 比对避免了 Windows 工作区换行转换干扰。

原始安装元数据和 SHA 在 [installed_manifest.json](installed_manifest.json)；逐文件比较在 [source_hash_comparison.json](source_hash_comparison.json)；host 候选源、构建文件指纹和剩余门槛在 [source_match_report.json](source_match_report.json)。本地稀疏 clone 被 `.gitignore` 排除，可由上述固定 commit 恢复。

| 文件 | 安装文件 / 官方 Git blob 的共同 SHA256 |
|---|---|
| `prompt_flash_attention_var_len_score_sab.h` | `9a4a3777adf643832eb2c1c7ffca77b45faddd5ee004a4ac07c7b5e5799ba340` |
| `prompt_flash_attention_s1s2_bns1_mla.h` | `86216236ab253c185a782577dd18716cd34143d0d90469c3a489e38373fc228d` |
| `prompt_flash_attention_bnstilling_n_s_no_tail.h` | `b1988b58048beb6b6c6d7f65640598853f8015ee7bc86bbf7889d98a76300074` |
| PFA `prompt_flash_attention_tilingkey.h` | `c5b7ffa59784b248091789cb3f30385aa2d12511e80400e9faf908279737445b` |
| FIA `fused_infer_attention_score_tilingkey.h` | `ca5dd685ba91365d0ec9cdc22a4026a71789d54cb3dcc473f4e5c9e987b3e974` |
| 公共 `fia_tiling_data_public.h` | `665e7d6ffacb835f68ddccb79338e4beff2f361c1a5638cb82cc66e33777ed78` |

## Host 证明边界

安装 `libophost_transformer.so` 的 SHA 为 `3f5758ed76ce537f3d005bb03b55e3030b210e206f5cb9809ab4346b98bae33b`；`.comment` 为 `GCC: (GNU) 7.3.1 20180303`。检查未发现 GNU build ID、debug/symtab 或明确源 commit 标记。它包含该 release 相应 FIA/PFA host 文件名与类型字符串，并导出 `DeviceDoOpTilingFusedInferAttentionScore`，但字符串对应不能证明 C++ 实现逐字节相同。

安装目录中未找到完整 `PromptFlashAttentionTilingData` / `FusedInferAttentionScoreTilingData` host 结构定义。公共数据头相同不等于整个序列化结构 ABI 已验证。因此当前不能称“已精确复现安装 host 二进制”，更不能依据字段猜测修改 opaque executor 或 raw buffer。

准入验证建议：在同一冻结镜像内构建未修改提交，给安装 host 与重建 host 输入同一组合法 `gert::TilingContext`，比较成功状态、完整 raw tiling 字节/长度、tiling key、blockDim、workspace 和 schedule。覆盖实际模型捕获的 B1/B8 参数；通过后才冻结结构布局及解码器。构建产物 SHA 不同可受工具链影响，不能仅凭版本号或文件名消除该门槛。

## 分支候选与实际选中路径

以下路径均相对固定提交。前四项是初始分支候选，不能替代之后从 selected launch key 得到的实际 SplitFuse 路径：

1. `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_register.cpp:25` 注册 `DoOpTilingFusedInferAttentionScore`；`fused_infer_attention_score_tiling.cpp:2252` 按架构分发。`:2198` 先 `RouteToFia → TilingFusedInferAttentionScoreV3`，否则进入 legacy IFA/PFA。
2. V3 `op_host/arch22/fused_infer_attention_score_tiling_v3.cpp:902` 解析和检查输入后交由 `FiaTilingRegistry`。`:1371` 的 `RouteToFia` 有 dtype、shape、特例约束，不能由 Python API 名字直接推断。
3. `attention/common/op_host/arch22/fia_tiling_nonquant.cpp:253–324` 中 TND/NTD 固定 `sInnerSize_=512`、`mBaseSize_=512`。`:411` 的 `Split` 会产生分核结果；`:737` 的完整流程依次生成 split、data、block、schedule、workspace、key。**不得先认定当前 TND 因 batch 改变 M，也不能用跨核 SplitKV 替代本故障。**
4. legacy PFA `op_host/prompt_flash_attention_tiling.cpp:3513` 的 `MatchTemplate` 为 TND 选择 S1/S2 与 nRatio；`:4861` 写 `PFAcoreParams`。非 TND 的 `AdjustCVTiling:6027` 选择候选 S1，并通过 `PromptFlashAttentionCheckArgsLegal` 检验 UB/L1/L0 等限制。真正修改应在被运行证据确认的 typed 策略选择处，让后续尾块、任务、workspace、key 一起重算；不能只改最终 `s1BaseSize`。

## 实际 selected key：SplitFuse 的 FAInferTilingData

主实验 `outputs/native_log_a1/native_synthetic_b8.log` 的原生 `NnopbaseExecutorDoTiling` 和 `LaunchKernel` 均报告 key **`5000000000000200103`**、blockDim 20、scheduleMode 1。该校准使用 FP16、TND、Q `[8192,16,128]`、K/V `[8192,8,128]`、每请求 1024 tokens、sparseMode 3、innerPrecise 0；它是原生算子校准，不能代替模型捕获。对应固定源码：

- `op_kernel/fused_infer_attention_score_tilingkey.h:251` 将该 key 定义为 `QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING`。
- `op_host/fused_infer_attention_score_tiling.cpp:38` 将该 key 注册为 **`FAInferTilingData`**；`:1618` 的 `TilingProcess4SplitFuse` 调用 `FAInferTiling::DoTiling`，随后统一保存 raw tiling、workspace、blockDim 和 key。
- `op_kernel/fused_infer_attention_score.cpp:249` 选择 `SplitFuse::FAInfer<ColumnMajor, RowMajor, half, half, float, false, false, MASK_CAUSAL, TND>`。保留 FP16 输入和现有 float 中间计算类型。
- `op_host/flash_attention_infer_tiling.h:216` 与 `op_kernel/fia_kernel_common.hpp:138` 的 `GetQSBlockTile` 都固定返回 `Q_TILE_CEIL=128`。不能将前述通用 V3 的 M=512 推到此实际 key，也不能从 batch 变化直接推出此路径 M 已变化。

[faI 全字段 CPU 检查](fainfer_schema_roundtrip.json) 对实际 schema 生成 **2488 字节**布局，34 个 scalar、468 个 array element 的官方 setter→序列化→metadata 推导偏移读取→重新序列化逐字节一致，另拒绝两种错误长度。无设备容器完成记录见 [fainfer_schema_check_completion.json](fainfer_schema_check_completion.json)。这证明固定源码加已安装 SDK 的序列化自洽，**不证明 stock host 实际输出就是此布局**，仍需实际 raw 大小和同输入 OFF/stock 比较。

`fai_schema_projection.h` 保留固定源码中的原始声明，`fai_schema_check.cpp` 使用 SDK `GetFieldInfo`（含显式 padding 项）推导范围，没有手猜字段偏移。SDK 标量 getter 读取 cached member，不能将 `TilingData(raw)` 后调用 getter 误当作解码 raw。通用 V3 的 [1576 字节检查](schema_roundtrip.json) 仅保留为另一 schema 的 CPU 校验，不用于此 selected key。

## 私有构建可行入口与 CPU 构建进度

[固定提交构建文档](https://gitcode.com/cann/ops-transformer/blob/c77c0829f1f0f0b8f882db1d9b051a758b605472/docs/zh/install/compile.md) 支持 `--ops` 和 `--vendor_name` 构建自定义包；`--install-path` 可把包放到任务私有目录，随后仅当前实验进程设置 `ASCEND_CUSTOM_OPP_PATH`。不需替换全局 OPP。

已执行命令为 `bash build.sh --pkg --jit --noaicpu --soc=ascend910b --vendor_name=fault28_off --ops=fused_infer_attention_score -j4`。`--jit` 表示包内不预编译 kernel，**不等于运行时必然复用已安装 kernel 二进制**，需验证实际加载和编译轨迹。完整编译可以去掉 `--jit`；只构建 host 可研究 `--ophost`，但单个 `.so` 尚不足以证明 vendor 的 ACLNN/注册/配置与 kernel 查找闭环。

`attention/fused_infer_attention_score/op_host/CMakeLists.txt:16` 明确依赖 `incre_flash_attention`、`prompt_flash_attention`、`attention/common`。自定义构建会生成 `cust_opapi`、`cust_opsproto_rt2.0` 和 `cust_opmaster_rt2.0` 等组件，见 `cmake/custom_build.cmake`，不能直接把安装的 `libophost_transformer.so` 名称套到 vendor 产物。

固定构建依赖包括 opbase `cd48c600c4b37146f504217eab2bf3956f7fee8b`、ops-tensor `b24535e30a6f7a3909c53c8b09713a9c0d08e54b`、cann-cmake `master-026`（下载归档 SHA `b0db2d4d0d2e94bd0dd961d88dc16b6b042fbacb4de4acb9530128c98e637fca`，以最终构建清单为准）及第三方库。构建使用同一固定镜像、4 CPUs、16 GiB RAM、无 NPU devices、非 privileged、每轮 1800 秒上限；仅在任务目录安装 Python dependencies，宿主 GNU patch 只读挂入，不安装系统包。第 4 轮显式固定 Python 3.11 并继续官方 CMake package target，748.97 秒后 rc=0、完整 package 成功；自有容器已移除。此前工具依赖和解释器选择失败保留日志。私有安装第一次未 source CANN 环境，官方共享库验证因 libascendalog 查找失败拒绝；第 2 次 source 9.1 环境后 rc=0，未使用 --force，容器也已移除。`build.sh --tiling_key` 是构建 kernel 变体筛选参数，不是已证实的运行时 tiling override。

原生 OFF 构建的目标是先证明与安装版本透明等价，再从同一个构建配置生成 ON。只有 host 策略与 kernel 约束均通过后，才能称为方案 A 的原生 tiling 注入；当前是源码匹配与构建准备结论。

## M64/M128 协同策略的待验证范围

当前选中变体不是一个可任意覆写 M 的运行时模板。host 的 query task 数、coreInfo 分配和 workspace，以及 kernel 的 task 索引、causal 边界、L1/L0 shape 和 Q buffer 都依赖 128。`flash_attention_interface.cpp:73–91` 包括 QK L1 `<Q_TILE_CEIL,128,128>`、QK L0 `<128,128,128>`、PV L1 `<128,128,256>` 和 PV L0 `<128,128,128>`；`flash_attention_regular.h:503–553` 以相同 query tile 解释任务及 mask。仅将 host 改为 64 会使 stock kernel 错读任务语义，不能当作合法 tiling 变化。

方案 A 若继续，应在同一最终私有二进制中保留原 128 OFF 变体，新增独立 key 的 64 变体，让 host 使用同一 tile 参数重新生成 task/core/workspace，并让该 key 选择匹配 kernel 模板。还需审查 vector/softmax/update 组件、L1/L0 容量和尾块处理后通过编译与数值验收。该变体**尚未实现、尚未证明合法，也未运行注入**。不得为制造差异而改 dtype、插入 cast 或换成 SplitKV。当前 batch 是否触发预期现象，以真实模型 profile 和后续合法策略实验为准。

## 仅跳过 SplitFuse 改走旧 PFA：当前 FP16 不能直接采用

已审查完整 host route：`fused_infer_attention_score_tiling.cpp:2235–2247` 可经 PFA 生成 TND 的 M=512、S2base=128、nRatio=4（受长度裁剪）及完整 key/workspace。可是固定 release 的 `prompt_flash_attention_arch22.h:520–584` 将这些 TND key 和 `VarLenScoreSameABBaseApi` 实例放在 **DT_BF16 输入/输出条件块**，明确实例化 `bfloat16_t,float`；FP16 条件块 `:205–518` 没有对应 TND 实例。宏名中的 QFP16 不代表当前条件编译实际支持 FP16。FIA 的 fallback kernel 直接调用这个 entry（`:1009`）。因此只改变 host route 不能证明当前 FP16 具有合法 kernel，不实施该候选。

改为 BF16 会破坏固定 dtype 的实验条件；新增 `half,float` TND 实例也属于新的 kernel 变体，必须另行编译、类型与数学路径审计及数值验收。详细源指纹与边界见 [route_candidate_review.json](route_candidate_review.json)。

## 原版 OFF 交付路径（CPU 准备完成）

- 完整 `.run` 包：远端 `source_matching/ops-transformer-v9.1.0/build_out/cann-ops-transformer-fault28_off_linux-aarch64.run`，5,818,098 bytes，SHA256 `b70e237da3e2bfffb7a0ccd9b5486006d93d486dfd34aaff9316136048210635`。
- 私有 vendor：远端 `/home/fault28-native-tiling-20261029/build_off/vendor_stage_a2/vendors/fault28_off_transformer`。名称由官方构建附加 `_transformer`。host SHA256 `39de5e4571c73e3441a045306eccc98fdef7afbfc0462ac2fe316caa19d664fa`；opapi SHA256 `89f4a4fc6edd595b8824fbbb56e6157c222935f18203c8447671abda1a95f969`。
- 运行进程须将 `ASCEND_CUSTOM_OPP_PATH` 设为其容器内可见的 vendor 完整路径，并在 `LD_LIBRARY_PATH` 前置 `$vendor/op_api/lib`。安装器生成的 `set_env.bash` 写死安装时 `/build/...` 路径；如果后续挂载到 `/audit`，不得直接 source 该脚本。
- 构建和私有安装均无 NPU devices、非 privileged；私有安装还禁用网络。没有系统 SDK 安装或替换。依赖 opbase 出现的 34 个 aicpu_common 删除由固定源码 `cmake/third_party/opbase.cmake:41–51` 的官方 FetchContent 后处理产生，算子仓库 tracked diff 为空。

完整 842 个 vendor 文件指纹、依赖和构建命令见 [off_build_manifest.json](cpu_build_evidence/off_build_manifest.json)，成功记录见 [build completion](cpu_build_evidence/build_off_attempt4_completion.json) 和 [private stage](cpu_build_evidence/private_stage_a2.json)。**完成范围为 CPU package/stage；NPU kernel 执行、stock/OFF 输出等价、完整 tiling ABI 和故障 ON 仍未验证。**
