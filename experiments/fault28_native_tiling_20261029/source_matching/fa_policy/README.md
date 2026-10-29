# FA 私有 typed tiling 策略：CPU 构建与独立解码证据

本目录记录真实 HF SDPA → `aclnnFlashAttentionScore` 路线的私有构建。它与先前 vLLM 的 FIA SplitFuse 路线不同。a2 已完成私有 OFF 与 stock 兼容验收及 ON/不匹配/恢复诊断；本目录独立核验了诊断 9 条、默认统计 B/F 4 条和高精度统计 B/F 4 条实际 host 记录。默认统计的首差验收未通过，原因是统计精度未显露 SDPA 输出位差。随后使用明确披露的任务内 high 采集器修复重采：独立 msprobe 分析在两个 step 都将首个可见统计差定位到 layer 1 SDPA 输出。默认 low 的原始失败证据保留；修改后的采集器不能称为未经修改的官方发行包。

源码固定为官方 ops-transformer `v9.1.0`，commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`。FA 的 13 个安装 kernel 文件与 release 文件哈希一致，详见上级 `fa_kernel_source_match.json`；这不单独证明安装 host 库与重建 host 的 ABI、策略完全一致。

## 修改及恢复

`apply_policy.py` 只接受固定 commit 的干净副本，并按唯一源码锚点修改两个 host 文件，另增加 `fault28_fa_policy.h`。当前 a2 源码清单为 `build_evidence_a2/policy_source_manifest.json`，SHA256 `e760be8f4b4dd42b4f44898a8d91c68585c9f96f42c47eecb038eb8d37859294`。a1 的旧源码与指纹保留在 `archive_a1/`、`build_evidence/`，不用于 ON。kernel 源码未改。

唯一策略变化发生在原 `FlashAttentionScoreTilingS1s2Bn2gs1` 的 `CalcS1S2BasicBlock`、`CalcNRatio`：匹配时 ON 选择 M64、nRatio4；OFF 保留原 M/nRatio 选择。S2base 仍由原逻辑选择。后续 matmul、softmax、分核、尾块和 workspace 全部由原 typed host 逻辑生成，没有改写已经生成的 raw 字节。a2 实际 profile 确认了这些策略值；数值差异和 msprobe 定位仍按各自独立证据判断。

进程环境契约：`FAULT28_FA_MODE=off|on`，`FAULT28_FA_SCOPE_PID`、`FAULT28_FA_SCOPE_TID` 分别匹配实际 PID 与 native TID，`FAULT28_FA_SCOPE_LAYER=1`，`FAULT28_FA_SCOPE_PHASE=prefill`，非负十进制 `FAULT28_FA_SCOPE_FORWARD`，唯一 ASCII token `FAULT28_FA_SCOPE_TOKEN`，绝对路径 `FAULT28_FA_LOG_PATH`。模型侧仅在原 layer 1 SDPA 同步调用前设置作用域，并在 `finally` 中恢复。

host 同时核验 FP16 Q `[8,16,2048,128]`、K/V `[8,8,2048,128]`，BOOL mask `[2048,2048]`，BNSD、16 heads、inner precision 0、sparse 2、keep probability 1、pre/next tokens、scale，以及相关可选输入缺省。不使用调用次数选择故障。每次符合条件的 host 回调持续命中；若原生缓存绕过回调，不能凭期望次数声称命中。缺省、OFF 或不匹配作用域保留原生选择。

同一二进制同时提供 OFF/ON。包安装在任务私有 vendor 目录，不替换全局 SDK。最终 `vendor_manifest.json` 为 670 项文件清单，a2 SHA256 `63a5171d2031aa1a8ae5b37c8c9b488614779af7759841e224079a7e746d7b07`；其 `vendor_root` 相对 `/audit` 为 `build_off/fa_vendor_stage_a2/vendors/fault28_fa_transformer`。不要直接 source 包内写死 `/build/...` 的安装环境脚本；启动器必须先验证 manifest 与全部文件，再设置当前进程的 vendor 环境。

## CPU 构建结果

固定镜像 `8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9`，无设备、无 privileged，4 CPU，最多 16 GiB。a2 使用新的 checkout 和空 build 目录，216 秒完成干净构建及官方打包，返回 0；构建前后源码指纹相同。新 stage 私有安装返回 0，相关 CPU 容器全部已删除。a1 的 clean build 2、package attempt 3 记录保留作探索证据。

中间探索记录保留：首轮构建未包含最终补充字段；首次 CPack 只生成 staging/JSON，随后补上官方 `ENABLE_BUILD_PKG=ON`；两次打包配置因容器内依赖路径未挂载而失败。最终将固定依赖挂载回原编译绝对路径，完成官方 makeself 打包。这些失败不计作通过。

当前 a2 host 库 SHA256 为 `310a0d4daa2e6a27a20ebca24533af6cfe0339d45d997d3dfd3fa031fd5e3ea8`，op-api 库为 `e928b686d50ef8934cb980b3a842c7a8bb543db910ee88587ba530c0e92bcac0`，安装包为 `6524468fa534f47cb19b7cfd1e4135519f68483bf72521dd4dae40d327b15b5f`。构建的是 JIT source 私有包；实机的 kernel 装载与执行证据单独记录，不从 CPU 构建成功推断。

## 实际 host profile

每行 JSON 的 schema 为 `fault28.fa.typed_host.v1`。顶层包含 PID/TID/token/forward/layer/phase/mode、eligible、policy_applied、callback_status、tiling_key、block_dim、schedule_mode、workspace_count、workspace_sizes、typed、raw_size、raw_hex。失败回调仅记录状态，输出字段为 null，不读取失败输出；成功时 raw 长度同时受到 1 MiB 和公开 `GetCapacity()` 限制。

`typed` 包含 template、M (`m`)、s2_base、n_ratio、s2_group、s1/s2 outer/tail、core_num、total_query_tasks、split_factor、split_factor_tail、sparse_start_idx[48]、enable_l1_reuse、sparse_type、s1/s2_sparse_valid_size、B/N1/N2/G/S1/S2/D/Dv、dtype_enum、layout、calc_bytes、high_precision、impl_mode、sparse_mode、mask/dropout/PSE 标志、keep_prob、scale。

`enable_l1_reuse` 在 S1S2 派生类原 `SetEnableL1Reuse()` 调用之后读取；该类与基类存在同名成员，通用基类成员不能代表实际选择。workspace 使用原生 `PostTiling` 已取得并最终更新的指针，只复制值；没有增加可能改变数量的 `GetWorkspaceSizes()` 调用。`GetScheduleMode()` 为 SDK 公共只读接口，单独的 0 值可能是空指针 fallback，不能单据此认定实际 dispatch。

## 独立 POD 解码与 key 验证

`FlashAttentionScoreGeneralTilingData` 是官方标准布局、可平凡复制的 POD，并非 `GetFieldInfo()` 宏类。`fa_pod_decoder.cpp` 使用固定头文件与实际 SDK 的 `sizeof` / `offsetof`，编译期断言布局性质，通过有界 `memcpy` 解码。实际 sizeof 为 **1464 字节**、对齐 8。105 个公共 primitive 字段和 65 个数组元素的 sentinel 往返通过；SDK 嵌套 matmul/softmax/transpose 区块记录整体 offset/size，不凭猜测展开。CLI 要求输入长度恰好 1464，否则失败。

```sh
fa_pod_decoder --schema
fa_pod_decoder --self-test
fa_pod_decoder call_001.declared_prefix.bin
```

已只读解码 stock 四组共 112 份日志 raw 候选：全部长度 1464；B1、B8 均为 M128、S2base128、nRatio8、BNSD、raw sparseType CAUSAL(3)、implMode0、coreNum40。B1 totalSize256/splitFactor14，B8 totalSize2048/splitFactor104。同条件重复一致。记录为 `build_evidence/summary.json`。随后 112 份 raw 已分别与实际 a1 OFF 同 B 的全部公共 context raw 逐字节一致，确认这些已观察形状的日志 framing 与公共 POD ABI（`build_evidence/off_actual_review/raw_framing_match.json`）。这不解决 a1 的 selector 缺陷，也不代替 a2 的新 OFF 验收。

`fa_key_check.cpp` 使用官方 SDK `GET_TPL_TILING_KEY`，精确得到实测 key **22619644976**，对应 `FlashAttentionScoreS1s2Bn2gs1<..., half, float, ..., ND, L1Reuse=true>`。key 的 Sparse 参数为 ANY(2)，实际 causal 类型由 raw inputParams.sparseType=3 决定。模板参数采用 SDK 编码，不能直接把某些位段的编码索引当成声明值，例如 NONE(9) 编码后低位为 6。

## 从实际 profile 推导 query/key 分组

```sh
python3 derive_query_trace.py --profile host.jsonl --record-index 0 \
  --source-root /audit/source_matching/ops-transformer-v9.1.0 \
  --target-batch 0 --target-query 2047 --output query_trace.json
```

脚本先核验五份固定 kernel / key 源码 SHA，然后读取实际 typed profile。它不使用 mode、eligible、policy_applied 或预设命中数，支持本次 BNSD causal S1S2 路线，其他情况拒绝。根据 `sparseStartIdx` 重建每个 AIV 区间、L1Reuse 相对奇偶分工、query block 轴索引和实际 key 分组，检查所有真实 query 恰好一个所属任务、每个 query 的合法 causal keys 覆盖一次。

L1Reuse 两个 AIV 共享同一 query-task 区间，但按相对奇偶交替处理不同 query blocks。尾部 fake pair 的 `lastNotPair` 在原 `ProcessVec1` 1253–1256 和 `ProcessVec2` 2164–2167 提前返回，不写 output；脚本单列这些空消息，不计作重复 query。S2 循环在同一个 query task 内完成，并进行 online softmax/output 累积，不能把多次 key-group 循环当成跨核 SplitKV。

四个合成 profile（M64/128 × L1Reuse 开/关）每个覆盖 262144 个 queries，20 个破坏约束的负例被拒绝。这只验证 CPU 推导器逻辑。实际 profile 的推导结果是**根据固定源码重建的执行计划**，不是硬件指令 trace，也不是数值等价证明。


补充：a1 OFF 的实际 raw/输出可兼容，但 B8 eligible=false，不能进入 ON。已定位 OptionalAbsent 未按原生 FA 的 rank-0 optional 缺省规则处理；a1 作为探索证据保留。修复已使用新的 a2 包、manifest，并重新通过 OFF 验收，没有覆盖旧证据。


当前 a2 额外记录 `eligibility_checks` 与每次 `guard_reports`：逐项失败原因、公开 IR 0–18 的 storage/origin shape、dtype，以及实际 attrs。IR 12/13/14 为 d_scale_q/k/v，18 为 p_scale；和其余可选 tensor 一起按固定 FP16 路线要求缺省。此处没有改动 dtype、mask、inner precision、输入张量或算子输出。

可提交的复现材料在 [BUILD.md](BUILD.md)、`build_requirements.txt`、`write_vendor_manifest.py`。这些文件给出实际 CMake/容器/安装参数、固定依赖和源码/SDK 指纹、逐文件清单生成方法；不含 SDK、模型权重或构建 binary。

## a2 实际 B/F 的独立核验与统计限制

`build_evidence_a2/bf_actual_policy_review/summary.json` 逐条核对正式 B/F 的两个 forward：26 个 typed 字段与公开 POD 解码一致，全部严格 guard 通过，五份 kernel 源码指纹符合冻结版本。四条记录的 raw/typed/guard/key/block/schedule/workspace 与相应诊断记录完全相同。CPU 核验没有 NPU 设备、没有网络，临时容器已删除；原始采集文件未修改。

| 实际策略 | M / S2base / nRatio | 实际 query tasks | 目标末尾 query 2047、head 0 | 合法 key 分组 |
| --- | --- | --- | --- | --- |
| B / OFF | 128 / 128 / 8 | 2048 | AIV 1、task 15 | `[0,1024)`、`[1024,2048)` |
| F / ON | 64 / 128 / 4 | 4096 | AIV 1、task 31 | `[0,512)`、`[512,1024)`、`[1024,1536)`、`[1536,2048)` |

每条 B8 profile 检查 262144 个 query：每个真实 query 只有一个所属任务，其合法 causal keys 覆盖恰好一次，同一 query 没有跨核 SplitKV。B 的 8 个、F 的 30 个 L1Reuse fake pairs 单列排除，不计作额外 output 写入。这是源于实际 profile 和固定源码的执行计划推导，不是硬件指令 trace。

正式默认精度采集中，SDPA.1 输出的 Max/Min/Mean/Norm 在 B/F 相同，不能据此判断张量无差异。已安装 msprobe 26.1.0 的 `get_stat_info` 仅在 `precision != high` 时给非标量浮点张量赋 Mean/Norm；当前 high 分支没有转换 FP32，也没有补算这两项。该文件哈希与安装包 RECORD 相符。证据为 `build_evidence_a2/msprobe_precision_source_audit.json`；仅修改 `precision=high` 不能当作已经解决精度问题，尚未据此改动采集器。

`build_evidence_a2/checkpoint/` 保存 ON 前与正式 B/F 后四个选定文件的哈希清单及比较。逐路径字节数与 SHA256 完全相同，共 1503311025 字节；只读计算文件哈希，没有加载或复制模型张量。

## 高精度采集器重采验收

任务内副本只修改 `PytorchDataProcessor.get_stat_info`：high 时将 FP16/BF16 的 detached 临时数据转为 FP32 计算统计，保留原张量、dtype/shape 元数据及官方 API 归属。补丁不读取 B/F、layer、target 或命中状态。源码、安装包 RECORD、513 项副本文件、实际 import 路径及 processor SHA 均有证据；原安装包没有变化。详细材料见 `../../analysis/collector_high/`。

CPU 的 23 项检查验证了原对象/指针/bytes/device/shape/stride/storage offset/version 不变、low 与原函数一致、DataInfo 与真实 DataWriter 元数据保持；11 项有限值还与 represented-value FP64 参考比较。无注入高精度 stock pilot 的两个 forward，对旧 stock 及 low 各比较 28 层，共 112 次目标调用：Q/K/V、attention output 与 logits 哈希一致。每 step 的 1621 个节点、425 个 module、28 个 SDPA，以及 API 顺序、非统计元数据、construct 和模型 stack 都与 low 相同，未发现采集器内部 cast 伪节点。详见 `build_evidence_a2/collector_high_independent_final_review.json`。

正式 `bf_statistics_high_a1` 的四条 host 记录再次通过 public POD、严格 guard 与源码推导检查，并与相应诊断策略精确相同；采集器和四个选定模型文件的哈希保持不变。汇总为 `build_evidence_a2/high_bf_independent_review_summary.json`。另一个只读原生 msprobe 分析器从 dump/stack/construct 得到两个 step 的首差均为 `Functional.scaled_dot_product_attention.1.forward` 的输出 Mean；该统计结论不是由本目录的 policy/hit 字段生成。范围仍是本次固定真实 HF FA 路线的机制验证，不是历史事故或硬件缺陷复现。
