# vLLM Ascend Operator Fault Injection

这是一个面向 vLLM-Ascend/CANN 的开发者模式原生算子故障注入原型。实现覆盖 CANN repeat executor-cache replay 边界、typed operator adapter 和 temporary device buffer 替换，并提供受控、可审计的输入数值退化、非有限输出代理与 allocation-boundary 写错地址代理。

本仓库包含运行源码、构建/容器脚本和能力说明；后续实验另提交经过审查的统计证据与验算工具。
完整私有张量、模型权重和 SDK/kernel 二进制不随统计证据上传。

## 算子执行策略注入（第一期）

新增 [`native_execution_strategy`](native_execution_strategy/README.md)：统一调用上下文、
逐次触发、合法候选选择、版本/构建证据解析和原生执行验证。首个适配器是
`FlashAttentionScore/S1S2`，实际在 host 的 M/nRatio 选择完成后、依赖 tiling 生成前原子切换，
由真实 kernel 产生输出。用户配置选择 `query_tile`、`key_group` 或 `joint`，不写 raw 字段。

原生 Batch 策略发现与受控策略切换分别取证；每次匹配重新选择，没有固定 B8 条件或单次命中上限。
适配器解释请求数与 token/矩阵语义；未验收的形状不会因发现成功自动进入受控支持范围。
策略切换、数值变化、模型影响分别验收，允许合法切换后数值完全相同。

124上第一期20组/56次正式调用及4次 `validated` 别名检查通过；B1/B8的连续触发、关闭后完整输出恢复，
以及独立 msprobe 的 FA 主输出首差均已核验。原生 B1/2/4/8 的 M/nRatio未切换，模型级影响未测。
本轮状态、真实版本和完整矩阵见
[124 第一阶段验证](experiments/native_execution_strategy_20261030/README.md)。
原始 `statistics/mix` 与审计记录见[本轮证据](artifacts/native_execution_strategy/fa_phase1_statistics_20261030/README.md)。
MatMul、Reduce/Softmax、FIA 和自然模型接入仍待独立适配验收。

## 已验证能力

| 项目 | 状态 |
|---|---|
| `aclnnQuantMatmulWeightNz`，INT8 x1=`addresses[0]` | `PROVEN_E2E` |
| `aclnnMatmulWeightNz`，BF16 self=`addresses[0]` | `PROVEN_E2E` |
| INT2-equivalent / INT4-equivalent primitive | `PROVEN_E2E` |
| BF16 mantissa truncate | `PROVEN_E2E` |
| single/multi-hit、dual-rank、all-hit selector | `PROVEN_E2E` |
| QMM x1 extreme-value single/progressive/all-hit | `PROVEN_E2E`（CANN 8.5.1，最大 12,160 hits） |
| QMM x2/weight runtime slot 与 NZ storage metadata | `PROVEN_E2E_READ_ONLY`（CANN 8.5.1） |
| QMM x1+x2 双 temporary buffer（lossless / 127 extreme） | `PROVEN_E2E`（CANN 8.5.1，rank0 single-hit） |
| QMM x2_scale formal tensor、runtime slot 与 FP32 metadata | `PROVEN_E2E_READ_ONLY`（formal ordinal 3，`addresses[2]`） |
| QMM x2_scale temporary buffer（lossless / finite ×2） | `PROVEN_E2E`（CANN 8.5.1，rank0 decode single-hit） |
| `aclnnMatmul` BF16/ND slot mapping 与 self/mat2/dual lossless temporary buffer | `PROVEN_E2E`（CANN 8.5.1，rank0 decode single-hit） |
| `aclnnMatmul` 有限 BF16 双输入到真实 `+Inf` 输出 | `MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED`（`2^58` 首次 non-finite） |
| `aclnnMatmul` 有限 BF16 正负抵消、数学结果为零的排列对照 | `MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED`（half non-finite；block/alternating 全零） |
| `aclnnMatmul` post-Run output `+Inf/-Inf/qNaN` proxy | `PROVEN_E2E`（1/5/51/512 elements，readback exact） |
| Fault26 QMM output-base underflow → live residual stomp proxy | `PROVEN_E2E`（rank0，allocation-boundary，231 all-eligible hits） |
| Fault26 fault request 输出分叉与 disarm 后精确恢复 | `PROVEN_E2E`（token 7 首次分叉，post-fault exact recovery） |
| Fault20 v17 metadata ABI handshake + immutable bundle | `PROVEN_E2E`（CANN 8.5.1，TP=2，C1/C2/C3/recovery） |

结论严格限定于已验证环境和算子，不代表任意模型、任意算子或跨版本生产能力。详细统计见[当前注入能力统计](fault20_fast_validation/reports/fault20_injection_capability_inventory_20260807.md)。

## Fault39 current capability

| 路线 | 已验证内容 | 结论 |
|---|---|---|
| QMM extreme | x1 single/progressive/all-hit；x1+x2 dual extreme | 输入替换与模型传播已闭环，真实 QMM output 全 finite |
| QMM x2_scale | formal ordinal 3、Run slot 2、lossless、有限 `×2` | scale 扰动传播已闭环，真实 QMM output 全 finite |
| Floating MatMul finite dual | self/mat2 temporary 均为有限 BF16 | `MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED` |
| Floating MatMul cancellation | 理论结果为 0；half 与 block/alternating 排列对照 | `MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED` |
| MatMul output proxy | post-Run `+Inf/-Inf/qNaN`，1/5/51/512 elements | 三种 `MATMUL_OUTPUT_*_PROXY_PROVEN` |

当前证据层级：

```text
Root-cause approximation
  finite dual BF16 -> real aclnnMatmul natural +Inf
  cancellation     -> theoretical zero, layout-dependent natural +/-Inf

Outcome proxy
  real finite output -> direct +Inf / -Inf / qNaN -> downstream anomaly
```

最强 A2 判定是 `MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED`。它证明真实
`aclnnMatmul` 对 K 轴正负排列表现出不同的 non-finite 行为，但没有定位 kernel 内
accumulator 的类型、位置或首次 overflow 时刻，因此不能写成
`ACCUMULATOR_OVERFLOW_PROVEN`。完整证据见
[Fault39 A2 复现总结](fault20_fast_validation/reports/fault39_matmul_fault_reproduction_summary.md)。

## Fault26 current capability

Fault26 的当前最强能力不是“直接改坏 residual”，而是在已由运行时 allocator
只读证明的相邻 allocation pair 上，把真实 QMM output base 向后偏移，使真实
`aclnnQuantMatmulWeightNz` 写入跨过 allocation boundary 并覆盖 live residual；
随后恢复 padding 与正常 output，让精确的下一次 `aclnnAddRmsNormBias` 消费受控破坏，
并在 consumer 完成后恢复 residual。

```text
live residual allocation end == QMM output allocation begin
  -> real QMM Run with bounded backward-shifted output base
  -> allocation boundary crossed
  -> next exact AddRms consumes corrupted residual
  -> padding/output/residual byte-exact recovery
```

Phase 30 在 rank0 的三个精确、live-proven 地址对上完成一次请求内 231 次
all-eligible hit（`7 + 7 + 217`），失败/中止为 0；模型响应在 completion token 7
首次分叉，显式 disarm 后下一请求精确回到 baseline。`all eligible` 只表示命中控制面
列出的、已证明相邻且仍存活的地址对，不表示对未审计 QMM 地址做无条件写入。

该能力应准确命名为：

```text
QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN
ALL_ELIGIBLE_QMM_HITS_PROVEN
MODEL_RESPONSE_DIVERGENCE_PROVEN
POST_FAULT_EXACT_RECOVERY_PROVEN
```

它是本地开发者模式下用于诊断工具验证的受控算子故障代理，不证明健康 QMM kernel
会自然发生 OOB，也不证明历史 FIA workspace 缺陷的物理根因。

## 仓库内容

```text
fault20_fast_validation/
├── evidence/                            # 完整 Fault26 验证证据压缩包与校验说明
├── source/
│   ├── l4_buffer_substitute_allhit_v1/   # 最终 runtime、控制面和容器脚本
│   ├── l4_metadata_lifecycle_proxy/      # typed GWS metadata adapter
│   ├── fault26_readonly_trace_v1/        # residual/QMM trace、underflow runtime 与 worker 控制面
│   └── fault26_fia_writer_proxy_v1/      # FIA actual-writer / wrong-destination 分阶段代理
└── reports/
    ├── fault20_final_architecture.md
    ├── fault20_capability_matrix_final.md
    ├── fault20_fault_primitive_design.md
    └── fault20_injection_capability_inventory_20260807.md
```

## 已验证环境

```text
model       Qwen3-32B-w8a8-nopdmix (dense)
vLLM image  quay.io/ascend/vllm-ascend:v0.12.0rc1
vLLM pkg    0.12.0
torch_npu   2.8.0
CANN target 8.3.RC2
arch        aarch64
C++ ABI     _GLIBCXX_USE_CXX11_ABI=0
execution   TP=2, eager, repeat executor cache
```

CANN 8.5.1 / vLLM 0.18.0 / torch_npu 2.9.0.post1 已完成独立编译、
`libnnopbase.so` ABI smoke 和 Qwen3-32B TP=2 模型级 E2E。3/16/64/all-hit extreme
均已验证；all-hit 的 12,160 次真实 QMM output 全部 finite。随后已证明 x2/weight 为
`addresses[1]`，并完成 x1+x2 无损门禁及单次双极值 E2E。双极值将真实 QMM BF16 output
放大到 `max_abs=872`，但仍全部 finite，因此不构成 accumulator overflow 证据。
当前还证明了真实 QMM 的 `x2_scale` 为 formal ordinal 3、compact Run slot
`addresses[2]`、连续 FP32 `[5120]` storage。无损 temporary substitution 与模型响应/token
精确等价；有限 `×2` 令 5,120 个 scale 全部改变，并使生成 token 从 0-based index 4 开始
分叉，但真实 QMM BF16 output 仍全部 finite。该结论命名为 scale amplification，不命名为
accumulator overflow。

Fault20 v17 的最终 msprobe 交接版本增加了显式 metadata ABI descriptor 和 fail-closed
握手，并将 preload、control、proxy、真实 `libopapi`、manifest、校验器与参数化 rootfs
launcher 固定为只读 bundle。远端 bundle 是
`/home/haoran/fault20_fast_validation/runtime/cann851_semantic_slot_bundle_v17_hs2`，模型级
证据是 `/home/haoran/f20runs/v17h2`。交接前必须运行 bundle 内
`harness/verify_v17_bundle.sh` 并得到 `FAULT20_V17_BUNDLE_VERIFY_PASS`；不得把其中任一
`.so` 与旧版本混用。详细 hash、设备映射、C1/C2/C3 和 Dump-ON 交接步骤见
[CANN 8.5.1 使用说明](FAULT20_CANN851_USAGE.md)。

## 构建

必须在与推理服务相同的 vLLM-Ascend/CANN 镜像内构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export FAULT20_BUFFER_OUTPUT_DIR=/fault20/runtime/l4_allhit_v1
bash /fault20/source/l4_buffer_substitute_allhit_v1/build.sh

export FAULT20_METADATA_PROXY_OUTPUT_DIR=/fault20/runtime/l4_metadata_lifecycle_proxy
bash /fault20/source/l4_metadata_lifecycle_proxy/build.sh
```

CANN 8.5.1 独立构建必须显式锁定 toolkit 与真实库路径片段，并为每次构建使用新的
`OUTPUT_DIR`。构建只写入该目录内同名二进制，不会清理其他 runtime、logs 或 evidence：

```bash
export FAULT20_BUFFER_OUTPUT_DIR=/fault20/runtime/cann851_validation_v1
export FAULT20_ASCEND_TOOLKIT=/usr/local/Ascend/cann-8.5.1/aarch64-linux
export FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT=/cann-8.5.1/
bash /fault20/source/l4_buffer_substitute_allhit_v1/build.sh

export FAULT20_METADATA_PROXY_OUTPUT_DIR=/fault20/runtime/cann851_metadata_proxy_v17
export FAULT20_REAL_LIBOPAPI=/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so
bash /fault20/source/l4_metadata_lifecycle_proxy/build.sh
```

runtime 构建脚本会检查 ABI、SONAME、目标导出符号、意外 CANN `DT_NEEDED` 依赖和 `LD_BIND_NOW` 加载。离线穷举测试及其结果属于本地验证资产，不在公开仓库中。

主要产物：

```text
/fault20/runtime/l4_allhit_v1/libfault20_buffer_substitute_abi0.so
/fault20/runtime/l4_allhit_v1/fault20_buffer_ctl
/fault20/runtime/l4_metadata_lifecycle_proxy/libopapi.so
```

## 容器参考

```bash
bash /fault20/source/l4_buffer_substitute_allhit_v1/create_allhit_container_v2.sh
```

该脚本包含实验机专属的模型路径、NPU 设备 2/5、端口 18005 和挂载路径，只应作为已验证配置参考；迁移前必须调整路径、设备和版本门禁。

关键加载顺序：

```bash
export LD_LIBRARY_PATH=/fault20/runtime/l4_metadata_lifecycle_proxy:$LD_LIBRARY_PATH
export LD_PRELOAD=/fault20/runtime/l4_allhit_v1/libfault20_buffer_substitute_abi0.so
```

control v17：

```text
magic   0x4632304255463137
version 17
SHM     /fault20_l4_buffer_substitute_ctl_v17
```

## 命中范围

v17 保留历史 `arm-int4` / `arm-int2` 的 arm-lifetime bounded 语义：`MAX_HITS`
是一次 arm 生命周期内的累计预算，候选窗口也跨请求累计。该行为用于可复现实验，未改变。

v17 继续支持 `arm-int4-observe` / `arm-int2-observe` 组合模式：先执行 QMM x1
数值退化，再在同一次 `OpExecCache::Run` 后同步并扫描真实 QMM output，输出
`qmm_output_observation` 事件。这样 Fault20 的 INT4/INT2 路径可以获得与 Fault39
cancellation 类似的 “fault + post-Run observe” 证据链。

在 CANN 8.5.1 的 `aclnnQuantMatmulWeightNz` 路径上，`OpExecCache::Run`
的 `addresses[]` 可能只包含输入/scale/bias 等 storage，不包含 output storage。v17 继续把
GWS 阶段捕获到的 formal output metadata 复制到 legacy metadata 头部；post-Run observe
优先使用已证明的 `addresses[out]`，否则使用 GWS formal `out.data_addr` 做只读 D2H
扫描。若两者都不可用则 fail-closed。

v17 新增 QMM x1 的 proof policy，但不改变历史命令语义：

```text
arm-int4
  -> proof_mode=STRICT_PRODUCER
  -> 仍要求前驱 AscendQuantV3 output 与 QMM Run slot 0 地址一致

arm-int4-slotproof
  -> proof_mode=SEMANTIC_SLOT
  -> 不要求前驱 producer 地址守卫
  -> 必须证明 formal tensor ordinal=0、compact Run slot=0、参数顺序/地址数/空槽规则精确、无共享 storage、x1 storage 唯一
```

两种策略都使用同一个 `ResolveSemanticTarget(QMM_X1)` 和同一个 INT4 transform/substitution 核心。候选计数只在目标 shape/dtype/layout/bytes 与所选 proof policy 全部通过后递增。GWS metadata 中的 `data_addr/storage_addr` 与 Run address 的直接匹配只写入日志作诊断，不能单独作为注入依据。未知 proof mode、槽位歧义、共享 storage、参数顺序或地址数量不一致均 fail-closed。

新增的 request-scoped 模式是独立功能，适用于本地开发者模式中的重复在线实验：

```text
arm-int[24]-request
  -> 每请求最多 MAX_HITS_PER_REQUEST 次
  -> 所有请求累计最多 MAX_HITS_TOTAL 次；到达后自动 disarm
request-begin REQUEST_ID
  -> 用单调 request id 开启下一请求并清空请求内 candidate/hit 计数
disarm
  -> 立即停止后续请求的注入
```

`request-begin` 必须在同一 Docker IPC namespace 中、且在目标 HTTP 请求进入模型前调用。
同一个 request id 可幂等重复提交；较小 id 被拒绝。TP worker 不得各自递增 id：控制端应为
同一 HTTP 请求写入同一个单调 id。native `OpExecCache::Run` 本身无法推断 HTTP 请求边界，
因此自动化接入需由 worker/controller 在请求边界调用该命令或等价控制 RPC。

## 控制命令

默认必须保持 disabled。所有 `arm-*` 命令及实际 runtime substitution 都要求进程环境中
`VLLM_SERVER_DEV_MODE=1`；否则 fail closed。示例中的 device mask `3` 表示已验证的两个逻辑设备 0/1。

```bash
fault20_buffer_ctl status
fault20_buffer_ctl reset
fault20_buffer_ctl disarm

# 历史兼容：STRICT_PRODUCER，参数语义不变
fault20_buffer_ctl arm-int4 0 1 5120 1 1 5120

# 新模式：SEMANTIC_SLOT；只在完整 formal-slot proof 通过后命中
fault20_buffer_ctl arm-int4-slotproof 0 1 5120 1 1 5120

# QMM all-hit
fault20_buffer_ctl arm-all-qmm-lossless 3 1 5120 5120
fault20_buffer_ctl arm-all-qmm-int2     3 1 5120 5120
fault20_buffer_ctl arm-all-qmm-int4     3 1 5120 5120

# 可重复在线 QMM 数值退化：每个请求 3 hit，所有请求共最多 96 hit。
# 首次 request-begin 后才会命中；REQUEST_ID 必须单调递增。
fault20_buffer_ctl arm-int4-request 0 1 5120 1 3 96 5120
fault20_buffer_ctl request-begin 1001
# 发送一次 HTTP 推理请求
fault20_buffer_ctl request-begin 1002
# 发送下一次 HTTP 推理请求；本请求的 candidate/hit 重新从 1 开始
fault20_buffer_ctl disarm

# QMM INT4 + post-Run output observe；最后一个 MAX_OUTPUT_BYTES 可省略，默认 65536。
fault20_buffer_ctl arm-int4-observe 0 1 5120 1 1 5120 65536

# request-scoped 版本：每请求最多 3 hit，所有请求共最多 96 hit，并记录每次 output stats。
fault20_buffer_ctl arm-int4-observe-request 0 1 5120 1 3 96 5120 65536
fault20_buffer_ctl request-begin 2001

# QMM extreme x1 bounded single-hit
# DEVICE M K CALL_INDEX MAX_BYTES VALUE RATIO_BPS PATTERN
fault20_buffer_ctl arm-extreme 0 1 5120 1 5120 127 100 positive_max

# QMM extreme x1 progressive（2..64 hits）
# DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES VALUE RATIO_BPS PATTERN
fault20_buffer_ctl arm-progressive-extreme 0 1 5120 1 16 5120 127 10000 positive_max

# QMM extreme x1 all-hit；mask=1 只命中 rank0/device0
# DEVICE_MASK M K MAX_BYTES VALUE RATIO_BPS PATTERN
fault20_buffer_ctl arm-all-qmm-extreme 1 1 5120 5120 127 10000 positive_max

# QMM x2/weight 只读槽位观察
# DEVICE M K CALL_INDEX MAX_WEIGHT_BYTES
fault20_buffer_ctl arm-observe-x2 0 1 5120 1 268435456

# QMM x1+x2 双 temporary buffer；已验证 x2=[5120,5120]、26,214,400 bytes
# DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES
fault20_buffer_ctl arm-lossless-x1-x2 0 1 5120 5120 1 5120 26214400

# DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES VALUE PATTERN
fault20_buffer_ctl arm-extreme-x1-x2 0 1 5120 5120 1 5120 26214400 127 positive_max

# QMM x2_scale 只读证明、无损替换与有限倍率放大
# DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES
fault20_buffer_ctl arm-observe-x2-scale 0 1 5120 5120 1 20480
fault20_buffer_ctl arm-lossless-x2-scale 0 1 5120 5120 1 20480

# DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES FACTOR
# FACTOR 仅允许 2、10、100、1000；已完成模型 E2E 的档位为 2
fault20_buffer_ctl arm-amplify-x2-scale 0 1 5120 5120 1 20480 2

# aclnnMatmul 精确 decode projection：
# self=[1,12800]、mat2=[12800,5120]、out=[1,5120]
# 仅支持 device/rank0 single-hit 的已验证路径
fault20_buffer_ctl arm-float-matmul-observe 0 1
fault20_buffer_ctl arm-float-matmul-lossless self 0 1
fault20_buffer_ctl arm-float-matmul-lossless mat2 0 1
fault20_buffer_ctl arm-float-matmul-lossless dual 0 1

# 有限 BF16 双输入幅值；EXPONENT 仅允许 48、56、57、58、59
# 首个 non-finite 会自动 disarm；已验证在 2^58 首次得到全 +Inf
fault20_buffer_ctl arm-float-matmul-finite 0 1 58

# A=2^58 正负抵消；每个 pattern 都保证 6400 正项和 6400 负项
# 仅支持 device/rank0、exact decode shape、single-hit
fault20_buffer_ctl arm-float-matmul-cancel half        0 1
fault20_buffer_ctl arm-float-matmul-cancel block256   0 1
fault20_buffer_ctl arm-float-matmul-cancel block512   0 1
fault20_buffer_ctl arm-float-matmul-cancel block1024  0 1
fault20_buffer_ctl arm-float-matmul-cancel alternating 0 1

# 真实 MatMul 正常执行后写回原 output storage；COUNT 仅允许 1/5/51/512
fault20_buffer_ctl arm-float-matmul-output-proxy pos_inf 0 1 1
fault20_buffer_ctl arm-float-matmul-output-proxy neg_inf 0 1 5
fault20_buffer_ctl arm-float-matmul-output-proxy qnan    0 1 51

# MatmulWeightNz all-hit
fault20_buffer_ctl arm-all-matmul-lossless 3 1 12800 25600
fault20_buffer_ctl arm-all-matmul-mantissa 3 1 12800 25600 bf16 3
```

完整 bounded/multi-rank 命令见 `fault20_buffer_ctl` usage 和[能力矩阵](fault20_fast_validation/reports/fault20_capability_matrix_final.md)。

控制器必须在 worker 所在 Docker IPC namespace 内执行，例如
`docker exec -e VLLM_SERVER_DEV_MODE=1 CONTAINER fault20_buffer_ctl ...`；宿主机与容器内的
同名 POSIX SHM 不是同一个控制块。progressive/all-hit 每次命中后扫描真实 QMM output，
首次发现 non-finite 或 observer/runtime 失败即自动 disarm。

## Fail-closed 边界

以下条件任一失败时，runtime 应原样调用真实 Run：

- exact API/cache/adapter 不匹配；
- `VLLM_SERVER_DEV_MODE` 不等于 `1`；
- device/dtype/format/shape/stride/offset/bytes 不匹配；
- producer address 或 Matmul storage guard 失败；
- ABI/runtime API/分配/拷贝/同步条件不满足；
- primitive 参数非法、元素未实际改变或选择条件不明确。

禁止依赖 private executor layout、长期缓存 activation 地址、创建新 `aclTensor` 填回 cache，或原地修改调用方 address vector。

## 文档

- [当前注入能力统计](fault20_fast_validation/reports/fault20_injection_capability_inventory_20260807.md)
- [最终能力矩阵](fault20_fast_validation/reports/fault20_capability_matrix_final.md)
- [最终架构](fault20_fast_validation/reports/fault20_final_architecture.md)
- [Primitive 设计](fault20_fast_validation/reports/fault20_fault_primitive_design.md)
- [CANN 8.5.1 使用说明](FAULT20_CANN851_USAGE.md)
- [Fault39 QMM 验证报告](fault20_fast_validation/reports/fault39_qmm_overflow_validation.md)
- [Fault39 Floating MatMul 路径清点](fault20_fast_validation/reports/fault39_float_matmul_path_inventory.md)
- [Fault39 Floating MatMul 有限输入到 Non-finite 输出验证](fault20_fast_validation/reports/fault39_float_matmul_finite_nonfinite_validation.md)
- [Fault39 Floating MatMul 正负抵消与中间数值范围验证](fault20_fast_validation/reports/fault39_float_matmul_cancellation_validation.md)
- [Fault39 Floating MatMul Output Inf/NaN Proxy 验证](fault20_fast_validation/reports/fault39_float_matmul_output_nonfinite_proxy_validation.md)
- [Fault39 A2 MatMul 故障复现总结](fault20_fast_validation/reports/fault39_matmul_fault_reproduction_summary.md)
- [Fault39 A3 验证交接](fault20_fast_validation/reports/fault39_a3_validation_handoff.md)
- [Fault26 allocation topology 与 single-hit underflow](fault20_fast_validation/reports/fault26_writer_allocation_topology_and_qmm_underflow_20260818.md)
- [Fault26 bounded multi-hit](fault20_fast_validation/reports/fault26_qmm_underflow_bounded_multihit_20260818.md)
- [Fault26 all-eligible-hit 与模型响应分叉](fault20_fast_validation/reports/fault26_qmm_underflow_all_eligible_20260818.md)

## 当前不支持

- 任意 ACLNN 算子自动适配；
- FP16 模型级 E2E；
- packed INT2/INT4 或 kernel/指令/硬件故障；
- prefill 大 shape、warmup/cold、non-repeat-cache；
- 并发、soak、异步 buffer pool、性能 SLA；
- 跨 CANN/torch_npu/vLLM 版本和其他架构；
- MoE、GroupedMatmul、MatmulAllReduce；
- QMM post-Run output Inf/NaN proxy（当前只完成了 floating `aclnnMatmul`）；
- 外部故障框架集成。

## Batch 共享量化尺度的 msprobe 采集

[2026-10-27 双卡采集数据与说明](artifacts/quantization/batch_shared_scale_msprobe_20261027/README.md)包含两个真实 W8A8 合批条件的正常、故障重复与恢复数据，共 140 份原生 msprobe 张量。故障数值传播可重复，两个最终答案仍正确；该结果不证明原生量化器缺陷或答案准确率下降。

实现采用目标层动态量化之后、量化矩阵乘之前的目标行 Q/scale 配对替换，用于模拟同批其他请求扩大目标量化尺度的数值后果。详见[故障实现、调用链与关键源码](artifacts/quantization/batch_shared_scale_msprobe_20261027/README.md#故障实现与调用链)及[模拟依据与适用边界](artifacts/quantization/batch_shared_scale_msprobe_20261027/README.md#为什么这种方式可以模拟故障)。

### v2：原生归约域注入的 B/F 证据

[已完成的 B/F statistics/mix 证据](artifacts/quantization/batch_shared_scale_v2_bf_statistics_20261028/README.md)包含真实 2 行、8 行条件下双 rank 的原生 dump/stack/construct、配置、完成记录、官方及独立分析报告和冻结源码。v2 在原生 DynamicQuant 执行前改变后端归约描述符，使各行共享当前 rank 的 batch 极值尺度；第 30–32 层观测窗口内，B/F 首差异定位到 `model.layers.31.mlp.down_proj → npu_dynamic_quant.127` 的输出。

原采集配置为 `task=statistics, level=mix, summary_mode=md5, tensor_list=["."]`。本次提交公开其 JSON 证据子集，完整张量包私有保留；它不是重新采集的 `summary_mode=statistics` 纯统计版。原生 JSON 与历史报告保留原字节，随包提供 SHA-256 校验；不包含 `.pt` 或模型权重。

## 故障 15：私有格式视图别名断开与原张量更新丢失

本机制此前误标为故障 27，现统一纠正为**故障 15**。[持续条件触发的实现、调用链与验收说明](experiments/fault15_alias_detach_allhit_20261028/README.md)及[新 B/F statistics/mix 证据](artifacts/private_format_as_strided/alias_detach_allhit_statistics_bf_20261028/README.md)已在 124 物理卡 5 通过 `PASS_ALL_ELIGIBLE_MECHANISM_AND_STATISTICS`，**目前仅通过机制级验收**。

注入器持续处理每次满足指定进程、rank、活动 forward、调用位置、已注册活 base 身份及 NZ29/BF16 固定几何的调用，没有首次命中后禁用或第二次匹配报错的限制。F 每次将原 base 转为逻辑等值、独立存储的 ND2 临时张量，再用相同 size/stride/offset 调用真实 as_strided；两组随后执行相同的真实 copy_ 和 original_base+1。

本次每组使用同一个注入器跨两个真实 forward，每个 forward 对同一个 base 连续写入三次且不重置，共六次匹配。F 六次全部触发，每次 view 写入正确而原 base 丢失 64 个目标元素的写回；B 每次正常更新。每次消费者差异恰好对应这 64 个位置，两组 192 个非目标均不变；每组四次未匹配对照正常透传。六个 F 临时存储两两独立，且不与任一注册 base 共享存储。

msprobe 对每个新 base 的第一次调用，首个被观测数值差异在写后原 base 读回，语义模块为 ReadOriginalBase；后续两次在 ReadBaseBefore 已看到前次差异，报告明确标为继承传播。每次再次发生的别名断开与写回丢失由独立存储关系和本次前后状态确认，不能仅凭累计命中数或持续输出差异推断。null 统计不可判相等，原生 offset 警告保留。

新采集配置为 task=statistics、level=mix、summary_mode=statistics。B/F 各包含 step0–1/rank0 的 dump/stack/construct，加 config 与 receipt，共八个原生 JSON；小规模完整逻辑值、BF16 位模式与存储证据单列 runtime JSON，没有 .pt 等张量二进制。实际微基准步骤不包装成模型 token 步骤；自然模型写回链与历史 CANN 缺陷仍未验证。

[历史单次实现](experiments/fault27_alias_detach_20261028/README.md)和[历史单次证据包](artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/README.md)保留追溯入口；旧源码、采集数据和包内 SHA 不改写，旧路径及 schema 中的 fault27 属于此前误标，不代表另一项故障。


## 故障 28：原生 FA Tiling 切分策略异常

**补充因果验证（2026-10-29）已完成：** [固定真实Q/K/V的实现与机理](experiments/fault28_causal_20261029/README.md)和[32例statistics/mix证据](artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029.md)分离了M、nRatio及交互。M-only改变27个元素，R-only 83,176个，联合83,181个；M差异不在最后query，只看末token会漏检。固定profile时B1/B8完整目标输出逐bit相同；受控C++ batch策略切换后，独立msprobe首个主输出差异在原生FA API输出Mean。该结论不等于默认SDK或历史FIA缺陷复现，也未唯一识别Reduce指令。

[实现、真实调用链与关键代码](experiments/fault28_native_tiling_20261029/README.md)及[新 B/F statistics/mix 证据](artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029.md)已在124物理卡5、910B4、TP1通过受控机理和真实HF模型传播验收。相同Batch8输入下，原生CANN FA从M128/两组1024 keys切至M64/四组512 keys；每次符合条件都触发，两次forward均验证，Batch1保护和同进程关闭恢复通过。所有合法keys仍各参与一次，无跨核SplitKV；原生返回结果未重算或替换。logits有差异，Top-1未变，未证明业务质量下降。

**独立定位通过依赖明确披露的任务私有msprobe FP32统计修正。** 未修改官方26.1.0默认FP16四统计未分辨Attention输出，首差在下游pow输入；失败证据保留，不能当成pow故障。修正统一提高所有FP16/BF16节点的统计归约精度，保持原张量/dtype/shape和模型数值不变，仍使用statistics/mix/summary_mode=statistics及公开precision=high。新B/F两step的独立分析均首先观测到layer1 SDPA输出Mean不同，输入统计匹配；完整输入字节、真实FA身份及tiling由单独证据核验。null仍是不可观测，msprobe本身不唯一证明内部tiling根因。

该实现使用Qwen3-0.6B原checkpoint统一加载为FP16的原生HF SDPA→FlashAttentionScore路径；不是已复现的历史38Bv2、vLLM FIA或TP2缺陷。匹配源码、stock/OFF逐字节兼容、合法计划、保护/重复/恢复、默认统计局限和采集器补丁均见上述README。主交付只有B/F原始JSON，无整批张量、模型权重或编译库。
