# vLLM Ascend Operator Fault Injection

这是一个面向 vLLM-Ascend/CANN 的开发者模式原生算子故障注入原型。实现覆盖 CANN repeat executor-cache replay 边界、typed operator adapter 和 temporary device buffer 替换，并提供受控、可审计的输入数值退化、非有限输出代理与 allocation-boundary 写错地址代理。

本仓库仅包含最终运行源码、构建/容器脚本和能力说明。请求响应、token 对比、trace、结构化 evidence、离线测试程序和历史探针不上传。

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

## 构建

必须在与推理服务相同的 vLLM-Ascend/CANN 镜像内构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export FAULT20_BUFFER_OUTPUT_DIR=/fault20/runtime/l4_allhit_v1
bash /fault20/source/l4_buffer_substitute_allhit_v1/build.sh

export FAULT20_METADATA_PROXY_OUTPUT_DIR=/fault20/runtime/l4_metadata_lifecycle_proxy
bash /fault20/source/l4_metadata_lifecycle_proxy/build.sh
```

CANN 8.5.1 独立构建必须显式锁定 toolkit 与真实库路径片段：

```bash
export FAULT20_BUFFER_OUTPUT_DIR=/fault20/runtime/cann851_validation_v1
export FAULT20_ASCEND_TOOLKIT=/usr/local/Ascend/cann-8.5.1/aarch64-linux
export FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT=/cann-8.5.1/
bash /fault20/source/l4_buffer_substitute_allhit_v1/build.sh

export FAULT20_METADATA_PROXY_OUTPUT_DIR=/fault20/runtime/cann851_metadata_proxy
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

control v13：

```text
magic   0x4632304255463133
version 13
SHM     /fault20_l4_buffer_substitute_ctl_v13
```

## 控制命令

默认必须保持 disabled。所有 `arm-*` 命令及实际 runtime substitution 都要求进程环境中
`VLLM_SERVER_DEV_MODE=1`；否则 fail closed。示例中的 device mask `3` 表示已验证的两个逻辑设备 0/1。

```bash
fault20_buffer_ctl status
fault20_buffer_ctl reset
fault20_buffer_ctl disarm

# QMM all-hit
fault20_buffer_ctl arm-all-qmm-lossless 3 1 5120 5120
fault20_buffer_ctl arm-all-qmm-int2     3 1 5120 5120
fault20_buffer_ctl arm-all-qmm-int4     3 1 5120 5120

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
