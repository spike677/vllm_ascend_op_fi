# Fault20 / Fault39 CANN 8.5.1 开发模式使用说明

## 适用边界

本能力只用于 vLLM Ascend 开发者模式下的受控故障识别验证：

```bash
export VLLM_SERVER_DEV_MODE=1
```

control 与 LD_PRELOAD runtime 会分别检查该变量。变量缺失或值不等于 `1` 时，
`arm-*` 被拒绝，runtime 原样调用真实 `OpExecCache::Run`。

当前 CANN 8.5.1 状态：

```text
runtime build            PASS
metadata proxy build     PASS
executor ABI smoke       PASS
QMM extreme offline test PASS
Qwen3-32B TP=2 E2E       PASS（single/progressive/all-hit）
QMM x2 slot proof        PASS（addresses[1]，INT8 NZ）
QMM x1+x2 temporary E2E PASS（lossless/extreme single-hit）
QMM x2_scale slot proof  PASS（formal ordinal 3，addresses[2]，FP32 ND）
QMM x2_scale temporary  PASS（lossless / finite ×2 single-hit）
Floating aclnnMatmul API/slot proof PASS（self/mat2/out -> 0/1/2）
Floating aclnnMatmul lossless       PASS（self/mat2/dual）
Finite BF16 natural non-finite      PASS（2^58 -> 5120 +Inf）
Cancellation intermediate range    PASS（half +/-Inf；local patterns zero）
MatMul output Inf/NaN proxy         PASS（3 types × 4 ratios）
QMM semantic-slot v17 build         PASS（独立 runtime）
QMM semantic-slot model E2E         PASS（2026-08-27，Qwen3-32B TP=2，NPU 2/5，Dump OFF）
QMM v17 metadata ABI handshake      PASS（显式 descriptor；不匹配 fail-closed）
QMM v17 immutable bundle E2E        PASS（C1/C2/C3/recovery，独立 harness）
```

### v17 双 NPU Dump-OFF 验证记录（2026-08-27）

远端证据目录：

```text
/home/haoran/fault20_fast_validation/runtime/cann851_semantic_slot_v1/v17e2e_r2
```

验证结果：

```text
C1 STRICT_PRODUCER  PASS：rank0/device0 hit=1，changed_elements=4722，Run/sync/free=0
C2 SEMANTIC_SLOT    PASS：x1 formal ordinal=0，Run slot=0，semantic proof=true
C3 wrong K=5119    PASS：candidate=0，hit=0，无 substitution
recovery            PASS：disarm/reset 后 HTTP/health=200，token 与 baseline 精确一致
coherent recovery   PASS：Paris，token IDs=[59604,151645]
```

交接时必须同时校验 runtime 与 metadata proxy；旧 proxy 的较短
`Fault20QmmMetadata` ABI 不能与 v17 runtime 混用：

```text
libfault20_buffer_substitute_abi0.so
d3b9c1b76704517dea6734f77de4ed5f6e441c1a0b1faaef79816b83b6e559bf

fault20_buffer_ctl
b6215dfcba4a90755238345f85f6e9b7b904b9f3d9a63ac7f911bf5688ffaa9e

cann851_metadata_proxy_v17/libopapi.so
91bdadb70587de1f1517d595260df6adcdbc07a2e6064d9a4495aa22e8bd680d
```

### v17 最终 msprobe 交接 bundle（2026-08-27）

上面的 `v17e2e_r2` 和三个 hash 是握手改造前的历史 PASS 证据，继续原样保留。
供后续 msprobe 验证使用的是另一份只读 bundle：

```text
bundle   /home/haoran/fault20_fast_validation/runtime/cann851_semantic_slot_bundle_v17_hs2
evidence /home/haoran/f20runs/v17h2
```

最终 bundle 把 preload、control、typed metadata proxy、真实 `libopapi`、manifest、
校验器和参数化 rootfs launcher 固定在同一目录中。主要 artifact：

```text
libfault20_buffer_substitute_abi0.so
02f614f8cfc2db9fc05b4fcaf550adc6dfac93cade0514065b60a607f855a2b3

fault20_buffer_ctl
bac3b00ccc607c1f7e4ce6ccc0b2dd9b091c18fa612e70c923aee4387dc33ea5

proxy/libopapi.so
2c4b2c23ea317e49847ae5ab929ba055366db3cd3d4c80c644793c1913421ea2

proxy/real/libopapi_real.so
234b7582c6c8866001dc8be727c2572e4fe5a4636ec1edd7c1ca143c83ee0996
```

metadata ABI 使用固定 32-byte descriptor：

```text
magic           0x4632304d45543137
version         17
descriptor_size 32
payload kinds   QMM=1, MatmulWeightNz=2, FloatMatmul=3
```

runtime 在读取 payload 前验证 magic、version、descriptor size、payload kind 和 payload
size；旧无版本 commit 入口只记录 `legacy_unversioned_commit_rejected`，不会解引用来历不明的
结构体。负向 smoke 已验证：版本 16 和仅 8-byte legacy buffer 均被拒绝、进程不崩溃：

```text
/home/haoran/fault20_fast_validation/runtime/cann851_semantic_slot_v17_handshake/abi_smoke/buffer.jsonl
METADATA_ABI_FAIL_CLOSED_SMOKE_PASS
```

最终 TP=2 回归使用物理 NPU 2/5，映射为逻辑 device 0/1、TP rank 0/1。模型启动时记录
4 个 `metadata_abi_handshake`，`metadata_abi_mismatch=0`；请求后 QMM metadata
`registered=18, rejected=0`。最小回归结果：

```text
C1 STRICT_PRODUCER  PASS：candidate=252，hit=1，changed_elements=3218
C2 SEMANTIC_SLOT    PASS：candidate=242，hit=1，formal ordinal=0，Run slot=0
C3 wrong K=5119    PASS：candidate=0，hit=0，无 substitution
recovery            PASS：disarm/reset 后 HTTP/health=200，Paris token IDs=[59604,151645]
runtime errors      0
```

交接前和在目标环境中都必须先校验 bundle：

```bash
BUNDLE=/home/haoran/fault20_fast_validation/runtime/cann851_semantic_slot_bundle_v17_hs2
"$BUNDLE/harness/verify_v17_bundle.sh" "$BUNDLE"
# expected: FAULT20_V17_BUNDLE_VERIFY_PASS
```

Claude/msprobe 环境必须使用 manifest 中同一组文件，加载顺序为 proxy 在真实库之前：

```bash
export VLLM_SERVER_DEV_MODE=1
export LD_LIBRARY_PATH="$BUNDLE/proxy:$BUNDLE/proxy/real:$LD_LIBRARY_PATH"
export LD_PRELOAD="$BUNDLE/preload/libfault20_buffer_substitute_abi0.so"
CTL="$BUNDLE/ctl/fault20_buffer_ctl"
```

Dump-ON 验证顺序保持最小化：D1 `STRICT_PRODUCER` 观察其候选是否因 producer 证明失败而
为 0；D2 `SEMANTIC_SLOT` 必须命中 1 次并保留 ABI handshake；D3 `disarm/reset` 后请求精确
恢复。不要重编译或混用另一个 metadata proxy，否则 manifest 校验和 ABI 闭环失效。

可重复 rootfs 启动器位于：

```text
fault20_fast_validation/harness/v17_bundle/run_v17_bundle_rootfs.sh
```

它要求 rootfs、run dir、模型和 bundle 均位于 `/home`，启动前检查 CANN 8.5.1、端口、
两张空闲 NPU、剩余空间、bundle hash 与 IPC 路径长度，并只清理自己创建的进程组和挂载。
它不会启动 dockerd、清理镜像或写入默认 `/var/lib/docker`。

## 版本

```text
image     quay.io/ascend/vllm-ascend:v0.18.0rc1
CANN      8.5.1
vLLM      0.18.0
torch_npu 2.9.0.post1+gitee7ba04
arch      aarch64
C++ ABI   _GLIBCXX_USE_CXX11_ABI=0
```

## 构建 executor-cache runtime

在目标镜像内：

```bash
export FAULT20_BUFFER_OUTPUT_DIR=/fault20/runtime/cann851_semantic_slot_v1
export FAULT20_ASCEND_TOOLKIT=/usr/local/Ascend/cann-8.5.1/aarch64-linux
export FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT=/cann-8.5.1/

bash /fault20/source/l4_buffer_substitute_allhit_v1/build.sh
```

构建时将 CANN 路径片段固化进 `.so`。不要把 8.3 构建产物复制到 8.5.1 服务。

## 构建 typed metadata proxy

```bash
export FAULT20_METADATA_PROXY_OUTPUT_DIR=/fault20/runtime/cann851_metadata_proxy_v17
export FAULT20_ASCEND_TOOLKIT=/usr/local/Ascend/cann-8.5.1/aarch64-linux
export FAULT20_REAL_LIBOPAPI=/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so

bash /fault20/source/l4_metadata_lifecycle_proxy/build.sh
```

预期产物：

```text
/fault20/runtime/cann851_metadata_proxy_v17/libopapi.so
/fault20/runtime/cann851_metadata_proxy_v17/real/libopapi_real.so
/fault20/runtime/cann851_semantic_slot_v1/libfault20_buffer_substitute_abi0.so
/fault20/runtime/cann851_semantic_slot_v1/fault20_buffer_ctl
```

## 服务加载环境

服务进程启动前设置：

```bash
export VLLM_SERVER_DEV_MODE=1
export LD_LIBRARY_PATH=/fault20/runtime/cann851_metadata_proxy_v17:$LD_LIBRARY_PATH
export LD_PRELOAD=/fault20/runtime/cann851_semantic_slot_v1/libfault20_buffer_substitute_abi0.so
export FAULT20_METADATA_LOG_PATH=/fault20/runtime/logs/metadata.jsonl
export FAULT20_BUFFER_LOG_PATH=/fault20/runtime/logs/run_trace.jsonl
```

必须由新的 vLLM 进程继承这些变量；不能向已经运行的 worker 动态补装 `.so`。

启动后先确认：

```text
/health = 200
abi_ok = true
developer_mode_ok = true
expected_library_fragment = /cann-8.5.1/
library = /usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libnnopbase.so
control enabled = 0
```

## Control v17

```text
magic   0x4632304255463137
version 17
SHM     /fault20_l4_buffer_substitute_ctl_v17
```

v17 保留旧 `arm-int4` 的 `STRICT_PRODUCER` 行为，并新增独立的
`arm-int4-slotproof`（`SEMANTIC_SLOT`）策略。新策略仅在 QMM formal x1 的 ordinal、
compact Run slot、参数顺序、地址数量、空槽规则、storage 唯一性及 shape/dtype/layout/bytes
全部证明后才计入 candidate。direct GWS address match 只记录诊断，不作为放行条件。

基础命令：

```bash
CTL=/fault20/runtime/cann851_semantic_slot_v1/fault20_buffer_ctl

docker exec -e VLLM_SERVER_DEV_MODE=1 CONTAINER "$CTL" status
docker exec -e VLLM_SERVER_DEV_MODE=1 CONTAINER "$CTL" reset
docker exec -e VLLM_SERVER_DEV_MODE=1 CONTAINER "$CTL" disarm

# 历史兼容 producer proof
$CTL arm-int4 0 1 5120 1 1 5120

# 新 formal semantic-slot proof
$CTL arm-int4-slotproof 0 1 5120 1 1 5120
```

控制器必须在 vLLM worker 所在容器的 IPC namespace 内执行。Docker 默认隔离 POSIX SHM；
直接在宿主机执行同一路径的控制器会访问另一个同名 SHM，不会控制容器内 worker。

Fault39 QMM x1 extreme（只允许 bounded single-hit）：

```bash
# DEVICE M K CALL_INDEX MAX_BYTES VALUE RATIO_BPS PATTERN
$CTL arm-extreme 0 1 5120 1 5120 127 100 positive_max
$CTL arm-extreme 0 1 5120 1 5120 127 1000 positive_max
$CTL arm-extreme 0 1 5120 1 5120 127 5000 positive_max
$CTL arm-extreme 0 1 5120 1 5120 127 10000 positive_max
```

Progressive 与 all-hit：

```bash
# DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES VALUE RATIO_BPS PATTERN
$CTL arm-progressive-extreme 0 1 5120 1 3  5120 127 10000 positive_max
$CTL arm-progressive-extreme 0 1 5120 1 16 5120 127 10000 positive_max
$CTL arm-progressive-extreme 0 1 5120 1 64 5120 127 10000 positive_max

# DEVICE_MASK M K MAX_BYTES VALUE RATIO_BPS PATTERN；mask=1 只命中 rank0/device0
$CTL arm-all-qmm-extreme 1 1 5120 5120 127 10000 positive_max
```

QMM x2/weight 槽位只读观察与双 temporary buffer：

```bash
# DEVICE M K CALL_INDEX MAX_WEIGHT_BYTES
$CTL arm-observe-x2 0 1 5120 1 268435456

# DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES
$CTL arm-lossless-x1-x2 0 1 5120 5120 1 5120 26214400

# DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES VALUE PATTERN
$CTL arm-extreme-x1-x2 0 1 5120 5120 1 5120 26214400 127 positive_max
```

双缓冲模式只接受已经证明的 x2 runtime slot 和 metadata。当前 E2E 固定为 x1 `[1,5120]`
连续 INT8 ND storage 及 x2 `[5120,5120]` INT8 NZ storage（physical storage
`[160,320,16,32]`，26,214,400 bytes）。它不会原地修改 activation、weight 或调用方
address vector；临时 host/device buffer 在真实 Run 同步完成后回收。

QMM x2_scale 只读证明与 temporary substitution：

```bash
# DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES
$CTL arm-observe-x2-scale 0 1 5120 5120 1 20480
$CTL arm-lossless-x2-scale 0 1 5120 5120 1 20480

# DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES FACTOR
$CTL arm-amplify-x2-scale 0 1 5120 5120 1 20480 2
```

当前已证明 scale 角色为 `x2_scale`，formal ordinal 3，对应 compact Run slot
`addresses[2]`；其 dtype/format/shape/stride/bytes 为 FP32、ND、`[5120]`、`[1]`、20,480。
`x1_scale` 不存在，`y_scale` 是零字节 placeholder。放大命令只允许倍率
`2/10/100/1000`，但当前模型级 E2E 只验证了 `×2`；不得据此自动宣称其他倍率有效。
scale substitution 在 pre-sync 后 D2H 到 pinned host，变换完成后 H2D 到 temporary device
buffer，复制 Run address vector 并只替换 slot 2；真实 Run 与 post-sync 完成后释放 temporary
和 host buffer。原 scale storage 不回写。

Fault39 floating `aclnnMatmul`（以下参数只对应已验证的 A2 decode projection）：

```bash
# DEVICE CALL_INDEX
$CTL arm-float-matmul-observe 0 1

# TARGET(self|mat2|dual) DEVICE CALL_INDEX
$CTL arm-float-matmul-lossless self 0 1
$CTL arm-float-matmul-lossless mat2 0 1
$CTL arm-float-matmul-lossless dual 0 1

# DEVICE CALL_INDEX EXPONENT；A2 已验证 57 为 finite、58 为全 +Inf
$CTL arm-float-matmul-finite 0 1 57
$CTL arm-float-matmul-finite 0 1 58

# PATTERN DEVICE CALL_INDEX
$CTL arm-float-matmul-cancel half 0 1
$CTL arm-float-matmul-cancel block256 0 1
$CTL arm-float-matmul-cancel block512 0 1
$CTL arm-float-matmul-cancel block1024 0 1
$CTL arm-float-matmul-cancel alternating 0 1

# MODE(pos_inf|neg_inf|qnan) DEVICE CALL_INDEX COUNT
$CTL arm-float-matmul-output-proxy pos_inf 0 1 1
$CTL arm-float-matmul-output-proxy neg_inf 0 1 1
$CTL arm-float-matmul-output-proxy qnan 0 1 1
```

A2 已证明该调用的 self、mat2、out 分别对应 compact Run slot 0、1、2。这个映射不能直接
带到 A3；A3 必须先重新发现实际 API、formal tensor 与 Run slot，再执行无损替换。

`arm-progressive-extreme` 只接受 2–64 hits；all-hit 使用显式 selection mode，不能用任意大
bounded hit 数模拟。每次 extreme 成功后都会扫描真实 QMM output；首次发现 Inf/NaN 或任一
runtime/observer 失败时，runtime 原子设置 `enabled=0`，后续 Run 原样执行。

其他已实现但不属于第一轮模型验证的 pattern：

```bash
$CTL arm-extreme 0 1 5120 1 5120 -128 100 negative_max
$CTL arm-extreme 0 1 5120 1 5120 127 100 alternating
```

## 模型 E2E 顺序

严格串行执行：

```text
1. baseline observer
2. lossless single-hit，要求 response/token 与 baseline 一致
3. progressive 3-hit -> recovery
4. progressive 16-hit -> recovery
5. progressive 64-hit -> recovery
6. all-hit -> recovery
7. x2/weight read-only observe
8. x1+x2 lossless single-hit，要求 response/token 与 baseline 精确一致
9. x1+x2=127 single-hit -> recovery
10. x2_scale read-only observe
11. x2_scale lossless single-hit，要求 response/token 与 baseline 精确一致
12. x2_scale ×2 single-hit -> recovery
13. floating MatMul API/slot read-only proof
14. self、mat2、dual lossless，逐项要求 response/token 与 baseline 精确一致
15. floating MatMul output observer
16. finite dual input：2^57 -> recovery；2^58 -> recovery
17. cancellation：half、block256、block512、block1024、alternating；每项后 recovery
18. 可选 output proxy：+Inf、-Inf、qNaN；每项后 recovery
```

每一级保存 HTTP、token IDs、first differing token、worker 状态、control 状态和
`qmm_extreme_x1_substitution`/`qmm_output_observation` 日志。必须确认命中数准确、
`changed_elements > 0`、temporary address 实际被使用、真实 Run 成功、observer 成功且
post 恢复。任一级出现 non-finite 或运行失败即停止升级。

## 结论命名

当前可使用：

```text
CANN851_RUNTIME_BUILD_PASS
CANN851_METADATA_PROXY_BUILD_PASS
CANN851_ABI_SMOKE_PASS
QMM_EXTREME_X1_PROVEN
QMM_PROGRESSIVE_EXTREME_X1_PROVEN
QMM_ALL_HIT_EXTREME_X1_PROVEN
X1_EXTREME_NO_NONFINITE
QMM_X2_SLOT_PROVEN
QMM_X1_X2_LOSSLESS_SUBSTITUTION_PROVEN
QMM_X1_X2_EXTREME_PROVEN
X1_X2_EXTREME_NO_NONFINITE
QMM_X2_SCALE_SLOT_PROVEN
QMM_X2_SCALE_LOSSLESS_SUBSTITUTION_PROVEN
QMM_X2_SCALE_AMPLIFY_X2_PROVEN
QMM_X2_SCALE_X2_NO_NONFINITE
MATMUL_FLOAT_API_SLOT_PROVEN
MATMUL_FLOAT_LOSSLESS_SUBSTITUTION_PROVEN
MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED
MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED
MATMUL_OUTPUT_POS_INF_PROXY_PROVEN
MATMUL_OUTPUT_NEG_INF_PROXY_PROVEN
MATMUL_OUTPUT_QNAN_PROXY_PROVEN
```

本次 all-hit 的 12,160 次真实 QMM output，以及单次 x1+x2=127 的 5,120 个真实 QMM
BF16 output 均为 finite。x2_scale ×2 后的 5,120 个 scale 与 5,120 个真实 QMM BF16 output
同样全部 finite；模型生成从第 5 个 token（0-based index 4）开始分叉，解除后的 96-token
序列与 baseline 精确恢复。因此这些 QMM 结果不得命名为
`EXTREME_MATMUL_NONFINITE_REPRODUCED`。

floating `aclnnMatmul` 的 A2 证据不同：双输入 `2^58` 时真实 MatMul 的 5,120 个 BF16 输出
全部为 `+Inf`；同样的有限输入在 cancellation 实验中，half 排列产生 `+Inf/-Inf/0`，而
block256/512/1024 与 alternating 产生全零 finite 输出。该结果可准确命名为
`MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED`。它证明有限输入导致真实 MatMul 中间范围故障，
但没有定位到 kernel/硬件 accumulator，任何情况下都不得写成
`ACCUMULATOR_OVERFLOW_PROVEN`。

历史 QMM control v13 严格构建产物（CANN 8.5.1，`-Wall -Wextra -Werror`）：

```text
libfault20_buffer_substitute_abi0.so
707f061aa0ee89df96a186fd9b6d9f9d24332a09a40afdc5d29b7dff40afe7f7

fault20_buffer_ctl
cd90018d71737638772b76ca5f9d4474346e0d8257f3133b57fbe42cd7695056
```

A2 最新 floating MatMul / cancellation / output proxy runtime（v19）SHA-256：

```text
libfault20_buffer_substitute_abi0.so
788078269318b016dd465182329253cef0d114ce2b0b0ebcd65fbd7e93000567
```

该哈希仅用于 A2 证据关联。A3 必须在目标 CANN/驱动/架构环境重新构建并记录新哈希，不能
直接复用 A2 二进制。
