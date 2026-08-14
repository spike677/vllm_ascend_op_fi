# Fault39 MatMul 数值范围故障复现总结（A2）

## 文档目的

本文汇总 A2 上已经完成的 Fault39 证据，供代码审查、故障诊断工具开发和 A3 跨硬件复核使用。本文不把可观测的 non-finite 结果外推为已定位的硬件 accumulator 故障。

A2 当前最准确的模型级结论是：

```text
MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED
```

其因果条件为：

```text
finite BF16 inputs
+ theoretical output = 0
+ identical magnitude / positive count / negative count
+ different K-axis arrangement
-> different real aclnnMatmul non-finite behavior
```

本文明确不使用：

```text
ACCUMULATOR_OVERFLOW_PROVEN
```

## 1. Fault39 historical fault definition

Fault39 的历史语义是矩阵乘数值执行期间的范围异常：中间累加或 reduction 数值失控，产生 Inf/NaN，并继续传播为 token、文本或服务异常。A2 复现工作的目标分成两层：

1. **Root-cause approximation**：让有限输入进入真实 MatMul，寻找真实算子自然产生 non-finite 的条件，并尽量区分最终输出超界与中间部分和超界。
2. **Outcome proxy**：在真实 MatMul 正常完成后，确定性写入少量 Inf/NaN，用于验证诊断工具的检测、传播和恢复能力。

A2 没有修改模型权重、原始 activation 或 executor cache 私有布局；所有输入实验都通过临时 buffer，output proxy 则只写回本次已经证明的真实 output storage。

## 2. Injection architecture

### 2.1 环境

```text
model       Qwen3-32B-w8a8-nopdmix (dense)
image       quay.io/ascend/vllm-ascend:v0.18.0rc1
CANN        8.5.1
vLLM        0.18.0
execution   TP=2, eager, repeat executor cache
SoC marker  ascend910b1 (container SOC_VERSION)
C++ ABI     _GLIBCXX_USE_CXX11_ABI=0
```

### 2.2 开发模式门禁

控制面和 runtime 都要求：

```text
VLLM_SERVER_DEV_MODE=1
```

变量缺失或不等于 `1` 时，控制器拒绝 `arm-*`，runtime 原样调用真实 Run。所有模式默认 `enabled=0`，单次实验后显式或自动 disarm。

### 2.3 调用链

```text
typed aclnn*GetWorkspaceSize
  -> 读取正式 aclTensor metadata
  -> AddParamToBuf 收集 formal tensor 顺序
  -> AddOpExecCache 关联 immutable metadata 与 cache pointer
  -> OpExecCache::Run 取得本次动态 addresses
  -> storage-order / presence / uniqueness proof
  -> temporary buffer substitution 或 post-Run observer/proxy
  -> 真实 OpExecCache::Run
```

关键边界：

- 不使用 `formal index == Run slot` 的假设；
- 不从 opaque executor wrapper 猜私有布局；
- 不长期缓存冷态 activation/device address；
- 不创建新 `aclTensor` 填回既有 cache；
- 不原地修改调用方的 address vector；
- metadata、版本、ABI、shape、stride、offset、bytes 或 slot proof 任一失败即 fail closed。

### 2.4 Fault39 两条能力线

```text
Fault39
   |
   +-- Root-cause approximation
   |      |
   |      +-- finite BF16 dual input
   |      |      -> real MatMul natural +Inf
   |      |
   |      +-- cancellation / partial-sum pattern
   |             -> theoretical output = 0
   |             -> real MatMul natural +/-Inf for half layout
   |
   +-- Outcome proxy
          |
          +-- real MatMul finite output
          +-- direct output +Inf / -Inf / qNaN
          +-- downstream detection / propagation / recovery
```

## 3. QMM INT8 exploratory route

量化 dense 模型首先命中了真实 QMM：

```text
aclnnQuantMatmulWeightNz
```

已证明的 QMM 运行时角色：

```text
x1 activation   -> addresses[0], INT8 ND
x2 weight       -> addresses[1], INT8 NZ
```

QMM x1 极值化使用临时 INT8 buffer，将选中元素写为 127。已经完成：

```text
single-hit 1% / 10% / 50% / 100%
progressive 3 / 16 / 64 hits
rank0 all-hit 12,160 / 12,160 hits
```

每次均进入真实 QMM 并改变模型 token；all-hit 的 12,160 次真实 BF16 output observer 全部 finite，峰值 `max_abs=1480`，没有 NaN/Inf。

随后完成 x1+x2 双临时 buffer：

- lossless 双缓冲与 baseline response/token 精确一致；
- `x1=x2=127` 单次命中使模型输出明显坍缩；
- 真实 QMM 5,120 个 BF16 输出仍全部 finite，`max_abs=872`。

因此 QMM 路线证明了可控输入替换和模型级传播，但没有复现 non-finite：

```text
QMM_EXTREME_X1_PROVEN
QMM_PROGRESSIVE_EXTREME_X1_PROVEN
QMM_ALL_HIT_EXTREME_X1_PROVEN
QMM_X1_X2_EXTREME_PROVEN
X1_EXTREME_NO_NONFINITE
X1_X2_EXTREME_NO_NONFINITE
```

## 4. QMM x2_scale route

正式 tensor 和 runtime slot 证明为：

```text
role            x2_scale
formal ordinal  3
Run slot         addresses[2]
dtype/format     FP32 / ND
shape/stride     [5120] / [1]
bytes            20,480
```

slot 2 来自 formal tensor 收集、非空 storage compact order、cache 关联和当前 Run vector，不是 ordinal 3 到 slot 2 的猜测。

验证结果：

- lossless temporary scale：response/token 与 baseline 精确一致；
- scale `x2`：5,120 个元素全部改变，真实 Run 成功；
- 真实 QMM BF16 output 仍为 5,120 finite，`max_abs=0.283203125`；
- token 从 0-based index 4 分叉，disarm 后精确恢复。

对应结论：

```text
QMM_X2_SCALE_SLOT_PROVEN
QMM_X2_SCALE_LOSSLESS_SUBSTITUTION_PROVEN
QMM_X2_SCALE_AMPLIFY_X2_PROVEN
QMM_X2_SCALE_X2_NO_NONFINITE
```

## 5. Floating aclnnMatmul path discovery

QMM 路线无法自然产生 non-finite 后，API inventory 和 typed GWS 证明模型还存在真实 floating MatMul 主路径：

```text
aclnnMatmul
```

目标 decode projection：

```text
self  BF16 [1,12800],     stride [12800,1]
mat2  BF16 [12800,5120], stride [1,12800]
out   BF16 [1,5120],     stride [5120,1]
```

只读请求完成后，模型 response 与无 preload baseline 一致，worker 和 health 正常。路径清点阶段只证明 API 和 metadata，尚未用冷态地址推断动态 Run slot。

## 6. Slot proof

正式证明链为：

```text
typed aclnnMatmul GWS
  -> AddParamToBuf formal collection order
  -> storage presence / empty / sharing metadata
  -> AddOpExecCache(cache)
  -> registry[cache] = immutable formal metadata
  -> current OpExecCache::Run(cache, addresses)
  -> compact non-empty storage order
  -> exact address count and unique storage proof
```

最终结果：

```text
self  -> addresses[0]
mat2  -> addresses[1]
out   -> addresses[2]
```

同时满足：

```text
formal_ordinals      self=0, mat2=1, out=2
storage_order_slots  self=0, mat2=1, out=2
address_count        3
all_present          true
no_empty_storage     true
no_shared_storage    true
all slot_proven      true
```

GWS 记录的 device address 与目标动态 Run address 均不相等，直接地址匹配为 `-1`。这排除了依赖冷态地址的错误证明方式。

## 7. Lossless dual-buffer proof

严格串行验证：

```text
self-only temporary substitution
mat2-only temporary substitution
self + mat2 dual temporary substitution
```

三项均满足：

```text
changed_elements                0
original address != temporary  true
real Run                       0
copy / sync / free             0
baseline == lossless == recovery
```

其中 self 为 25,600 bytes，mat2 为 131,072,000 bytes。该门禁证明两个 BF16 输入可以在不修改原 Tensor 或模型参数的情况下，通过 temporary device buffer 进入真实 `aclnnMatmul`。

## 8. Finite-input natural non-finite result

将 self 和 mat2 的临时 storage 全部填充为同一个有限 BF16 幅值 `A`：

| A | 真实 `aclnnMatmul` 输出 |
|---|---|
| `2^48` | 5,120 finite，约 `1.0141e33` |
| `2^56` | 5,120 finite，约 `6.6461e37` |
| `2^57` | 5,120 finite，约 `2.6585e38` |
| `2^58` | 5,120 `+Inf` |
| `2^59` | 未执行；首个 non-finite 后停止 |

由此得到：

```text
MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED
```

但 `2^58` 的精确 dot-product 本身约为 `1.063e39`，已经超过 BF16/FP32 最大有限范围。Executor boundary 无法区分首次 non-finite 出现在乘法、FP32 accumulator、block reduction 还是 output cast，所以该结果不能证明 accumulator overflow。

## 9. Cancellation / intermediate-range failure

### 9.1 构造

固定：

```text
A = 2^58
self[k] = +A
mat2 每个输出列 = 6400 个 +A + 6400 个 -A
```

每个单乘积 `A^2=2^116` 有限，每个输出的精确数学结果为：

```text
6400*A^2 - 6400*A^2 = 0
```

真实 mat2 为 stride `[1,12800]` 的转置 view，写入严格按：

```text
offset = k * stride[0] + n * stride[1]
       = k + n * 12800
```

所有 pattern 的 dtype、shape、幅值、正负数量和最终数学答案相同，只改变 K 轴排列。

### 9.2 结果

| K-axis pattern | theoretical | real `aclnnMatmul` output | model response |
|---|---:|---|---|
| `half` | 0 | 2,048 `+Inf`、2,048 `-Inf`、1,024 zero | `A!` |
| `block256` | 0 | 5,120 zero | finite response |
| `block512` | 0 | 5,120 zero | finite response |
| `block1024` | 0 | 5,120 zero | finite response |
| `alternating` | 0 | 5,120 zero | finite response |

每档都证明 65,548,800 个输入元素实际改变、输入仍全部 finite、真实 Run 成功，并在 disarm 后精确恢复 baseline。

### 9.3 判定

`half` 的最终数学结果为 0，不能再用“正确最终结果本身超出输出 dtype”解释其 `+/-Inf`。相同数学问题只因 K 轴排列不同就得到 non-finite 或精确零，构成单次真实 MatMul 内 partial-sum、tile reduction 或执行顺序影响数值范围的强证据：

```text
MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED
```

仍不能命名为 `ACCUMULATOR_OVERFLOW_PROVEN`，因为尚未定位 kernel 内 accumulator 类型、具体位置和首次 overflow 时刻。

## 10. Output Inf/NaN proxy

Outcome proxy 与自然 cancellation 结果是两种独立能力。它先让真实 MatMul 正常执行并证明 output 全 finite，再在 post-Run 阶段写回原 output storage：

```text
+Inf  0x7f80
-Inf  0xff80
qNaN  0x7fc0
```

每类分别验证 1、5、51、512 个 prefix 元素，共 12 个 case。所有 case 均满足：

```text
pre-mutation output  5120 finite
changed_elements     requested_elements
readback             exact type and count
Run/copy/sync/free   0
fault HTTP           200
recovery             exact baseline
```

对应能力：

```text
MATMUL_OUTPUT_POS_INF_PROXY_PROVEN
MATMUL_OUTPUT_NEG_INF_PROXY_PROVEN
MATMUL_OUTPUT_QNAN_PROXY_PROVEN
```

当前 target 下，单个 non-finite 元素（`1/5120 = 0.01953125%`）已经使 response 稳定退化为 `A!`。这是诊断工具的结果覆盖能力，不是自然 overflow 根因证据。

## 11. Model-level anomaly

最接近历史 Fault39 的完整自然传播链已经形成：

```text
finite BF16 self/mat2
  -> theoretical dot-product = 0
  -> half cancellation layout
  -> real aclnnMatmul output contains +/-Inf
  -> downstream model consumes output
  -> response degenerates to "A!"
  -> disarm
  -> response/token/usage recover exactly
```

自然 cancellation 和 direct proxy 都没有令 worker 崩溃；HTTP 仍可为 200，因此诊断工具不能只依赖 HTTP 或进程存活状态判断数值故障。

## 12. Evidence hierarchy

| 层级 | 证据 | 状态 |
|---|---|---|
| API | 在线真实 floating API 为 `aclnnMatmul` | PROVEN |
| Metadata | self/mat2/out BF16 shape/stride/bytes | PROVEN |
| Slot | current Run addresses 0/1/2 | PROVEN |
| Substitution | self、mat2、dual lossless temporary | PROVEN_E2E |
| Natural range | finite dual input 到全 `+Inf` | `MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED` |
| Intermediate range | theoretical zero、排列相关 `+/-Inf` | `MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED` |
| Model propagation | natural `+/-Inf` 到 response `A!` | PROVEN_E2E |
| Outcome proxy | `+Inf/-Inf/qNaN`、1/5/51/512 | PROVEN_E2E |
| Exact kernel root | accumulator dtype/location/first overflow | NOT PROVEN |

## 13. Current limitations

当前结论严格限定于 A2 的固定模型、API、shape、rank0 decode first-hit、CANN/vLLM 版本和 eager repeat-cache 路径。尚未证明：

- 具体 kernel、tilingKey、accumulator dtype 或首次 overflow 指令；
- prefill shape、其他 projection 或其他 floating MatMul；
- multi-hit cancellation、并发、soak、异步 buffer pool和性能 SLA；
- 其他模型、MoE、GroupedMatmul、MatmulAllReduce；
- A3 或其他硬件代际具有相同 API、slot、tiling、threshold 或 reduction order；
- output proxy 的 prefix 位置结论可外推到任意元素位置。

原始 JSONL、response 包、容器 runtime dump 和模型文件只保留在实验机，不进入 Git。

## 14. A3 validation objectives

A3 的第一目标不是重放 output proxy，而是重新证明并比较：

```text
finite BF16
+ real aclnnMatmul
+ half / block / alternating cancellation
-> 是否仍出现排列相关 non-finite
```

A3 必须从环境、API、metadata、slot 和 lossless 门禁重新开始，不能复用 A2 假设。结果与 A2 不同不代表失败；差异可能揭示硬件代际、kernel、tiling、reduction order 或 accumulator 实现不同。

完整顺序和比较表见 [Fault39 A3 验证交接](fault39_a3_validation_handoff.md)。

## 证据索引

- [QMM 设计](fault39_qmm_overflow_design.md)
- [QMM E2E 验证](fault39_qmm_overflow_validation.md)
- [Floating MatMul 路径清点](fault39_float_matmul_path_inventory.md)
- [Floating MatMul slot/lossless/finite scan](fault39_float_matmul_finite_nonfinite_validation.md)
- [Cancellation 与 intermediate range](fault39_float_matmul_cancellation_validation.md)
- [Output Inf/NaN proxy](fault39_float_matmul_output_nonfinite_proxy_validation.md)

这些 Markdown 报告进入 Git；其引用的远端 JSONL 和 response 文件不进入 Git。
