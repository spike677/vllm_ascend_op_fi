# Fault39 A3 实机验证交接

## 文档状态

本文只定义 A3 验证方案。本轮没有连接、检查或操作 A3，也没有在 A2 新增实验。

## 1. A3 目标

第一优先验证 A2 上已经得到的：

```text
MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED
```

核心问题是：

```text
finite BF16 inputs
+ theoretical output = 0
+ real aclnnMatmul
+ different K-axis cancellation layout
-> A3 是否也表现出排列相关 non-finite
```

Output Inf/NaN proxy 只作为可选的诊断工具覆盖验证，不是 A3 首要根因复核目标。

## 2. 不可继承的 A2 假设

A3 不得直接假设：

```text
API             aclnnMatmul
self slot       addresses[0]
mat2 slot       addresses[1]
out slot        addresses[2]
decode shape    [1,12800] x [12800,5120]
tiling          与 A2 相同
2^58 threshold  与 A2 相同
reduction order 与 A2 相同
```

这些项目必须在 A3 上重新证明。A3 应重新构建 `.so`，不能直接复制 A2 的 CANN 8.5.1 二进制产物跨版本使用。

## 3. 需要交接的 Git 资产

```text
fault20_fast_validation/source/l4_metadata_lifecycle_proxy/
fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/
fault20_fast_validation/reports/fault39_matmul_fault_reproduction_summary.md
fault20_fast_validation/reports/fault39_float_matmul_path_inventory.md
fault20_fast_validation/reports/fault39_float_matmul_finite_nonfinite_validation.md
fault20_fast_validation/reports/fault39_float_matmul_cancellation_validation.md
fault20_fast_validation/reports/fault39_float_matmul_output_nonfinite_proxy_validation.md
FAULT20_CANN851_USAGE.md
```

不交接到 Git：

- A2/A3 原始 `run_trace.jsonl`、`metadata.jsonl`；
- response 大证据包；
- 模型权重；
- 容器 runtime dump；
- A2 编译出的 `.so`；
- 临时脚本和本地验证目录。

## 4. 推荐验证顺序与阶段门禁

严格串行执行；前一阶段不 PASS，不进入下一阶段。

### Stage 1：环境审计

记录但不预设：

```text
host/VM/bare-metal
SoC model and revision
driver
CANN
torch / torch_npu
vLLM / vllm-ascend
image digest
arch and C++ ABI
NPU topology
```

确认只在本地开发模式启动：

```text
VLLM_SERVER_DEV_MODE=1
control enabled=0
health=200
```

### Stage 2：模型 baseline

固定 model、prompt、temperature、seed、max tokens 和 chat template。至少连续执行两次，保存 response、token/usage、HTTP、worker/health 状态。baseline 不稳定时停止。

### Stage 3：重新确认 floating MatMul API

先做只读 API inventory 和 typed GWS metadata。不得因 A2 命中 `aclnnMatmul` 就跳过。如果 A3 使用其他 API、fusion 或 shape，应先扩展只读 adapter，再评审后续实验。

### Stage 4：self/mat2/out slot proof

必须重新形成：

```text
typed GWS
-> AddParamToBuf formal collection
-> storage order/presence/uniqueness
-> AddOpExecCache
-> current OpExecCache::Run addresses
-> slot_proven
```

禁止用 `formal index == slot` 或冷态 GWS address 直接推断。

### Stage 5：self lossless

要求：

```text
original != temporary
changed_elements = 0
Run/copy/sync/free = 0
response/token == baseline
recovery == baseline
```

### Stage 6：mat2 lossless

要求同 Stage 5，并核对物理 storage bytes、stride 和 offset。

### Stage 7：dual lossless

self 和 mat2 同时替换。任何模型差异、异常状态或不精确恢复都必须停止。

### Stage 8：output observer

真实 Run 后只读 output，记录 finite/NaN/`+Inf`/`-Inf`/zero、min/max/max_abs/mean_abs。baseline output 必须全 finite。

### Stage 9：all-positive finite sweep

在 A3 重新证明 BF16 metadata 和 temporary 路径后，严格串行：

```text
self=mat2=2^57
recovery
self=mat2=2^58
recovery
```

若 `2^57` 已 non-finite，停止升级并保留阈值差异；若 `2^58` 仍 finite，也保留为跨硬件差异，不继续盲目增加幅值。

### Stage 10：cancellation comparison

只有前九阶段通过后才执行：

```text
A = 2^58
self = +A
mat2 per output column = 6400 * +A + 6400 * -A
theoretical output = 0
```

按顺序：

```text
half
recovery
block256 mirrored
recovery
block512 mirrored
recovery
block1024 mirrored
recovery
alternating
recovery
```

每档必须证明：

```text
input finite count exact
positive count == negative count
theoretical_output_exact_zero = true
changed_elements > 0
temporary addresses used
real Run = 0
observer/readback success
control disarmed
recovery exact
```

### Stage 11：与 A2 比较

不要只写 PASS/FAIL。逐项比较 API、shape、slot、阈值、各 pattern 输出分布、模型 response 和恢复状态。

### Stage 12：可选 output proxy

只有 root-cause comparison 完成后才评审。若执行，先从单个 `+Inf` 开始，再决定是否需要 `-Inf`、qNaN 和更高比例。Proxy 结果必须单独命名，不能作为自然 overflow 证据。

## 5. A2/A3 比较表

| item | A2 | A3 |
|---|---|---|
| deployment | 当前 A2 本地环境 | |
| CANN | 8.5.1 | |
| SoC | `ascend910b1`（`SOC_VERSION`） | A3 |
| vLLM image | `v0.18.0rc1` | |
| model | Qwen3-32B-w8a8-nopdmix | |
| API | `aclnnMatmul` | |
| self shape | BF16 `[1,12800]` | |
| mat2 shape | BF16 `[12800,5120]` | |
| out shape | BF16 `[1,5120]` | |
| self slot | 0 | |
| mat2 slot | 1 | |
| out slot | 2 | |
| `2^57` | finite，约 `2.658e38` | |
| `2^58` | 5,120 `+Inf` | |
| half | 2,048 `+Inf`、2,048 `-Inf`、1,024 zero | |
| block256 | 5,120 zero | |
| block512 | 5,120 zero | |
| block1024 | 5,120 zero | |
| alternating | 5,120 zero | |
| half model response | `A!` | |
| local-cancel response | finite response | |
| recovery | exact baseline | |
| worker/health | alive / 200 | |

## 6. 结果分类

### A3 与 A2 相同

若 half 产生 non-finite，而 block/alternating 为零或 finite，可记录：

```text
MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED_ON_A3
```

仍不得记录 `ACCUMULATOR_OVERFLOW_PROVEN`。

### A3 所有 cancellation 都为零

不判为实验失败。可能意味着 A3 的 kernel、tiling、reduction order 或中间精度避免了 A2 行为。这是有价值的跨硬件差异。

### A3 所有 pattern 都 non-finite

先复核符号计数、stride-aware 写入、精确数学和与 output observer；随后将结果记录为更低区分度的中间范围差异，不能直接归因 accumulator。

### A3 API/shape/slot 不同

停止数值阶段，回到只读路径发现和 slot proof。不得把 A2 hard-coded slot 迁移过去。

## 7. 停止条件

以下任一出现即停止当前阶段并 disarm：

- baseline 不稳定或 health 非 200；
- API、metadata、slot 未重新证明；
- lossless response/token 不等价；
- rank、call index 或命中次数不明确；
- 输入不是预期 finite BF16；
- stride/offset/storage bytes 不满足 guard；
- allocation/copy/sync/Run/readback/free 任一失败；
- worker crash、OOM、HCCL/CANN 错误；
- recovery 不精确；
- 首个未预期 non-finite 出现。

## 8. 最小证据包

每阶段至少保存：

```text
environment/version manifest
source and built artifact SHA-256
control arm/post/disarm status
API name and cache pointer
formal metadata and Run slot proof
input finite/positive/negative/changed counts
output finite/NaN/+Inf/-Inf/zero statistics
HTTP, response, usage/token comparison
worker/health/OOM/restart state
recovery comparison
```

原始 tensor 内容不记录。完整 JSONL 保留在 A3 实验机，不提交 Git；Git 只提交结论 Markdown 和必要的小型摘要。

## 9. A3 交接完成条件

```text
environment audited
+ baseline stable
+ API rediscovered
+ slots reproven
+ lossless gates passed
+ output observer passed
+ 2^57/2^58 compared
+ five cancellation patterns compared
+ recovery verified
+ A2/A3 table completed
+ verdict uses evidence-bounded naming
```

A3 结果无论相同或不同，都应保留为硬件代际、kernel、tiling、reduction order 或 accumulator 实现差异的证据。
