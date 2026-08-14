# Fault39 `aclnnMatmul` Output Inf/NaN Proxy 验证

## 最终判定

本轮完成了三种独立、可控的 L4 outcome proxy：

```text
MATMUL_OUTPUT_POS_INF_PROXY_PROVEN
MATMUL_OUTPUT_NEG_INF_PROXY_PROVEN
MATMUL_OUTPUT_QNAN_PROXY_PROVEN
```

这些结论表示：真实 `aclnnMatmul` 先正常执行并产生全 finite 输出，随后 L4 wrapper 在已证明的真实 output storage 上写入指定 BF16 非有限值，下游模型实际消费修改后的结果。

它们不是自然溢出，也不构成 accumulator overflow 证据。

## 1. 固定环境与点位

```text
model       Qwen3-32B-w8a8-nopdmix (dense)
image       quay.io/ascend/vllm-ascend:v0.18.0rc1
CANN        8.5.1
execution   TP=2, enforce-eager, developer mode
API         aclnnMatmul
device      rank0 / logical device0
phase       decode
call        first exact-shape candidate, single-hit
out         BF16 [1,5120], Run slot 2, 10,240 bytes
```

执行顺序：

```text
real OpExecCache::Run
  -> synchronize stream
  -> 原 output D2H
  -> 证明原始 5120 元素全部 finite
  -> host 对 prefix N 个 BF16 元素直接赋 bit pattern
  -> 整块 H2D 写回原 output storage
  -> synchronize stream
  -> D2H readback
  -> 精确核对异常类型和数量
  -> 自动 disarm
  -> wrapper 返回，下游继续执行
```

不能通过替换本次 output address 达成该目标；必须写回真实 output storage，因为下游算子继续引用原 storage。

## 2. Proxy 定义

| 模式 | BF16 bits |
|---|---|
| `assign_pos_inf` | `0x7f80` |
| `assign_neg_inf` | `0xff80` |
| `assign_qnan` | `0x7fc0` |

第一版使用 prefix selection，严格只允许：

| 元素数 | 实际占比 |
|---:|---:|
| 1 | 0.01953125% |
| 5 | 0.09765625%（约 0.1%） |
| 51 | 0.99609375%（约 1%） |
| 512 | 10% |

采用直接 IEEE bit assignment，而不是 `×Inf` 或 `×NaN`，避免把 `0×Inf→NaN` 等额外算术语义混入第一版验证。

## 3. Runtime/readback 结果

12 个档位的 mutation 前统计完全一致：

```text
finite       5120
NaN          0
+Inf         0
-Inf         0
min          -1.5390625
max          4.8125
max_abs      4.8125
mean_abs     0.080082635313738132
```

每个档位均满足：

```text
real Run status       0
sync/copy/free status 0
changed_elements      requested_elements
readback_exact        true
control auto-disarmed true
```

readback 汇总：

| Proxy | N | finite | 目标异常 | 其他 non-finite |
|---|---:|---:|---:|---:|
| `+Inf` | 1 | 5,119 | 1 | 0 |
| `+Inf` | 5 | 5,115 | 5 | 0 |
| `+Inf` | 51 | 5,069 | 51 | 0 |
| `+Inf` | 512 | 4,608 | 512 | 0 |
| `-Inf` | 1 | 5,119 | 1 | 0 |
| `-Inf` | 5 | 5,115 | 5 | 0 |
| `-Inf` | 51 | 5,069 | 51 | 0 |
| `-Inf` | 512 | 4,608 | 512 | 0 |
| `qNaN` | 1 | 5,119 | 1 | 0 |
| `qNaN` | 5 | 5,115 | 5 | 0 |
| `qNaN` | 51 | 5,069 | 51 | 0 |
| `qNaN` | 512 | 4,608 | 512 | 0 |

## 4. 模型 response 与最小可见阈值

固定请求、temperature、seed、token 上限与 cancellation 阶段一致。

baseline 为正常的 91-token checksum 解释：

```text
A checksum is a small-sized datum derived from a block of digital data ...
```

choices+usage SHA-256：

```text
2b51e9ad41cb5eccf97028e0003abcc1ad29b697d9878d95cfc0bb53612be4d6
```

12 个 fault response 全部稳定退化为：

```text
A!
```

均为 3 completion tokens，choices+usage SHA-256：

```text
03f198c119503cb5901828d44a1897d376e61108ad1fa49aee92c086656c79a5
```

因此，在当前 target output 和 prefix selection 下，已经观察到的最小模型级可见阈值是：

```text
1 / 5120 output elements
= 0.01953125%
```

这不代表任意位置或任意 MatMul 的通用阈值。

所有 12 个 fault 请求和 12 个 recovery 请求均 HTTP 200。每次 recovery 都精确恢复 baseline 文本、token 用量和哈希。

## 5. 构建与证据

```text
runtime
/home/haoran/fault39_dev_20260812/runtime/cann851_float_matmul_output_proxy_v19/

logs
/home/haoran/fault39_dev_20260812/runtime/float_matmul_output_proxy_logs_v19/
```

构建哈希：

```text
v19 runtime  788078269318b016dd465182329253cef0d114ce2b0b0ebcd65fbd7e93000567
typed proxy  d2ce5d48948ab57bdc375dbb4420d49e056dcbf455ebf25eb43e125137665393
```

JSONL 完整性：

```text
run_trace.jsonl  478,345 lines, 0 parse errors
metadata.jsonl   2,236,340 lines, 0 parse errors
```

12 个 `*_events.jsonl` 各恰好包含一条成功的 `float_matmul_output_nonfinite_proxy` 事件。

## 6. 当前运行状态

当前保留推理容器：

```text
fault39-qwen3-32b-cann851-float-matmul-output-proxy-v19
```

最终状态：

```text
health       200
Running      true
OOMKilled    false
RestartCount 0
control      version 13, mode 19, enabled 0
```

v17/v18 均已正常停止并保留，历史日志和构建产物未删除。

## 证据边界

本轮用于确定性验证诊断工具对 MatMul 输出非有限值的识别、传播和恢复能力。它明确区分于：

- `2^58 × 2^58` 的自然 `+Inf`；
- 正负抵消实验的 intermediate range failure；
- 硬件或 kernel accumulator overflow。

当前只验证固定 API、shape、rank、call index、prefix selection 和单次命中。尚未验证随机/分层位置选择、多次命中、并发、soak、其他模型或其他 CANN 版本。
