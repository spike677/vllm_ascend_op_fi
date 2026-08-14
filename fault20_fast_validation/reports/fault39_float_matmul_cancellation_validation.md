# Fault39 `aclnnMatmul` 正负抵消与中间数值范围验证

## 最终判定

本轮在固定模型、固定 shape、rank0 decode single-hit 条件下得到：

```text
MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED
```

证据是：输入均为有限 BF16，且每个输出元素的精确数学结果为 0；仅改变 K 轴符号排列，真实 `aclnnMatmul` 输出便在“全零”和“正负 Inf”之间切换。

本轮仍不能命名为：

```text
ACCUMULATOR_OVERFLOW_PROVEN
```

因为当前证据没有定位 kernel 内 accumulator 的 dtype、位置、分块方式或首次溢出的指令。

## 1. 已验证对象

```text
model       Qwen3-32B-w8a8-nopdmix (dense)
image       quay.io/ascend/vllm-ascend:v0.18.0rc1
CANN        8.5.1
execution   TP=2, enforce-eager, developer mode
API         aclnnMatmul
device      rank0 / logical device0
phase       decode
call        first exact-shape candidate, single-hit
self        BF16 [1,12800], stride [12800,1], Run slot 0
mat2        BF16 [12800,5120], stride [1,12800], Run slot 1
out         BF16 [1,5120], stride [5120,1], Run slot 2
```

实现仍使用已经通过无损门禁的 L4 executor-cache 路径：

```text
GWS typed metadata
  -> AddOpExecCache registry
  -> OpExecCache::Run runtime addresses
  -> D2H original self/mat2
  -> host 构造有限 BF16 temporary 内容
  -> H2D temporary device buffers
  -> 替换 Run slots 0/1
  -> real aclnnMatmul Run
  -> 原 output storage D2H observer
  -> 释放 temporary buffers
```

原模型 storage 不被永久修改。

## 2. 数学构造

固定：

```text
A = 2^58
self[k] = +A
```

`mat2` 的每个输出列均包含：

```text
6400 个 +A
6400 个 -A
```

因此每个输出的精确数学结果为：

```text
6400 * A^2 - 6400 * A^2 = 0
```

每个单乘积 `A^2 = 2^116` 仍是有限值；但 6400 个同号项的理论部分和约为 `2^128.64`，超过 BF16/FP32 最大有限范围。这使实验能够区分“最终数学结果超界”和“中间 reduction/partial-sum 范围不足”。

### stride-aware 写入

真实 `mat2` 是转置 view，不能按物理 buffer 的连续前后半区直接写入。所有 pattern 都按正式 metadata 寻址：

```text
offset = view_offset + k * stride[0] + n * stride[1]
       = k + n * 12800
```

每次严格证明：

```text
self finite elements = 12,800
mat2 finite elements = 65,536,000
mat2 positive        = 32,768,000
mat2 negative        = 32,768,000
changed total        = 65,548,800
unchanged total      = 0
theoretical output   = exact zero
```

## 3. Pattern 定义

| Pattern | K 轴规则 |
|---|---|
| `half` | `k<6400` 为正，其余为负 |
| `block256_mirrored` | 前半区按 256 元素块交替，后半区逐点取反 |
| `block512_mirrored` | 前半区按 512 元素块交替，后半区逐点取反 |
| `block1024_mirrored` | 前半区按 1024 元素块交替，后半区逐点取反 |
| `alternating` | 偶数 k 为正，奇数 k 为负 |

mirrored 规则保证即使 block size 不能整除 6400，每个 `k` 和 `k+6400` 仍精确异号，最终数学和严格为 0。

## 4. 真实 MatMul 输出

| Pattern | finite | zero | `+Inf` | `-Inf` | NaN | real Run |
|---|---:|---:|---:|---:|---:|---:|
| `half` | 1,024 | 1,024 | 2,048 | 2,048 | 0 | 0 |
| `block256_mirrored` | 5,120 | 5,120 | 0 | 0 | 0 | 0 |
| `block512_mirrored` | 5,120 | 5,120 | 0 | 0 | 0 | 0 |
| `block1024_mirrored` | 5,120 | 5,120 | 0 | 0 | 0 | 0 |
| `alternating` | 5,120 | 5,120 | 0 | 0 | 0 | 0 |

最关键的对照是：

```text
same dtype / shape / magnitude / positive count / negative count
same exact mathematical result = 0

half ordering
  -> 2048 +Inf, 2048 -Inf, 1024 zero

block or alternating ordering
  -> 5120 exact zero
```

因此，`half` 的非有限输出不能由最终数学答案超出 output dtype 范围解释；结果对 K 轴排列敏感，构成中间 partial-sum / tile reduction 数值范围异常的强证据。

## 5. 模型 response

固定请求：

```text
Explain what a checksum is and why it is useful in software systems.
Answer in one concise paragraph.
```

固定 `temperature=0`、seed、128 max tokens，并关闭 thinking。

baseline 为正常的 91-token checksum 解释，choices+usage SHA-256：

```text
2b51e9ad41cb5eccf97028e0003abcc1ad29b697d9878d95cfc0bb53612be4d6
```

`half` 的实际 response：

```text
A!
```

仅生成 3 tokens，choices+usage SHA-256：

```text
03f198c119503cb5901828d44a1897d376e61108ad1fa49aee92c086656c79a5
```

三个 block pattern 和 alternating 的 response 均为另一段语义仍正确但措辞不同的 74-token checksum 解释，哈希相同：

```text
9bae61e9b8ac49acbc27c4fecdd93ce84aa42863c2571da23b30e47abb94ecb0
```

所有 fault 请求均 HTTP 200。每档显式 disarm 后的 recovery 均再次生成 baseline 的 91-token 文本，choices+usage 哈希全部精确恢复为 `2b51e9...`。

## 6. 证据和构建产物

```text
runtime
/home/haoran/fault39_dev_20260812/runtime/cann851_float_matmul_cancellation_v18/

logs
/home/haoran/fault39_dev_20260812/runtime/float_matmul_cancellation_logs_v18/
```

构建哈希：

```text
v18 runtime  dd870d5ac11b2bddcd1a329a06a0ba3ae9d00d9ee1c69dc52099393177ce8b00
typed proxy d2ce5d48948ab57bdc375dbb4420d49e056dcbf455ebf25eb43e125137665393
```

JSONL 完整性：

```text
run_trace.jsonl  329,351 lines, 0 parse errors
metadata.jsonl   1,539,492 lines, 0 parse errors
```

每个 `*_events.jsonl` 恰好包含一条 cancellation substitution 和一条 post-Run output observation。

## 7. 该阶段完成时的运行状态

该阶段完成时保留的推理容器：

```text
fault39-qwen3-32b-cann851-float-matmul-cancel-v18
```

最终状态：

```text
health       200
Running      true
OOMKilled    false
RestartCount 0
control      version 13, mode 18, enabled 0
```

后续 output proxy 验证启动 v19 时，v18 已正常停止并保留；v17 也仍保留，未删除历史证据。当前运行状态见 output proxy 验证报告。

## 证据边界

本轮证明的是固定模型、固定 CANN/vLLM 版本、固定 decode projection、rank0 single-hit 下：

```text
有限输入 + 有限精确数学结果
  -> 符号排列改变
  -> 真实 aclnnMatmul 的 non-finite 行为发生改变
```

它强烈指向 reduction ordering、tile partial sum 或其他中间数值范围行为，但尚未证明具体硬件 accumulator 的 dtype、内部实现或溢出指令。后续可以将 direct output Inf/NaN proxy 作为独立的诊断工具覆盖能力；它不替代本轮根因近似证据。
