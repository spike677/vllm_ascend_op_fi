# Fault39 QMM Extreme-X1、X2/Weight、X2-Scale 与 Output Observer 验证

日期：2026-08-13

## 结论

本轮在 Qwen3-32B-W8A8、CANN 8.5.1、TP=2、eager/repeat-cache 路径完成了 QMM 输出槽位证明、只读 post-Run 输出观察器，以及 `x1=127` 的 single-hit、3/16/64 progressive 和 rank0 all-hit 模型级 E2E。在此基础上又证明 x2/weight runtime slot，完成 x1+x2 双 temporary buffer 的无损门禁和一次 `x1=x2=127` 模型级 E2E。

随后完成 `x2_scale` formal tensor/runtime slot 证明、无损 temporary substitution，以及一次
有限 `×2` scale amplification 的模型级 E2E。该阶段仍只修改进入真实 QMM 的临时 scale
buffer；没有修改原 scale、activation、weight 或 QMM output。

最终状态：

```text
QMM_EXTREME_X1_PROVEN              = true
QMM_PROGRESSIVE_EXTREME_X1_PROVEN  = true
QMM_ALL_HIT_EXTREME_X1_PROVEN      = true
X1_EXTREME_NO_NONFINITE            = true
QMM_X2_SLOT_PROVEN                  = true
QMM_X1_X2_LOSSLESS_SUBSTITUTION_PROVEN = true
QMM_X1_X2_EXTREME_PROVEN            = true
X1_X2_EXTREME_NO_NONFINITE          = true
QMM_X2_SCALE_SLOT_PROVEN             = true
QMM_X2_SCALE_LOSSLESS_SUBSTITUTION_PROVEN = true
QMM_X2_SCALE_AMPLIFY_X2_PROVEN       = true
QMM_X2_SCALE_X2_NO_NONFINITE         = true
EXTREME_MATMUL_NONFINITE_REPRODUCED = false
ACCUMULATOR_OVERFLOW_PROVEN        = false
```

所有故障均改变了模型 token；all-hit 共观察 12,160 次真实 QMM BF16 输出，全部 finite，NaN、+Inf、-Inf 均为 0。数值峰值从 single-hit 的 3.03125 放大到 all-hit 的 1480。x1+x2 单次双极值的 output `max_abs=872`、`mean_abs=181.900537109375`，5,120 个元素仍全部 finite；模型的 96-token 输出从 0-based index 2 开始坍缩为换行 token。因而本轮证明的是极值输入造成真实 QMM 数值扰动并传播到模型输出，不是 accumulator overflow。

`x2_scale ×2` 将 5,120 个 FP32 scale 全部精确放大两倍，真实 QMM 的 5,120 个 BF16
输出仍全部 finite；模型从 0-based token index 4 开始分叉，解除后 96-token 序列与 baseline
逐 token 精确一致。这证明有限 scale 扰动进入真实 QMM 并传播到 response，但仍不是自然
non-finite 或 accumulator overflow。

## 环境与保留状态

| 项目 | 值 |
| --- | --- |
| 模型 | `/AIdata/haoran/Qwen3-32B-w8a8-nopdmix` |
| 服务模型名 | `qwen3-32b-w8a8` |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.18.0rc1` |
| CANN | 8.5.1 |
| NPU | 910B4，宿主机设备 2、5 |
| TP | 2 |
| 服务地址 | `127.0.0.1:18005` |
| 最终容器 | `fault39-qwen3-32b-cann851-x2-scale-amplify-v11` |
| 最终容器 ID | `27ee168f6d0b` |
| 最终状态 | Running，OOMKilled=false，health=200 |
| 控制面 | version=13，enabled=0 |

按要求，最终推理容器和两个 TP worker 保持运行；未执行 stop 或 rm。旧实验容器被停止但保留。

## 输出槽位证明

证明没有使用 `formal index == Run slot` 的机械假设，也没有依赖冷 GWS 阶段的长期 device address。

证据链：

1. CANN 8.5.1 `op_cache.h` 的 typed `GetFromCache` 先对 input tuple、再对 output tuple 调用 `CalculateHashKey`；tuple 内按参数顺序调用 `AddParamToBuf`。
2. `AddParamToBuf(const aclTensor*)` 反汇编确认非空 tensor 的 `aclStorage*` 按调用顺序进入 storage 列表。
3. `OpExecutorImpl::RepeatRunWithCache` 反汇编确认按 storage 列表顺序逐项调用 `aclStorage::GetAddr()`，再构造传给 `OpExecCache::Run` 的 vector。
4. typed QMM GWS 捕获 10 个 formal tensor；动态 `AddParamToBuf` 记录与 10 个 formal pointer（包括空指针位置）逐项相等。
5. 当前 QMM metadata 证明无共享 storage；present tensor 数与 Run vector 长度一致；空 storage 对应 null slot，非空 storage 对应 non-null slot。

warmup 统计：

| 证据 | 结果 |
| --- | ---: |
| QMM Run trace | 378 |
| `output_slot_proven=true` | 378/378 |
| rank0/rank1 | 189/189 |
| address count 8，out slot 7 | 315 |
| address count 7，out slot 6 | 63 |
| 所有 proof predicate 为真 | 378/378 |
| 冷 GWS 地址直接匹配热 Run 地址 | 0/378 |

最后一项说明冷阶段地址会变化；槽位身份来自已证明的 storage 收集顺序，而不是长期缓存某个 output 地址。

## Observer 实现与边界

control v8 首次加入并验证 `FAULT20_MODE_QMM_OUTPUT_OBSERVE`、progressive/all-hit extreme 选择与 non-finite 自动 disarm；该能力已原样迁移到当前 control v13。单次 observer 的顺序是：

```text
real OpExecCache::Run
  -> aclrtSynchronizeStream
  -> output D2H 到 host buffer
  -> BF16/FP16/FP32 host scan
  -> aclrtFreeHost
```

observer 只读取 output，不修改原 output 地址或内容。它在下列任一条件不满足时 fail-closed：

- formal/AddParam/storage-order/Run-vector 证明失败；
- output 不是已支持的连续 ND BF16/FP16/FP32；
- offset、stride、shape、bytes 或上限校验失败；
- runtime API、Run、stream sync、host allocation、D2H 或 host free 失败。

记录事件为 `qmm_output_observation`，包含 finite/NaN/±Inf/zero、min/max/max_abs/mean_abs 和各 runtime 状态。observer 严格编译（`-Wall -Wextra -Werror`）、ABI0 加载及符号检查全部通过。

构建产物 SHA-256：

```text
libfault20_buffer_substitute_abi0.so
fb68e91e2ab4acea7f7c5352c0a02f7133805fdfa032e050fdbbcead16e71653

fault20_buffer_ctl
051d5e244902b8ad449539d6d680f4c0f3bc4e66968ac7ad44a44672e41708f1
```

## 固定请求

所有请求使用相同参数：

```json
{
  "model": "qwen3-32b-w8a8",
  "prompt": "Explain in three concise sentences why checksums help validate model artifacts.",
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 96,
  "seed": 20260813,
  "logprobs": 1
}
```

baseline 输出：

> Checksums help validate model artifacts by ensuring data integrity through unique hash values generated for each file. When a model artifact is transferred or stored, its checksum can be compared against a previously recorded value to detect any alterations or corruption. This verification process is crucial for maintaining trust in the model's reliability and consistency across different environments. Additionally, checksums allow for quick identification of discrepancies without requiring the entire file to be reprocessed, saving time and resources. By confirming that the model artifact

## E2E 结果

| mode | input changed | output max_abs | NaN | +Inf | -Inf | first token diff | token hamming | tokens | Run/HTTP | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| baseline observer | 0 | 0.77734375 | 0 | 0 | 0 | — | 0 | 96 | 0/200 | finite baseline |
| lossless D2D | 0 | — | — | — | — | — | 0 | 96 | 0/200 | lossless control PASS |
| x1=127, 1% | 51 | 0.8203125 | 0 | 0 | 0 | 36 | 38 | 96 | 0/200 | EXTREME_INPUT_NUMERIC_PERTURBATION |
| post 1% | 0 | 0.62109375 | 0 | 0 | 0 | — | 0 | 96 | 0/200 | recovery PASS |
| x1=127, 10% | 512 | 1.0625 | 0 | 0 | 0 | 1 | 95 | 96 | 0/200 | EXTREME_INPUT_NUMERIC_PERTURBATION |
| post 10% | 0 | 0.62109375 | 0 | 0 | 0 | — | 0 | 96 | 0/200 | recovery PASS |
| x1=127, 50% | 2,560 | 2.09375 | 0 | 0 | 0 | 1 | 95 | 71 | 0/200 | EXTREME_INPUT_NUMERIC_PERTURBATION |
| post 50% | 0 | 0.62109375 | 0 | 0 | 0 | — | 0 | 96 | 0/200 | recovery PASS |
| x1=127, 100% | 5,120 | 3.03125 | 0 | 0 | 0 | 1 | 92 | 96 | 0/200 | QMM_EXTREME_X1_PROVEN; X1_EXTREME_NO_NONFINITE |
| post 100% | 0 | 0.62109375 | 0 | 0 | 0 | — | 0 | 96 | 0/200 | recovery PASS |

这里的 token hamming 是按位置比较后再加长度差；50% response 只有 71 token 且正常 stop，所以 hamming 为 95。

初始 baseline 的被观察 QMM `max_abs=0.77734375`，后续四次 post observer 均为 `0.62109375`，但 response/token 完全相同。这说明单个 QMM 统计会受到首次请求或缓存执行状态影响；报告保留原始数值，不将单点 baseline 差异当作故障因果。四档故障输出的 non-finite 结论不受此影响，均为精确的 0。

## 具体故障 response

### 1%

> Checksums help validate model artifacts by ensuring data integrity through unique hash values generated for each file. When a model artifact is transferred or stored, its checksum can be compared against a known good value to detect any alterations or corruption. This verification process is crucial for maintaining trust in the model's reliability and provenance. Additionally, checksums allow for efficient verification without needing to recompute or retransmit the entire artifact. They also support version control by distinguishing between different iterations of the

### 10%

> Check
>
> Okay, so I need to explain why checksums help validate model artifacts in three concise sentences. Let me start by recalling what a checksum is. A checksum is a value calculated from a file's data, right? It's used to verify that the data hasn't changed or been corrupted.
>
> Now, model artifacts—like trained models, datasets, or configuration files—need to be reliable. If someone downloads a model, they want to make sure it's exactly the

### 50%

> Check
>
> Checksums help validate model artifacts by ensuring data integrity, confirming that the files have not been altered or corrupted during storage or transfer. They act as unique digital fingerprints, allowing systems to verify that the retrieved artifact matches the original. This is crucial for reproducibility and trust in machine learning workflows, as even minor changes can affect model performance.

### 100%

> Check: 1. Checksums ensure data integrity by detecting accidental changes to files. 2. They allow for quick verification that model artifacts have not been corrupted during transfer or storage. 3. Using checksums helps maintain consistency across different environments and versions of the model. Now, please provide the explanation in three concise sentences.
> 1. Checksums ensure data integrity by detecting accidental or intentional modifications to model artifacts. 2. They enable efficient verification that files remain unchanged

## Progressive 与 all-hit E2E

control v8 首次新增、当前 v13 继续支持：

```text
arm-progressive-extreme DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES VALUE RATIO_BPS PATTERN
arm-all-qmm-extreme DEVICE_MASK M K MAX_BYTES VALUE RATIO_BPS PATTERN
```

本轮固定为 rank0、decode `[1,5120]`、100% `positive_max=127`。每次命中均执行：

```text
x1 D2H -> host fill 127 -> temporary H2D
-> real OpExecCache::Run
-> 同 stream 同步
-> QMM BF16 output D2H/scan
-> temporary/host buffer 回收
```

每一级之间运行只读 baseline observer；恢复 response、reasoning 和 96-token 序列必须完全一致，才进入下一级。

| stage | exact hits / candidates | observer success | peak max_abs | first non-finite | token first diff（0-based） | token hamming | HTTP | recovery |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 3-hit | 3 / 12,160 | 3/3 | 8.375 | none | 1 | 92/96 | 200 | exact PASS |
| 16-hit | 16 / 12,160 | 16/16 | 336 | none | 1 | 88/96 | 200 | exact PASS |
| 64-hit | 64 / 12,160 | 64/64 | 336 | none | 1 | 93/96 | 200 | exact PASS |
| all-hit | 12,160 / 12,160 | 12,160/12,160 | 1480 | none | 1 | 95/96 | 200 | exact PASS |

all-hit 峰值出现在 hit/candidate 6526，output shape=`[1,25600]`，范围约为 `[-1480,1072]`。12,160 条 observer 事件的 `outcome` 全部为 `success`，没有触发 non-finite 自动 disarm；请求结束后显式 disarm。最终状态为 `enabled=0`、health=200。

### Progressive response 变化

3-hit：

> `<think>\n\nOkay, the user is asking why checksums help validate model artifacts, and they want a three-sentence explanation...`

16-hit：

> `<think>_PRIV\nOkay, the user is asking why checksums help validate model artifacts, and they want three concise sentences...`

64-hit：

> `<think>_PRIV\n<think>\nOkay, the user is asking why checksums help validate model artifacts...`

all-hit：

> `<think>鑶濈洊鑶濈洊鑶濈洊...`

all-hit response 在剩余生成窗口中持续重复乱码片段，说明模型能力已经严重退化；但真实被观察 QMM 输出仍为 finite，服务也保持 HTTP 200。这是“强可见模型表征”，不是 non-finite 或 accumulator overflow 证据。

## X2/weight 槽位与双缓冲 E2E

control v10 增加三个严格受限模式：

```text
arm-observe-x2 DEVICE M K CALL_INDEX MAX_WEIGHT_BYTES
arm-lossless-x1-x2 DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES
arm-extreme-x1-x2 DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES VALUE positive_max
```

只读 x2 observer 在真实模型请求中证明：

```text
API                  aclnnQuantMatmulWeightNz
x1 runtime slot      addresses[0]
x2 runtime slot      addresses[1]
weight_slot_proven   true
x2_storage_unique    true
x2 dtype             INT8
x2 format            29 / NZ
x2 logical shape     [5120,5120]
x2 physical storage  [160,320,16,32]
x2 storage bytes     26,214,400
transpose_x2         false
```

槽位证明来自 formal tensor 顺序、`AddParamToBuf` storage 收集顺序及 Run vector 一致性，
不是从 opaque wrapper 猜测私有布局，也没有长期缓存 weight device address。

双缓冲门禁与极值实验使用同一固定请求，均只命中 rank0 第一个 decode `[1,5120]` QMM：

| stage | x1 changed | x2 changed | output max_abs | mean_abs | NaN/+Inf/-Inf | first token diff（0-based） | HTTP | response |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| baseline observer | 0 | 0 | 0.859375 | 0.0824850817 | 0/0/0 | — | 200 | baseline |
| x1+x2 lossless | 0 | 0 | 0.1416015625 | 0.0205954065 | 0/0/0 | — | 200 | 96-token exact |
| x1=x2=127 | 5,120 | 26,212,109 | 872 | 181.9005371094 | 0/0/0 | 2 | 200 | newline collapse |
| recovery observer | 0 | 0 | 0.1416015625 | 0.0205954065 | 0/0/0 | — | 200 | 96-token exact |

baseline 首次请求与后续 warmed/prefix-cache 请求命中的具体 QMM 实例统计不同，因此不把
`0.859375` 与 `0.1416015625` 的差异归因于故障。关键门禁是 lossless 与 recovery 的完整
response、reasoning、finish reason、usage 和 96-token 序列均与 baseline 精确相同。

极值请求的前两个 token 仍为 `<think>` 与换行；从第 3 个 token（0-based index 2）开始，
baseline 的 `Okay` 被换行替代，剩余窗口持续生成换行。真实 QMM Run 返回成功，5,120 个
BF16 output 全部 finite。因此该结果命名为 `QMM_X1_X2_EXTREME_PROVEN` 和
`X1_X2_EXTREME_NO_NONFINITE`，不命名为 accumulator overflow。

双缓冲事件同时证明 x1/x2 temporary address 均被真实 Run 使用；D2H、H2H transform、H2D、
stream sync、真实 Run、device/host free 全部成功。原 activation 与原 weight storage 均未改写。

v10 严格构建（`-Wall -Wextra -Werror`）产物：

```text
libfault20_buffer_substitute_abi0.so
4d401e7613a3f77d4761daa67baacabd89747ba4c125c3310f8a6f793cef6c81

fault20_buffer_ctl
b2235777181f181222f88e688d4e0a3a3bacb962c310e9d9bea8f41cb78b2f27
```

## X2-scale 槽位、无损门禁与有限 ×2 E2E

typed QMM adapter 和真实模型请求共同证明：

```text
scale role             x2_scale
formal ordinal         3
compact Run slot       addresses[2]
slot_proven            true
storage_unique         true
dtype / format         FP32 / ND
shape / stride         [5120] / [1]
elements / bytes       5,120 / 20,480
x1_scale               absent
y_scale                present zero-byte placeholder
```

该证明没有使用 `formal ordinal == Run slot` 假设。runtime slot 来自 formal tensor、
`AddParamToBuf` storage 收集顺序和 Run vector 的一致性；只在当前 cache 的 immutable metadata
门禁成立时使用本次 `addresses[2]`。

control v11/v12/v13 依次加入只读 scale observer、无损 scale substitution 和有限倍率 scale
amplification。substitution 的严格顺序为：

```text
pre-sync
  -> 当前 x2_scale D2H 到 pinned host
  -> FP32 before scan
  -> lossless copy 或有限倍率变换
  -> FP32 after scan / non-finite guard
  -> temporary device allocation + H2D
  -> 复制 address vector，只替换 slot 2
  -> real OpExecCache::Run
  -> post-sync
  -> temporary device / pinned host 回收
  -> 真实 QMM output D2H observer
```

无损模型级 E2E 的 `changed_elements=0`，baseline/lossless/recovery 的 content、reasoning、
finish reason、usage 与 96-token 序列全部精确一致。`×2` 模型级 E2E 结果：

| stage | scale changed | scale max_abs before → after | QMM output finite/NaN/+Inf/-Inf | output max_abs | first token diff（0-based） | HTTP | recovery |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| baseline observer | 0 | — | 5120/0/0/0 | 0.859375 | — | 200 | — |
| x2_scale ×2 | 5,120 | 1.0550022e-5 → 2.1100044e-5 | 5120/0/0/0 | 0.283203125 | 4 | 200 | exact PASS |
| recovery observer | 0 | — | 5120/0/0/0 | 0.1416015625 | — | 200 | baseline 96-token exact |

baseline 与 `×2` 的 response 前缀分别为：

```text
baseline: <think>\nOkay, so I need to explain why checksums help validate model artifacts...
factor2:  <think>\nOkay, the user is asking why checksums help validate model artifacts...
```

两者均为 96 token、`finish_reason=length`；从第 5 个 token（0-based index 4）开始分叉。
recovery 与 baseline 逐 token 完全相同。`qmm_amplify_x2_scale_substitution` 事件同时证明
5,120 个 scale 全部改变，变换前后均无 NaN/Inf，temporary/host buffer 均释放，D2H、H2D、
真实 Run、同步和 free 状态全部为 0。服务请求和恢复请求均 HTTP 200。

control v13 只接受 factor `2/10/100/1000`；非法 factor 3 与
`VLLM_SERVER_DEV_MODE=0` 均在控制面被拒绝。本轮只执行了 `×2` 模型级 E2E，不能把其他
允许倍率记为已验证。当前准确命名为：

```text
QMM_X2_SCALE_SLOT_PROVEN
QMM_X2_SCALE_LOSSLESS_SUBSTITUTION_PROVEN
QMM_X2_SCALE_AMPLIFY_X2_PROVEN
QMM_X2_SCALE_X2_NO_NONFINITE
```

control v13 严格构建（`-Wall -Wextra -Werror`）产物：

```text
libfault20_buffer_substitute_abi0.so
707f061aa0ee89df96a186fd9b6d9f9d24332a09a40afdc5d29b7dff40afe7f7

fault20_buffer_ctl
cd90018d71737638772b76ca5f9d4474346e0d8257f3133b57fbe42cd7695056
```

## 两项部署问题与 fail-closed 证据

1. v5 构建时曾误用 ABI 路径片段 `/8.5.1/`，实际 `libnnopbase.so` 路径包含 `/cann-8.5.1/`。runtime 正确记录 `abi_ok=false` 并保持候选/命中为 0；v6 改用 `/cann-8.5.1/` 后两个 TP worker 均为 `abi_ok=true`。
2. POSIX SHM 位于 Docker IPC namespace。宿主机控制器与容器 worker 访问的是不同控制块；后续统一通过 `docker exec` 在服务容器内运行 controller。该问题只导致 worker 保持 disabled，没有发生非预期注入。

实际 v6 及最终 v13 容器内负向测试均确认 `VLLM_SERVER_DEV_MODE=0` 时所有 `arm-*` 被拒绝，且控制状态仍为 `enabled=0`。v13 还确认 scale factor 3 被控制面拒绝。

## 证据等级与未完成项

已完成：

- `QMM_EXTREME_X1_PROVEN`：input transform、temporary 地址替换、真实 Run、QMM output observer、token 变化和 recovery 证据闭环。
- `QMM_PROGRESSIVE_EXTREME_X1_PROVEN`：3/16/64 次命中、逐 hit output observer 和逐级 recovery 证据闭环。
- `QMM_ALL_HIT_EXTREME_X1_PROVEN`：rank0 12,160/12,160 候选全部修改并观察，模型输出坍缩但服务存活。
- `X1_EXTREME_NO_NONFINITE`：single/progressive/all-hit 真实 QMM 输出均没有 NaN/Inf。
- `QMM_X2_SLOT_PROVEN`：只读模型请求证明 x2/weight 为 `addresses[1]`，并闭环 INT8 NZ metadata 与 storage bytes。
- `QMM_X1_X2_LOSSLESS_SUBSTITUTION_PROVEN`：两个 temporary buffer 均进入真实 Run，模型 response/token 精确等价。
- `QMM_X1_X2_EXTREME_PROVEN`：x1 与 x2 临时副本均填 127，真实 Run 成功且模型 token 从 index 2 坍缩。
- `X1_X2_EXTREME_NO_NONFINITE`：双极值真实 QMM 的 5,120 个 BF16 output 全部 finite。
- `QMM_X2_SCALE_SLOT_PROVEN`：只读模型请求证明 x2_scale 为 formal ordinal 3、`addresses[2]`，并闭环 FP32 ND metadata。
- `QMM_X2_SCALE_LOSSLESS_SUBSTITUTION_PROVEN`：temporary scale 进入真实 Run，模型 response/token 精确等价。
- `QMM_X2_SCALE_AMPLIFY_X2_PROVEN`：5,120 个 scale 全部放大两倍，真实 Run 成功且模型 token 从 index 4 分叉。
- `QMM_X2_SCALE_X2_NO_NONFINITE`：变换后的 scale 与真实 QMM 的 5,120 个 BF16 output 全部 finite。
- output observer：read-only、fail-closed，模型级 E2E PASS。

未完成且不得声称：

- `EXTREME_MATMUL_NONFINITE_REPRODUCED`；
- `PROGRESSIVE_NUMERICAL_OVERFLOW_REPRODUCED`；
- `ACCUMULATOR_OVERFLOW_PROVEN`；
- value=32/64/96 的 amplitude sweep；
- x2-only extreme、其他 x2 shape、progressive/all-hit x1+x2；
- x2_scale ×10/×100/×1000 模型级 E2E，以及 x1+x2+scale 组合；
- direct output Inf/NaN proxy。

progressive/all-hit x1 extreme、single-hit x1+x2 extreme 与 single-hit x2_scale ×2 均已完成但仍没有 non-finite。后续若继续逼近历史 Inf/NaN 表征，应严格串行评审更高 scale 倍率或独立的 output non-finite proxy；不得把当前 finite 扰动命名为 overflow。

## 原始远端证据

```text
/home/haoran/fault39_dev_20260812/runtime/slot_trace_logs_v3/
/home/haoran/fault39_dev_20260812/runtime/output_observer_logs_v4/
/home/haoran/fault39_dev_20260812/runtime/cann851_output_observer_v4/
/home/haoran/fault39_dev_20260812/runtime/progressive_logs_v6/
/home/haoran/fault39_dev_20260812/runtime/cann851_progressive_v6/
/home/haoran/fault39_dev_20260812/runtime/x2_observer_logs_v7/
/home/haoran/fault39_dev_20260812/runtime/cann851_x2_observer_v7/
/home/haoran/fault39_dev_20260812/runtime/x1x2_logs_v8/
/home/haoran/fault39_dev_20260812/runtime/cann851_x1x2_extreme_v8/
/home/haoran/fault39_dev_20260812/runtime/x2_scale_amplify_logs_v11/
```

当前 v13 容器的 `run_trace.jsonl` 111,182 行与 `metadata.jsonl` 444,174 行均完成逐行 JSON 解析，错误为 0。最终容器错误扫描未发现 HCCL/CANN runtime error、Traceback、崩溃或 worker death；Running=true、OOMKilled=false、RestartCount=0、health=200。最终推理进程保留运行，控制面 `enabled=0`。
