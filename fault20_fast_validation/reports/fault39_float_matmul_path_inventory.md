# Fault39 Floating MatMul 路径清点

日期：2026-08-13

## 结论

在 Qwen3-32B-W8A8、CANN 8.5.1、vLLM 0.18.0、TP=2、eager/repeat-cache
在线推理路径中，已经真实观察到 BF16 floating MatMul：

```text
MODEL_FLOAT_MATMUL_OBSERVED = true
FLOAT_MATMUL_PRIMARY_API    = aclnnMatmul
MODEL_FLOAT_MATMUL_NOT_OBSERVED = false
```

`aclnnMatmulWeightNz` 在本次 workload 中仍为 0。实际可继续适配的首选 API 是
`aclnnMatmul`，而不是已有 Fault20 adapter 的 `aclnnMatmulWeightNz`。

本轮只完成 API inventory 与 typed GWS metadata 采集，没有修改任何 tensor、address vector
或 output，也没有执行 lossless、finite extreme、scale amplification 或 Inf/NaN proxy。

当前状态：

```text
self_slot_proven = false
mat2_slot_proven = false
out_slot_proven  = false
```

当前不得声称：

```text
MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED
ACCUMULATOR_OVERFLOW_PROVEN
```

## 环境与门禁

| 项目 | 值 |
| --- | --- |
| 模型 | `/AIdata/haoran/Qwen3-32B-w8a8-nopdmix` |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.18.0rc1` |
| CANN | 8.5.1 |
| vLLM | 0.18.0 |
| torch_npu | 2.9.0.post1 |
| NPU | 910B4，宿主机设备 2、5 |
| 执行 | TP=2、eager、repeat executor cache |
| 服务 | `127.0.0.1:18005` |
| 开发模式 | `VLLM_SERVER_DEV_MODE=1` |
| 控制面 | v13，整个请求前后 `enabled=0` |

被动 API inventory 还要求显式设置：

```bash
FAULT20_API_INVENTORY=1
```

变量缺失时不记录全量 API identity。该开关只在 ABI 与开发模式校验同时通过时记录：

```text
api_info
cache pointer
device
address_count
```

它不执行分配、拷贝、同步、替换或数值变换。

## 采集器与加载证明

executor API inventory runtime：

```text
libfault20_buffer_substitute_abi0.so
SHA-256 4e0d188d73160a686245eee3d5209d0ab2c26c87fcd231463edf18999d7f0f40
```

typed `aclnnMatmul` metadata proxy：

```text
libopapi.so
SHA-256 9316be81510dbfddafd35823426a929c8dc8ac9c6edd6953dddcd7478b3fe81c
```

两个 worker 均记录：

```text
abi_ok=true
developer_mode_ok=true
run=true
add=true
api_info=true
runtime_api=true
library=/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libnnopbase.so
```

最终有效容器中，两个 worker 的 `/proc/<pid>/maps` 均证明实际加载：

```text
/fault39/runtime/cann851_float_matmul_metadata_v13/libopapi.so
/fault39/runtime/cann851_float_matmul_metadata_v13/real/libopapi_real.so
```

一次更早的启动曾因 `set_env.sh` 重排 `LD_LIBRARY_PATH` 而加载 CANN 原生
`libopapi.so`；该容器没有作为 typed 证据使用，随后已重建并通过上述映射检查。

## API inventory

固定请求：

```text
prompt      Explain in three concise sentences why checksums help validate model artifacts.
temperature 0
top_p       1
max_tokens  96
seed        20260813
logprobs    1
```

### Executor Run 计数

| API | warmup Run | 请求增量 | 总计 | device 分布 | address count | 总 unique cache |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `aclnnMatmul` | 126 | 12,474 | 12,600 | 0/1 各 6,300 | 全部 3 | 8 |
| `aclnnBatchMatMul` | 4 | 0 | 4 | 0/1 各 2 | 全部 3 | 4 |
| `aclnnMatmulWeightNz` | 0 | 0 | 0 | — | — | 0 |
| `aclnnBatchMatMulWeightNz` | 0 | 0 | 0 | — | — | 0 |
| `aclnnFusedMatmul` | 0 | 0 | 0 | — | — | 0 |
| `aclnnQuantMatmulWeightNz` | 378 | 36,852 | 37,230 | 0/1 各 18,615 | 7 或 8 | 18 |

最后一行是 INT8 QMM 参考，不属于 floating MatMul。

### Typed GWS 与 Execute 计数

| 事件 | warmup | 请求增量 | 总计 | status |
| --- | ---: | ---: | ---: | --- |
| `aclnnMatmulGetWorkspaceSize` | 4 | 6 | 10 | 全部 0 |
| `matmul_gws_summary` | 4 | 6 | 10 | metadata 全部有效 |
| `aclnnMatmul` execute enter | 130 | 12,480 | 12,610 | — |
| `aclnnMatmul` execute exit | 130 | 12,480 | 12,610 | 全部 0 |
| `aclnnMatmulWeightNzGetWorkspaceSize` | 0 | 0 | 0 | — |
| `aclnnMatmulWeightNz` execute | 0 | 0 | 0 | — |

public Execute wrapper 计数与 `GetLogApiInfo()` 标识的 executor Run 计数属于不同边界，
不能机械地按数值一一对应。API 身份与 cache 统计以 `OpExecCache::Run` inventory 为准；
formal tensor metadata 以 typed GWS 为准。

## `aclnnMatmul` 正式 Tensor metadata

10 次 GWS 形成 5 组 shape，每组在两个 TP worker 各出现一次。所有 tensor 都满足：

```text
dtype          27 / BF16
format         2 / ND
view_rank      2
stride_rank    2
view_offset    0
element_size   2
cube_math_type 0
all_present    true
empty storage  false
shared storage false
```

storage 在当前 `aclTensor` metadata 中表现为一维 physical storage，故
`storage_rank=1`；正式二维语义来自 view shape 与 stride。

| 次数 | self shape / stride / bytes | mat2 shape / stride / bytes | out shape / stride / bytes | 阶段解释 |
| ---: | --- | --- | --- | --- |
| 2 | `[2048,12800]` / `[12800,1]` / 52,428,800 | `[12800,5120]` / `[1,12800]` / 131,072,000 | `[2048,5120]` / `[5120,1]` / 20,971,520 | warmup/profile |
| 2 | `[2,5120]` / `[5120,1]` / 20,480 | `[5120,75968]` / `[1,5120]` / 777,912,320 | `[2,75968]` / `[75968,1]` / 303,872 | warmup logits |
| 2 | `[22,12800]` / `[12800,1]` / 563,200 | `[12800,5120]` / `[1,12800]` / 131,072,000 | `[22,5120]` / `[5120,1]` / 225,280 | 请求 prefill |
| 2 | `[1,5120]` / `[5120,1]` / 10,240 | `[5120,75968]` / `[1,5120]` / 777,912,320 | `[1,75968]` / `[75968,1]` / 151,936 | decode logits |
| 2 | `[1,12800]` / `[12800,1]` / 25,600 | `[12800,5120]` / `[1,12800]` / 131,072,000 | `[1,5120]` / `[5120,1]` / 10,240 | decode projection |

因此当前 workload 确实存在满足后续有限 BF16 大数实验前提的真实 floating MatMul，
其中 decode `[1,12800] × [12800,5120]` 是较小 self temporary buffer 的候选路径。
但 `mat2` 为转置式 stride `[1,K]` 且 storage 很大，后续设计必须以正式 metadata 为准，
不能按简单 contiguous `[K,N]` 假设处理。

## 模型级只读门禁

固定请求结果：

```text
HTTP              200
health            200
control enabled   0 -> 0
worker OOM        false
worker restart    0
```

与既有 x2_scale baseline 比较：

```text
content           exact
reasoning_content exact
finish_reason     exact
usage             exact
token sequence    96/96 exact
first token diff  none
```

证据日志逐行 JSON 解析：

```text
run_trace.jsonl  132,893 lines, 0 parse errors
metadata.jsonl   174,532 lines, 0 parse errors
```

未发现 HCCL/CANN runtime error、Traceback、segmentation fault、core dump、OOM 或
worker restart。

## 当前证据边界

已经证明：

```text
FLOAT_MATMUL_API_INVENTORY_PROVEN
ACLNN_MATMUL_MODEL_PATH_OBSERVED
ACLNN_MATMUL_BF16_ND_GWS_METADATA_PROVEN
ACLNN_MATMUL_READ_ONLY_E2E_PROVEN
```

尚未证明：

```text
self / mat2 / out 的 Run slot
formal tensor 到 AddParamToBuf storage order 的逐项映射
每个 GWS metadata 对应的具体 cache
lossless temporary substitution
output observer
finite BF16 amplitude sweep
自然 non-finite
```

3-address Run 不能直接推出：

```text
addresses[0] == self
addresses[1] == mat2
addresses[2] == out
```

也不能使用冷 GWS device address 与热 Run address 是否相等作为槽位证明。

## 下一阶段唯一建议

下一步只做 `aclnnMatmul` slot proof：

```text
typed aclnnMatmul GWS
  -> AddParamToBuf tensor/storage collection order
  -> AddOpExecCache cache pointer
  -> registry[cache] = immutable Matmul metadata
  -> 当前 OpExecCache::Run addresses vector
  -> self_slot_proven / mat2_slot_proven / out_slot_proven
```

必须先对目标 decode shape 完成上述三项证明，再依次进行 self、mat2、双输入 lossless。
双输入 lossless 与 output observer 均通过前，禁止执行 BF16 finite extreme。

## 远端证据

```text
/home/haoran/fault39_dev_20260812/runtime/float_api_inventory_logs_v12/
/home/haoran/fault39_dev_20260812/runtime/float_matmul_typed_inventory_logs_v13b/
/home/haoran/fault39_dev_20260812/runtime/cann851_float_api_inventory_v12/
/home/haoran/fault39_dev_20260812/runtime/cann851_float_matmul_metadata_v13/
```

采集完成后，identity-only 与 typed inventory 容器均正常停止并保留。原服务
`fault39-qwen3-32b-cann851-x2-scale-amplify-v11` 已恢复运行：

```text
health=200
Running=true
OOMKilled=false
RestartCount=0
control version=13
control enabled=0
```
