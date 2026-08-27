# Fault #15：Qwen3-32B W8A8 scale 粒度只读探查

日期：2026-08-11

## 结论

本次真实模型探查不支持“当前 Qwen3-32B W8A8 路径在运行时从 batch activation 统计 scale，并因跨 sequence 聚合而污染”的假设。

当前模型实际链路是：

```text
checkpoint 中每个 Linear 的静态 input_scale/input_offset（shape=[1]）
  -> 模型加载时 repeat(hidden_size)
  -> input_scale 取 reciprocal
  -> torch_npu.npu_quantize(..., axis=-1)
  -> aclnnAscendQuantV3(x=[M,H], scale=[H], offset=[H], axis=-1)
```

因此 ACLNN 层看到的 `[H]` scale 虽然结构上沿 hidden axis 广播，但每个向量内部所有元素相同，语义上是“每层一个静态 per-tensor scalar 的展开”，不是 runtime per-channel 统计，更不是 per-token 动态 scale。

## 环境

```text
model       /home/haoran/models/Qwen3-32B-w8a8-nopdmix
image       quay.io/ascend/vllm-ascend:v0.12.0rc1
vLLM        0.12.0
CANN        8.3 系列镜像运行时
host driver 25.5.0.b060
execution   TP=2, eager, max_num_seqs=8
container   fault15-qwen3-scale-trace
endpoint    127.0.0.1:18015
```

当前宿主驱动下，普通 device-cgroup 容器无法枚举 NPU；按本机现有可用容器底座使用 privileged 后，两张实际设备 `/dev/davinci2`、`/dev/davinci5` 正常可见。

## M=1/2/4/8 结果

四组请求均返回 HTTP 200，choice 数分别为 1、2、4、8。decode 阶段精确出现以下形状：

| M | x shape | scale shape | offset shape | axis | round | dst dtype |
|---:|---|---|---|---:|---|---:|
| 1 | `[1,5120]` / `[1,4096]` | `[5120]` / `[4096]` | 同 scale | -1 | `round` | 2（INT8） |
| 2 | `[2,5120]` / `[2,4096]` | `[5120]` / `[4096]` | 同 scale | -1 | `round` | 2（INT8） |
| 4 | `[4,5120]` / `[4,4096]` | `[5120]` / `[4096]` | 同 scale | -1 | `round` | 2（INT8） |
| 8 | `[8,5120]` / `[8,4096]` | `[5120]` / `[4096]` | 同 scale | -1 | `round` | 2（INT8） |

所有 scale/offset D2H 都成功，未发现 Inf/NaN。代表性 cache-create snapshot：

| 对应静态参数 | 展开后数值 | offset | FNV-1a hash |
|---|---:|---:|---|
| layer 0 q/k/v 输入 scale reciprocal，H=5120 | 322（5120 项全部相同） | -18 | `edb76b0bc9f57383` |
| layer 0 o_proj 输入 scale reciprocal，H=4096 | 199（4096 项全部相同） | -8 | `e5c349aaf26ba383` |
| layer 1 o_proj 输入 scale reciprocal，H=4096 | 254（4096 项全部相同） | -18 | `455bbd03d0db6383` |

第三行说明不同 layer 的静态 scale 可以不同；它不是 M=4 导致 scale 变化。

## Producer 证据

vLLM Worker Extension 对两个 rank 的 layer 0 返回一致结果：

- `input_scale`：`parameter`，BF16，shape `[1]`；
- `input_offset`：`parameter`，shape `[1]`；
- `aclnn_input_scale_reciprocal`：模型加载后生成的 `tensor_attr`，shape `[H]`；
- qkv/o_proj/gate_up_proj 使用 `AscendLinearMethod`；down_proj 当前为 unquantized。

checkpoint 直接读取结果：

```text
input_scale  数量 384，全部 shape=[1]，161 个不同标量
input_offset 数量 384，全部 shape=[1]， 91 个不同标量
scale 范围  0.0030975341796875 .. 1.1796875
offset 范围 -101 .. 59
```

源码闭环位于镜像内 `vllm_ascend/quantization/w8a8.py`：

1. `get_pertensor_param()` 创建 `[1]` 的 `input_scale/input_offset`；
2. `process_weights_after_loading()` 对 scalar 执行 `repeat(expanding_factor)` 并计算 reciprocal；
3. `quant_per_tensor()` 调用 `torch_npu.npu_quantize(..., axis=-1)`。

## 证据边界

metadata proxy 捕获的是 GWS/cache-create 阶段的正式 `aclTensor` metadata 和小型 scale/offset snapshot。executor cache 后续 Run 的动态 address vector 会变化，因此不能把某个 GWS raw address长期解释为所有 layer 的实际 runtime 地址，也不能只按 H 将不同 layer 的 scale 混为一类。本报告对 producer 的判断依靠 Worker Extension、checkpoint 和 vLLM-Ascend 源码三条独立证据，而不是仅靠地址复用。

## 对历史 #15 的判断

“scale 统计域错误导致跨 sequence 污染”仍然是合理的通用故障模型，但不能用当前 Qwen3 静态 W8A8 路径作为其真实根因证据。当前路径根本没有从本次 `x` 计算 min/max 的 runtime scale producer。

如果继续复现 #15，建议按优先级选择：

1. 找到真正使用 dynamic activation quantization 的模型/算子路径，再检查其 reduce 维与 batch reshape；
2. 若只要求模拟传播效果，可在明确标注“synthetic”的前提下做 wrong-scale/batch-shared-scale 实验；
3. 不应把修改当前静态 `input_scale` 描述成“复现了跨 batch 统计 bug”，它只能证明错误 scale 被 QuantV3 消费后的模型退化。

## 远端证据位置

```text
/home/haoran/fault15_scale_trace/logs/metadata_startup.jsonl
/home/haoran/fault15_scale_trace/logs/metadata_after_batches.jsonl
/home/haoran/fault15_scale_trace/logs/batch_analysis.json
/home/haoran/fault15_scale_trace/logs/layer0_modules.json
/home/haoran/fault15_scale_trace/logs/checkpoint_scale_inventory.json
/home/haoran/fault15_scale_trace/responses/20260811T100356Z/
```
