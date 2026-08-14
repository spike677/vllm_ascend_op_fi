# Fault39 `aclnnMatmul` 有限输入到 Non-finite 输出验证

日期：2026-08-14

## 结论

在本地部署的 Qwen3-32B-W8A8 dense 模型上，已经完成 `aclnnMatmul` 的 slot 证明、三项无损 temporary-buffer 门、BF16 输出 observer 和有限双输入幅值扫描。

本轮可以使用的准确结论是：

```text
MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED
```

不得把结果命名为：

```text
ACCUMULATOR_OVERFLOW_PROVEN
```

原因是本轮证明了有限 BF16 输入进入真实 `aclnnMatmul` 后产生 `+Inf`，但没有观测或定位 kernel 内部 accumulator。

## 验证边界

| 项目 | 值 |
| --- | --- |
| 模型 | `/AIdata/haoran/Qwen3-32B-w8a8-nopdmix` |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.18.0rc1` |
| CANN | 8.5.1 |
| NPU | 宿主机设备 2、5，容器内 device 0、1 |
| 执行 | TP=2、eager、repeat executor cache |
| 开发模式 | `VLLM_SERVER_DEV_MODE=1` |
| 目标 API | `aclnnMatmul` |
| 目标 rank | device/rank 0 |
| 目标调用 | arm 后 candidate 1，max hits 1 |
| self | BF16 ND `[1,12800]`，25,600 bytes |
| mat2 | BF16 ND `[12800,5120]`，stride `[1,12800]`，131,072,000 bytes |
| out | BF16 ND `[1,5120]`，10,240 bytes |

控制面默认关闭；每次实验只在服务健康和 baseline 完成后 arm，完成后立即 disarm。`2^58` 出现 non-finite 时 observer 在下游继续执行前自动 disarm。

## 1. Run slot 证明

真实链路为：

```text
typed aclnnMatmulGetWorkspaceSize
  -> AddParamToBuf formal tensor collection
  -> AddOpExecCache cache pointer
  -> registry[cache] = immutable metadata
  -> OpExecCache::Run current addresses
```

rank0、目标 shape、`run_index=1` 的证据：

```text
api_info                     aclnnMatmul_5028
tensor_param_count           3
formal_ordinals              self=0, mat2=1, out=2
storage_order_slots          self=0, mat2=1, out=2
address_count                3
compact_storage_count        3
param_order_exact            true
address_count_exact          true
all_present                  true
no_empty_storage             true
no_shared_storage            true
self_storage_unique          true
mat2_storage_unique          true
out_storage_unique           true
self_slot_proven             true
mat2_slot_proven             true
output_slot_proven           true
```

GWS 中记录的三个 device address 与本次 Run address 都不相等，直接匹配结果均为 `-1`。因此 slot 结论来自 formal collection order、cache 关联和当前 Run vector，而不是错误地复用冷态 GWS 地址。

## 2. 三项无损门

固定请求的 baseline、lossless 和 post-lossless recovery 都是 HTTP 200。三种模式的 choices+usage canonical SHA-256 都与 baseline 相同：

```text
eba6392fb5792354c789a38bd856c4af70192476f1cff80d3c0cc9a343f61e94
```

| 模式 | 临时 storage | 实际状态 | 模型响应 |
| --- | --- | --- | --- |
| self-only | 25,600 bytes | pre-sync/malloc/D2D/Run/post-sync/free 全 0 | exact |
| mat2-only | 131,072,000 bytes | pre-sync/malloc/D2D/Run/post-sync/free 全 0 | exact |
| dual-input | 两者同时 | 两组 malloc/D2D/free 与 Run/sync 全 0 | exact |

三种事件都满足：

```text
original address != temporary address
changed_elements = 0
temporary_used = true（仅对应被替换的 operand）
temporary_freed = true
```

mat2 复制的是正式 metadata 给出的物理 storage bytes；没有把逻辑转置 stride `[1,12800]` 当作普通 contiguous `[K,N]` 进行重排。

## 3. BF16 output observer

observer 的顺序是：

```text
real OpExecCache::Run
  -> synchronize stream
  -> D2H current out slot
  -> scan BF16 statistics
  -> free pinned host buffer
  -> wrapper returns
```

baseline observer 结果：

```text
output_slot_proven       true
finite_count             5120
nan_count                0
positive_inf_count       0
negative_inf_count       0
min                      -1.75
max / max_abs            7.59375
mean_abs                 0.11468038777820766
Run/sync/D2H/free        0
```

observer 请求与 recovery 响应均与 baseline 精确一致。

## 4. 有限双输入幅值扫描

每一档都执行：

```text
D2H original self and mat2
  -> host 填充同一个有限 BF16 2^e
  -> 统计实际 changed elements
  -> H2D 到两个 temporary device storage
  -> 替换 Run slot 0、1
  -> real aclnnMatmul Run
  -> BF16 output observer
```

每档都证明：

```text
self input finite after = 12,800
mat2 input finite after = 65,536,000
changed total           = 65,548,800
unchanged total         = 0
original != temporary
real Run status         = 0
copy/sync/free status   = 0
```

扫描结果：

| 输入值 | BF16 bits | 真实 out finite | `+Inf` | NaN | out 值/范围 | HTTP |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `2^48` | `0x5780` | 5,120 | 0 | 0 | `1.0141204801825835e33` | 200 |
| `2^56` | `0x5b80` | 5,120 | 0 | 0 | `6.6461399789245794e37` | 200 |
| `2^57` | `0x5c00` | 5,120 | 0 | 0 | `2.6584559915698317e38` | 200 |
| `2^58` | `0x5c80` | 0 | 5,120 | 0 | 全部 `+Inf` | 400 |
| `2^59` | — | — | — | — | 未执行：首个 non-finite 后停止 | — |

`2^58` 的输入仍全部是有限 BF16，真实 Run 返回成功，随后 observer 读取到 5,120/5,120 个 `+Inf`。下游请求返回：

```json
{"error":{"message":"Out of range float values are not JSON compliant","type":"BadRequestError","param":null,"code":400}}
```

这形成了完整因果链：

```text
finite BF16 self + finite BF16 mat2
  -> real aclnnMatmul
  -> BF16 out 全 +Inf
  -> observer auto-disarm
  -> 下游 HTTP 400
```

## 5. 模型响应与恢复

baseline 输出是正常 checksum 解释。`2^48`、`2^56`、`2^57` 虽然目标 MatMul 输出仍 finite，但模型响应已经稳定退化为不相关的 Markdown 图片链接片段；三档的 choices+usage canonical SHA-256 相同：

```text
e573afbff130c31a20037980d71f4e351e1884571a6d97285db9e7d817f41140
```

`2^58` 返回上述 HTTP 400。自动 disarm 后的最终 recovery 再次 HTTP 200，并与 baseline 的 choices+usage 哈希精确相同：

```text
eba6392fb5792354c789a38bd856c4af70192476f1cff80d3c0cc9a343f61e94
```

因此故障没有永久修改模型权重、activation 或 executor cache。

## 6. 构建产物与证据

| 阶段 | 产物 | SHA-256 |
| --- | --- | --- |
| slot proof runtime v14 | `cann851_float_matmul_slotproof_v14/libfault20_buffer_substitute_abi0.so` | `737db815b3aab836a05a008515dfba2f1e7e81929788a4d79b217bdb34c02727` |
| typed proxy v14 | `cann851_float_matmul_slotproxy_v14/libopapi.so` | `d2ce5d48948ab57bdc375dbb4420d49e056dcbf455ebf25eb43e125137665393` |
| lossless runtime v15 | `cann851_float_matmul_lossless_v15/libfault20_buffer_substitute_abi0.so` | `75c97832400a62a178751e6bc5e7f66b01f8206035f34793cb54c30ecd02a21b` |
| observer runtime v16 | `cann851_float_matmul_observer_v16/libfault20_buffer_substitute_abi0.so` | `05013620cbae8850ef2a373034483405098760ec56fa69defd239f1d5257e954` |
| finite sweep runtime v17 | `cann851_float_matmul_finite_v17/libfault20_buffer_substitute_abi0.so` | `fd51dd294ab5505b65b8c7a8c06ba72523e59aab70c3ddfa1d4de7f0bef662a2` |

远端日志：

```text
/home/haoran/fault39_dev_20260812/runtime/float_matmul_slotproof_logs_v14/
/home/haoran/fault39_dev_20260812/runtime/float_matmul_lossless_logs_v15/
/home/haoran/fault39_dev_20260812/runtime/float_matmul_observer_logs_v16/
/home/haoran/fault39_dev_20260812/runtime/float_matmul_finite_logs_v17/
```

v17 证据逐行 JSON 解析结果：

```text
run_trace.jsonl  332,421 lines, 0 parse errors
metadata.jsonl   1,553,860 lines, 0 parse errors
```

## 7. 该阶段完成时的运行状态

该阶段完成时保留运行的容器：

```text
fault39-qwen3-32b-cann851-float-matmul-finite-v17
```

最终状态：

```text
health       200
Running      true
OOMKilled    false
RestartCount 0
control      version 13, enabled 0
```

后续正负抵消验证启动 v18 时，v17 已正常停止并保留。早期 v14/v15/v16 和原 x2-scale baseline 容器也均保留为 stopped，不删除任何历史证据。当前运行状态见正负抵消验证报告。

## 证据边界

本轮证明的是固定版本、固定模型、固定 decode projection、rank0 single-hit 下的 executor-cache adapter 能力。尚未证明：

- kernel 内部 accumulator 的具体 dtype、位置或溢出时刻；
- 该行为适用于其他 shape、算子、模型、CANN 版本或硬件；
- 并发、soak、异步 buffer pool 或性能 SLA；
- `2^58` 的 `+Inf` 必然来自某一种特定微架构阶段。

因此最终 verdict 只能是 `MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED`。
