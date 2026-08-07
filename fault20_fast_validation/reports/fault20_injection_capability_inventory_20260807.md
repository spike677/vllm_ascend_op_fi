# Fault 20 当前注入能力统计

日期：2026-08-07

## 1. 统计范围

本报告只统计当前设备、当前源码和已有证据中的 native injection 能力。不设计 DemonCATPFI 集成，不新增实验，不把离线实现或未来设计计入“已验证注入能力”。

版本边界：

| 项目 | 已验证值 |
|---|---|
| 模型 | `Qwen3-32B-w8a8-nopdmix`，dense |
| vLLM | image `v0.12.0rc1`；容器包版本输出 `0.12.0` |
| torch_npu | `2.8.0` |
| CANN 构建目标 | `8.3.RC2` |
| 架构 / C++ ABI | aarch64 / `_GLIBCXX_USE_CXX11_ABI=0` |
| 执行模式 | TP=2、eager、Ascend quantization、repeat executor cache |
| final control | v5，SHM `/fault20_l4_buffer_substitute_ctl_v5` |

## 2. 一页计数

| 统计维度 | 当前数量 | 说明 |
|---|---:|---|
| 已验证 exact operator adapter | **2** | QMM、MatmulWeightNz |
| 已验证 numeric fault primitive | **3** | INT2-equivalent、INT4-equivalent、BF16 mantissa truncate |
| 已验证 lossless 验证模式 | **1** | D2D temporary address substitution，不是数值故障 |
| 已验证 selector 模式 | **4** | single hit、same-rank multi-hit、dual-rank bounded、dual-rank all-hit |
| numeric E2E 场景配置 | **10** | 见第 6 节；同一 primitive 在不同 selector/任务下分别计场景 |
| lossless E2E 场景配置 | **6** | QMM/Matmul 的 single/multi-rank/all-hit 预检 |
| 最终 all-hit 结构化 stage | **7** | 每个 stage 都有 baseline/fault/post/evidence |
| 最终 all-hit 成功 Run events | **78,848** | 含 73,088 numeric + 5,760 lossless |
| 最终 all-hit changed-element occurrences | **373,016,307** | 各 event 的 changed 之和，不是全局唯一元素数 |
| 已验证 target dtype/shape 组合 | **2** | INT8 `[1,5120]`；BF16 `[1,12800]` |
| FP16 模型级注入能力 | **0** | 有实现和离线测试，但没有真实 FP16 E2E |
| arbitrary operator / cross-version 能力 | **0** | 不计入当前能力 |

## 3. 已验证的两个 Adapter

### 3.1 `aclnnQuantMatmulWeightNz`

```text
target input = x1
target slot  = addresses[0]
dtype/format = INT8 / ND
shape        = [1,5120]
stride       = [5120,1]
offset       = 0
bytes        = 5120
producer     = aclnnAscendQuantV3
```

已验证故障：INT2-equivalent、INT4-equivalent。

已验证选择：single hit、same-rank 3-hit（INT4）、dual-rank each-one（INT4）、dual-rank all-hit（INT2/INT4）。

关键 guard：exact API、cache metadata、INT8/ND、rank/shape/stride/offset/bytes、连续 storage、view bytes=storage bytes、producer current output address 匹配、device mask、selection mode、runtime API、quant step。

### 3.2 `aclnnMatmulWeightNz`

```text
formal slots = 0:self, 1:mat2, 2:out
target input = self
target slot  = addresses[0]
dtype/format = BF16 / ND
shape        = [1,12800]
stride       = [12800,1]
offset       = 0
bytes        = 25600
```

已验证故障：BF16 mantissa truncate，`keep_bits=3`。

已验证选择：single hit、dual-rank all-hit。

附加 guard：三 slot 参数顺序、all present、no empty storage、no shared storage、`addresses.size()==3`、target dtype、mantissa bit 范围和 reentrancy guard。

## 4. 已验证的三个 Numeric Primitive

| Primitive | 数学/位规则 | storage 是否改变 | 状态 |
|---|---|---|---|
| INT2-equivalent | `round(x/64) -> clamp[-2,1] -> *64` | 否，INT8→INT8 | `PROVEN_E2E` |
| INT4-equivalent | `round(x/16) -> clamp[-8,7] -> *16` | 否，INT8→INT8 | `PROVEN_E2E` |
| BF16 mantissa truncate | 7-bit mantissa 保留高 3 bits，低 4 bits 清零 | 否，BF16→BF16 | `PROVEN_E2E` |

三者均使用：

```text
current device address
  -> D2H pinned host
  -> CPU transform
  -> H2D temporary device buffer
  -> vector_copy[target_slot] replacement
  -> real OpExecCache::Run
  -> post-sync and free
```

不改变 shape、dtype、bytes、layout、weight、scale、bias 或 cubeMathType。

## 5. Final all-hit 七阶段统计

所有阶段都满足：

```text
candidate_count_per_device
  == hit_count_per_device
  == successful_event_count_per_device
```

| Stage | rank0 | rank1 | events | changed occurrences | 模型输出 |
|---|---:|---:|---:|---:|---|
| QMM lossless short | 1,920 | 1,920 | 3,840 | 0 | 与 baseline 完全一致 |
| QMM INT2 short | 1,920 | 1,920 | 3,840 | 19,324,278 | 15/16 tokens 不同；后 15 个为 token 222 |
| QMM INT4 short | 1,920 | 1,920 | 3,840 | 18,418,650 | 15/16 tokens 不同；乱码但 token 多样 |
| QMM INT4 long | 15,488 | 15,488 | 30,976 | 148,614,144 | 122 tokens；严重语义破坏，95 unique tokens |
| QMM INT2 long | 16,256 | 16,256 | 32,512 | 163,625,418 | 128 tokens；`<think>` 后 token 222 连续 127 次 |
| Matmul lossless short | 960 | 960 | 1,920 | 0 | 与 baseline 完全一致 |
| Matmul BF16 short | 960 | 960 | 1,920 | 23,033,817 | token 与 baseline 一致；算子侧变化成立 |
| **合计** | **39,424** | **39,424** | **78,848** | **373,016,307** | 全部 HTTP 200，post 恢复 |

INT4 long 与 INT2 long 的 eligible Run 数不同，是 fault 输出改变后续生成轨迹和停止位置导致；两次各自请求窗口内仍为 100% all-hit coverage。

## 6. Numeric E2E 场景配置计数（10）

| # | Adapter | Primitive | Selector / workload | 模型输出变化 |
|---:|---|---|---|---|
| 1 | QMM | INT4 | single hit | 有 |
| 2 | QMM | INT2 | single hit | 有 |
| 3 | MatmulWeightNz | BF16 truncate | single hit | 有，早期固定请求 3-token difference |
| 4 | QMM | INT4 | same-rank 3 hit | 有 |
| 5 | QMM | INT4 | dual-rank each-one | 有 |
| 6 | QMM | INT2 | dual-rank all-hit short | 有，token collapse |
| 7 | QMM | INT2 | dual-rank all-hit long | 有，127-token collapse |
| 8 | QMM | INT4 | dual-rank all-hit short | 有，乱码 |
| 9 | QMM | INT4 | dual-rank all-hit long | 有，严重语义破坏 |
| 10 | MatmulWeightNz | BF16 truncate | dual-rank all-hit short | **无 token 变化**；23,033,817 elements changed、1,920 Run 成功 |

这里的“模型输出变化”与“算子侧注入成功”分别计量。场景 #10 的 mechanism E2E 成立，但该请求没有输出差异。

## 7. Lossless E2E 场景配置计数（6）

| # | Adapter | Selector | 作用 |
|---:|---|---|---|
| 1 | QMM | single hit | 第一次证明 temporary slot0 可用 |
| 2 | MatmulWeightNz | single hit | 第二算子 slot0/self 可用 |
| 3 | QMM | same-rank 3 hit | 连续同步生命周期可用 |
| 4 | QMM | dual-rank each-one | 两 rank 独立计数/替换可用 |
| 5 | QMM | dual-rank all-hit | 1,920 hits/rank，超过旧 1,024 限制 |
| 6 | MatmulWeightNz | dual-rank all-hit | 960 hits/rank |

Lossless 是机制验证模式，不计为 precision fault primitive。

## 8. 当前 Control v5 可用命令

| 命令族 | 当前验证情况 |
|---|---|
| `reset` / `status` / `disarm` | 已验证 |
| `arm-lossless` | QMM bounded 已验证 |
| `arm-int4` | QMM bounded 已验证 |
| `arm-int2` | QMM bounded 已验证 |
| `arm-mantissa ... bf16` | Matmul bounded 已验证 |
| `arm-multirank-lossless` | QMM 已验证 |
| `arm-multirank-int4` | QMM 已验证 |
| `arm-all-qmm-lossless` | 已验证 |
| `arm-all-qmm-int2` | 已验证 |
| `arm-all-qmm-int4` | 已验证 |
| `arm-all-matmul-lossless` | 已验证 |
| `arm-all-matmul-mantissa ... bf16` | 已验证 |
| `arm-mantissa ... fp16` | 代码/离线测试存在，模型 E2E 未验证，不计当前注入能力 |

## 9. 当前 Runtime 产物

最终 all-hit build：

```text
/home/haoran/fault20_fast_validation/runtime/l4_allhit_v1/
  libfault20_buffer_substitute_abi0.so
  fault20_buffer_ctl
  fault20_int_equivalent_test
  fault20_mantissa_truncate_test
```

SHA-256：

```text
0498e60e2501c693434b990ebfb4a4736a09c5f5b533f1d466eb4663af9abe81  libfault20_buffer_substitute_abi0.so
40c357c9b510a18b6f5352643e8b0c0c486dfe1b53c9f3dd16384c9b5facea21  fault20_buffer_ctl
9ee0caa05845e70dded0783ab29883ade742ef3d167cf19deb4fbb2f5f9c1310  fault20_int_equivalent_test
25656e15999c69ef1af39245198426690943d5b905c972c7dcdbabafac449e49  fault20_mantissa_truncate_test
```

最终实验容器 `fault20-qwen3-32b-allhit-v2`：Exited(0)、OOM=false、已保留。失败启动证据 `fault20-qwen3-32b-allhit-v1`：Exited(1)、OOM=false、已保留。当前健康基线 `fault20-qwen3-32b-l4-cacheadd-trace` 正在运行，最后只读检查 `/health=200`。

## 10. 不计入当前能力

- FP16 模型级 injection；
- 第三个或任意 ACLNN adapter；
- packed INT2/INT4、kernel/指令/硬件故障；
- prefill 大 shape、warmup/cold、non-repeat-cache；
- 并发请求、soak、async buffer pool、性能 SLA；
- 跨 CANN/torch_npu/vLLM 版本或其他架构；
- MoE、GroupedMatmul、MatmulAllReduce；
- 外部故障框架集成。当前设备不存在 DemonCATPFI，本报告不设计或评估集成。

## 11. 验证依据

模型请求/响应、token 对比、结构化 evidence、raw JSONL/trace 和阶段验证脚本均保留在本地实验工作区，不上传公开仓库。本文件只固化由这些材料审计得到的能力统计；它不能替代原始证据包。
