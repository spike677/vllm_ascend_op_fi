# Fault 20 Precision Fault Primitive 设计

日期：2026-08-07

## 1. 统一契约

Fault 20 primitive 只负责确定性的数值变换，不负责算子识别、cache 注册、rank/device/hit 选择或 device buffer 生命周期。

共同契约：

```text
input  = dtype + element_count + byte buffer + primitive parameters
output = same dtype + same element_count + same bytes
```

必须保持：

- dtype 不变；
- shape、stride、format 和 view offset 不变；
- storage bytes 不变；
- adapter 的其他输入（weight/scale/offset/bias/out）不变；
- 变换结果写入 temporary buffer，不原地修改调用方 activation。

`changed_elements == 0` 时，numeric injection 不算有效命中，应 fail closed 并调用真实 Run 的原始地址。

## 2. INT2-equivalent

### 2.1 数学定义

对每个有符号 INT8 输入 `x`：

```text
q = round_away_from_zero_on_half(x / 64)
q = clamp(q, -2, 1)
out = q * 64
```

实现等价于：

```cpp
int q2 = fault20_round_nearest_away_from_zero(x, 64);
q2 = clamp(q2, -2, 1);
int8_t out = static_cast<int8_t>(q2 * 64);
```

输出有效等级为：

```text
-128, -64, 0, 64
```

### 2.2 准确语义

它是：**INT8 storage 中的 2-bit-equivalent effective precision degradation**。

它不是：

- packed INT2；
- INT2 kernel；
- 修改 QMM weight/scale；
- 芯片计算单元真实 bit fault。

### 2.3 已验证状态

- 所有 256 个 INT8 输入离线测试通过；
- QMM single-hit E2E 通过；
- QMM dual-rank all-hit 短/长任务通过；
- 长任务形成 `151667, 222 × 127` token 坍缩。

## 3. INT4-equivalent

### 3.1 数学定义

```text
q = round_away_from_zero_on_half(x / 16)
q = clamp(q, -8, 7)
out = q * 16
```

输出仍是 INT8 storage，共 16 个主要等级，范围 `[-128,112]`，步长 16。

### 3.2 准确语义

它是：**INT8 storage 中的 4-bit-equivalent effective precision degradation**。

它不是 packed INT4 或 INT4 kernel。QMM 原始 x1 dtype、shape、scale、weight、bias 和输出 descriptor 均不改变。

第一版选择量化到 16 级，而没有简单 `clamp(x,-8,7)`，是为了保留原动态范围的主要结构，避免把“有效精度损失”和明显的 scale 压缩混成一个故障。

### 3.3 已验证状态

- 所有 256 个 INT8 输入离线测试通过；
- single-hit、same-rank 3-hit、dual-rank each-one、dual-rank all-hit E2E 通过；
- 短任务输出乱码；长任务保持 token diversity 但发生严重语义破坏。

## 4. BF16 mantissa truncate

### 4.1 格式与变换

BF16 位布局：

```text
sign: 1 bit | exponent: 8 bits | mantissa: 7 bits
```

当前 E2E 参数：

```text
keep_bits = 3
clear_bits = 7 - 3 = 4
```

对有限值清零 mantissa 的低 4 bits；sign、exponent 和高 3 mantissa bits 保持。指数全 1 的 NaN/Inf raw encoding 原样返回；`keep_bits >= 7` 为 identity。

位运算：

```cpp
if (exponent_all_ones || keep_bits >= 7) return bits;
mask = ~((1 << (7 - keep_bits)) - 1);
out_bits = bits & mask;
```

### 4.2 准确语义

- 输入输出都为 BF16；
- shape/bytes/layout 不变；
- 不是 BF16→FP16 cast；
- 不是修改 cubeMathType；
- 不是 kernel 重编译。

### 4.3 已验证状态

- 65,536 个 BF16 raw encoding 离线穷举测试通过；
- `aclnnMatmulWeightNz` single-hit E2E 通过，早期固定请求出现 3 个 token 差异；
- all-hit 双 rank 共 1,920 个真实 Run，23,033,817 个 BF16 elements changed；该短请求 token 与 baseline 一致。

因此必须分别陈述：

```text
operator-side precision change = proven
model-output change in all-hit short request = not observed
```

## 5. FP16 mantissa truncate

代码实现支持 FP16 10-bit mantissa 的低位清零，并已穷举 65,536 个 raw encoding；但当前真实 MatmulWeightNz adapter 只观察到 BF16 模型事件。

状态：`IMPLEMENTED_NOT_VALIDATED`。不得写成当前模型已支持 FP16 E2E。

## 6. 执行流程

数值 primitive 的当前正确性优先流程：

```text
pre-sync stream
  -> aclrtMallocHost(pinned host)
  -> D2H current addresses[target_slot]
  -> CPU primitive transform + statistics
  -> aclrtMalloc(temporary device)
  -> H2D on the same stream
  -> vector_copy[target_slot] = temporary
  -> real OpExecCache::Run
  -> post-sync stream
  -> free temporary device and host buffer
```

记录但不输出完整 tensor：changed/unchanged、min/max、zero/nonfinite、max error、每一步 status/timing、地址替换和释放状态。

## 7. 停止条件

以下任一条件成立时不得调用 numeric primitive：

- adapter/cache/API 不匹配；
- dtype/shape/stride/offset/format/bytes 未通过 exact guard；
- target slot 为空或 address vector 数量不符；
- producer guard 失败；
- primitive 参数与模式不一致（INT4 step≠16、INT2 step≠64、mantissa keep 超界）；
- runtime API、分配、D2H/H2D 或同步不可用；
- 变换后 `changed_elements == 0`；
- ABI、架构或版本门禁失败。

关联实现：

- `source/l4_buffer_substitute_allhit_v1/int_equivalent_transform.h`
- `source/l4_buffer_substitute_allhit_v1/mantissa_truncate.h`
- `source/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp`
