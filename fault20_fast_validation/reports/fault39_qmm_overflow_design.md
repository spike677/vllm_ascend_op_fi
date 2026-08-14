# Fault39 QMM 极值输入 / 累加器溢出验证设计

## 1. 本轮状态

状态：`IMPLEMENTED_OFFLINE_E2E_BLOCKED_RESOURCE`。

本轮于 2026-08-12 对 A2 目标服务器完成了启动前检查，并在独立目录完成 CANN 8.5.1 scoped runtime、开发模式 guard、`QMM_EXTREME_X1` 与离线测试。未启动 Qwen3-32B 服务，未对任何模型或 NPU buffer 执行数值注入；模型级 E2E 因双卡资源不满足而停止。

开发工作不占用 NPU；模型 E2E 仍遵循前一阶段未 PASS 时不得进入下一阶段的约束。

## 2. 启动前证据

### 2.1 已确认的基础镜像

目标镜像存在：

```text
quay.io/ascend/vllm-ascend:v0.18.0rc1
```

一次性、不挂 NPU 的只读容器核验结果：

```text
CANN      = 8.5.1
vLLM      = 0.18.0
torch     = 2.9.0+cpu
torch_npu = 2.9.0.post1+gitee7ba04
```

真实 executor-cache 库：

```text
/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libnnopbase.so
```

该库导出了本项目需要核验的符号：

```text
op::internal::OpExecCache::Run(...)
op::internal::AddOpExecCache(...)
op::internal::AddParamToBuf(const aclTensor*)
op::internal::AddParamToBuf(aclTensor*)
op::internal::GetLogApiInfo()
```

这只能证明基础版本与符号存在，不能替代 ABI/E2E 验证。

### 2.2 CANN 8.5.1 项目资产状态

附件指定先读取的下列资产，在 Windows 工作区和服务器 `/home/haoran/fault20_fast_validation` 中均不存在：

```text
reports/fault20_cann851_lossless_validation.md
reports/fault20_cann851_numeric_validation.md
FAULT20_CANN851_USAGE.md
runtime/cann851_validation_v1/
```

历史要求的上述验证证据仍缺失，不能据此宣称既有 CANN 8.5.1 模型 E2E。当前源码已把强制版本 guard 改为编译时常量；默认仍是：

```cpp
constexpr const char *kExpectedLibraryPathFragment = "/8.3.RC2/";
```

独立 CANN 8.5.1 产物使用以下显式构建参数生成：

```text
FAULT20_ASCEND_TOOLKIT=/usr/local/Ascend/cann-8.5.1/aarch64-linux
FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT=/cann-8.5.1/
output=/home/haoran/fault39_dev_20260812/runtime/cann851_validation_v1
metadata proxy=/home/haoran/fault39_dev_20260812/runtime/cann851_metadata_proxy
```

不挂 NPU 的真实加载链 ABI smoke 已得到：

```text
abi_ok=true
developer_mode_ok=true
runtime_api=true
library=/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libnnopbase.so
```

这仍只是 build/ABI smoke，不冒充 Qwen3-32B E2E。

### 2.3 当前 NPU 与端口状态

宿主机当前只枚举出一张卡：

```text
npu-smi total count = 1
NPU ID              = 2
device node         = /dev/davinci2
```

附件要求的 NPU 4/5 不存在；历史容器曾映射 `/dev/davinci2` 和 `/dev/davinci5`，但当前 `/dev/davinci5` 已不在宿主机上。因此当前不能建立 TP=2。

唯一的 NPU 2 已被非本项目容器占用：

```text
container = gyl_test
image     = quay.io/ascend/vllm-ascend:v0.20.2rc1
service   = Qwen3-0.6B, TP=1, port 8008
worker    = PID 815802
HBM       = 28020 MiB
```

未停止、重启或修改该容器。

其他启动前结果：

```text
port 18005                      = free
/dev/shm/fault20*               = absent
/dev/shm/psm_*                  = absent
/dev/shm/sem.loky-*             = absent
Fault20 shared control          = absent (equivalent to not armed)
```

没有 stale IPC 需要删除。

## 3. Fault39 的证据边界

当前 executor-cache 边界暴露的是算子输入/输出 storage 地址，不暴露 kernel 内部 accumulator、UB、L0 或私有 workspace 布局。因此本项目必须记录：

```text
actual accumulator dtype                       = ACCUMULATOR_DTYPE_UNKNOWN
internal accumulator at OpExecCache::Run       = INTERNAL_ACCUMULATOR_NOT_ADDRESSABLE_AT_RUN_BOUNDARY
```

禁止从 opaque/private 布局猜测 accumulator 地址。即使真实 QMM 在极值输入下出现非有限值，也只能先命名为：

```text
EXTREME_MATMUL_NUMERIC_OVERFLOW_REPRODUCED
```

除非出现 kernel/硬件级定位证据，否则不得命名为：

```text
ACCUMULATOR_OVERFLOW_PROVEN
```

## 4. 理论量级（不是硬件 accumulator 类型证明）

对 `K=5120`：

```text
127 * 127 * 5120     = 82,580,480
(-128) * (-128) * 5120 = 83,886,080
```

如果仅作为参考假设 accumulator 为有符号 INT32，则：

```text
82,580,480 / INT32_MAX = 3.845%
127*127 首次超过 INT32_MAX 的 K = 133,145
128*128 首次超过 INT32_MAX 的 K = 131,072
```

所以单独把 `[1,5120]` 的 x1 全部设为 127，不能预期或宣称必然发生 accumulator overflow。它仍是有价值的真实 QMM 极值输入实验。

## 5. QMM slot inventory 的当前完成度

已有模型级证据只闭环：

| formal parameter | address slot | dtype | shape | format | bytes | 状态 |
|---|---:|---|---|---|---:|---|
| x1 | 0 | INT8 | decode `[1,5120]` | ND | 5120 | `PROVEN_E2E`（历史 Fault20 证据） |
| x2 | UNKNOWN | metadata 待取 | metadata 待取 | 预计 NZ，未在本轮证明 | 待取 | `NOT_PROVEN` |
| x1Scale | UNKNOWN | 待取 | 待取 | 待取 | 待取 | `NOT_PROVEN` |
| x2Scale | UNKNOWN | 待取 | 待取 | 待取 | 待取 | `NOT_PROVEN` |
| yScale | UNKNOWN | 待取 | 待取 | 待取 | 待取 | `NOT_PROVEN` |
| bias | UNKNOWN | 待取 | 待取 | 待取 | 待取 | `NOT_PROVEN` |
| out | UNKNOWN | 待取 | 待取 | 待取 | 待取 | `NOT_PROVEN` |

当前 metadata proxy 会记录 QMM typed GWS 的所有 formal tensor metadata，但 runtime 的 `AddParamToBuf` 顺序收集只对 `aclnnMatmulWeightNz` 开启；QMM registry 只保存 x1 metadata。因此不能用 `formal parameter index == addresses index` 填充其余 slot。

后续 inventory 必须在 CANN 8.5.1 适配先通过 lossless 前置后，以同一 cache 指针关联：

```text
typed QMM GWS formal tensor pointer/address
  + AddParamToBuf 动态顺序
  + OpExecCache::Run addresses[] equality
```

## 6. `QMM_EXTREME_X1` 实现

### 6.1 控制面

control ABI 与 SHM 已升级，避免旧 v5 control 被新字段误读：

```text
FAULT20_CONTROL_VERSION = 6
FAULT20_CONTROL_SHM     = /fault20_l4_buffer_substitute_ctl_v6
FAULT20_MODE_QMM_EXTREME_X1
extreme_value           = 127 or -128
extreme_ratio_bps       = 100, 1000, 5000, 10000
extreme_pattern         = positive_max / negative_max / alternating
```

第一阶段只允许：

```text
target_api       = QMM
target device    = logical rank0/device0
target shape     = [1,5120]
selection        = bounded single-hit
pattern          = positive_max
value            = 127
max_bytes        = 5120
enabled default  = 0
developer mode   = VLLM_SERVER_DEV_MODE=1（control 与 runtime 双重检查）
```

### 6.2 数据路径

沿用已经验证过的同步 temporary-buffer 生命周期：

```text
pre-sync
  -> D2H original x1
  -> CPU 在 host temporary 上做 extreme transform
  -> H2D temporary device buffer
  -> copy addresses vector
  -> addresses_copy[0] = temporary
  -> real OpExecCache::Run
  -> post-sync
  -> free temporary device/host buffers
```

原始 x1 和 caller 的 address vector 均不得原地修改。

部分比例建议使用整数 basis points，并明确记录：

```text
selected_elements = floor(storage_elements * ratio_bps / 10000)
effective_ratio   = selected_elements / storage_elements
```

对 5120 元素：

```text
1%   -> 51 elements, effective 0.99609375%
10%  -> 512 elements
50%  -> 2560 elements
100% -> 5120 elements
```

为避免部分比例集中在一个连续前缀，可用确定性的均匀索引选择；实验报告必须同时记录 `selected_elements` 与实际 `changed_elements`，因为原值已经是 127 的元素不计入 changed。

### 6.3 必需 guard

任一条件不满足时原样调用真实 Run：

```text
abi_ok != true
CANN scope != 8.5.1
cache 未注册或 metadata 不完整
producer guard 失败
dtype != INT8
format != ND
shape/stride/offset/bytes 不匹配已验证条件
addresses[0] == nullptr
mode/value/ratio/pattern 非法
非 rank0、非 bounded single-hit 或 max_hits != 1
分配、同步、D2H、H2D 失败
selected_elements == 0
temporary 未与 original 分离
重入
```

### 6.4 离线测试要求

已在不挂 NPU 的 CANN 8.5.1 容器内覆盖：

- INT8 全范围作为输入；
- value 127 与 -128；
- 1%/10%/50%/100% 的 selected count；
- positive/negative/alternating pattern；
- 0、越界 ratio/value 被拒绝；
- canary 验证无越界写；
- dtype、shape、bytes 保持不变；
- 变换统计与逐元素 reference 一致。

结果：

```text
extreme_transform_test: PASS
CANN851_OFFLINE_BUILD_AND_CONTROL_PASS
CANN851_ABI_SMOKE_PASS
CANN851_METADATA_PROXY_BUILD_PASS
```

最终远端产物 SHA-256：

```text
d9642ae990d925fa0127b8394d960352cce49bfebe432f37f2af0f68896868ab  libfault20_buffer_substitute_abi0.so
ac289a8a70b707e438e90bbb7095680b2de96cbf8a9983d31fc3aabb63cc3b5c  fault20_buffer_ctl
9bb765b5ce53e98f4cc083288b122107bfca5890b046f443d62b85b8495d31b3  metadata proxy libopapi.so
234b7582c6c8866001dc8be727c2572e4fe5a4636ec1edd7c1ca143c83ee0996  libopapi_real.so
c4b8fef4861fda1e11a5a198041b06aa823bb8f67dabef4b29a97955cf0ac895  extreme_transform_test
```

## 7. 恢复执行的严格顺序

只有当宿主机重新提供两张可用 NPU，且没有未经授权的占用后，才继续：

1. `DONE_OFFLINE`：已建立独立 `cann851_validation_v1` 并完成精确版本 ABI smoke。
2. `PENDING_E2E`：恢复两张空闲 NPU 后，启动 Qwen3-32B W8A8、TP=2、port 18005，确认 `/health=200` 和 `enabled=0`。
3. 保存并审计 CANN 8.5.1 lossless/numeric 验证证据与使用文档。
4. 保存 deterministic baseline。
5. 动态完成 QMM 全 slot inventory；未证明的 slot 保持 UNKNOWN。
6. 实现并通过 `QMM_EXTREME_X1` offline test。
7. 重跑 QMM lossless single-hit，要求 `baseline == lossless == post`、`run_status=0`。
8. 依次执行 1% -> 10% -> 50% -> 100% 的 rank0 decode `[1,5120]` single-hit；每一级均执行 baseline -> arm -> fault -> reset -> post。
9. 任一级出现 Run 非零、worker crash、OOM、lossless 不一致或 post 不恢复，立即停止升级。

## 8. 本轮结论

```text
QMM_EXTREME_X1_OFFLINE_PASS                    = true
CANN851_RUNTIME_BUILD_PASS                     = true
CANN851_ABI_SMOKE_PASS                         = true
CANN851_METADATA_PROXY_BUILD_PASS              = true
DEVELOPER_MODE_FAIL_CLOSED_PASS                = true
X1_EXTREME_PROVEN                              = false (model E2E not executed)
WEIGHT_EXTREME_PROVEN                          = false (out of current stage)
X1_WEIGHT_EXTREME_PROVEN                       = false (out of current stage)
SCALE_AMPLIFY_PROVEN                           = false (out of current stage)
OUTPUT_INF_PROXY_PROVEN                        = false (out of current stage)
EXTREME_MATMUL_NUMERIC_OVERFLOW_REPRODUCED     = false
ACCUMULATOR_OVERFLOW_PROVEN                    = false
ACCUMULATOR_DTYPE_UNKNOWN                      = true
INTERNAL_ACCUMULATOR_NOT_ADDRESSABLE_AT_RUN_BOUNDARY = true
```

阻塞解除所需的最小外部条件：恢复第二张 NPU，并释放两张目标卡供独立 TP=2 容器使用。CANN 8.5.1 scoped runtime 已生成并通过离线 ABI smoke，但仍必须先通过无注入健康与 lossless single-hit，才能进入四档 extreme E2E。
