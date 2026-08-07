# vLLM Ascend Operator Fault Injection

这是一个面向 vLLM-Ascend/CANN 的原生算子输入精度故障注入原型。当前实现位于 CANN repeat executor-cache replay 边界，通过精确 `LD_PRELOAD` interpose、typed operator adapter 和 temporary device buffer 替换，实现可审计的输入数值退化。

本仓库仅包含最终运行源码、构建/容器脚本和能力说明。请求响应、token 对比、trace、结构化 evidence、离线测试程序和历史探针不上传。

## 已验证能力

| 项目 | 状态 |
|---|---|
| `aclnnQuantMatmulWeightNz`，INT8 x1=`addresses[0]` | `PROVEN_E2E` |
| `aclnnMatmulWeightNz`，BF16 self=`addresses[0]` | `PROVEN_E2E` |
| INT2-equivalent / INT4-equivalent primitive | `PROVEN_E2E` |
| BF16 mantissa truncate | `PROVEN_E2E` |
| single/multi-hit、dual-rank、all-hit selector | `PROVEN_E2E` |

结论严格限定于已验证环境和算子，不代表任意模型、任意算子或跨版本生产能力。详细统计见[当前注入能力统计](fault20_fast_validation/reports/fault20_injection_capability_inventory_20260807.md)。

## 仓库内容

```text
fault20_fast_validation/
├── source/
│   ├── l4_buffer_substitute_allhit_v1/   # 最终 runtime、控制面和容器脚本
│   └── l4_metadata_lifecycle_proxy/      # typed GWS metadata adapter
└── reports/
    ├── fault20_final_architecture.md
    ├── fault20_capability_matrix_final.md
    ├── fault20_fault_primitive_design.md
    └── fault20_injection_capability_inventory_20260807.md
```

## 已验证环境

```text
model       Qwen3-32B-w8a8-nopdmix (dense)
vLLM image  quay.io/ascend/vllm-ascend:v0.12.0rc1
vLLM pkg    0.12.0
torch_npu   2.8.0
CANN target 8.3.RC2
arch        aarch64
C++ ABI     _GLIBCXX_USE_CXX11_ABI=0
execution   TP=2, eager, repeat executor cache
```

## 构建

必须在与推理服务相同的 vLLM-Ascend/CANN 镜像内构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export FAULT20_BUFFER_OUTPUT_DIR=/fault20/runtime/l4_allhit_v1
bash /fault20/source/l4_buffer_substitute_allhit_v1/build.sh

export FAULT20_METADATA_PROXY_OUTPUT_DIR=/fault20/runtime/l4_metadata_lifecycle_proxy
bash /fault20/source/l4_metadata_lifecycle_proxy/build.sh
```

runtime 构建脚本会检查 ABI、SONAME、目标导出符号、意外 CANN `DT_NEEDED` 依赖和 `LD_BIND_NOW` 加载。离线穷举测试及其结果属于本地验证资产，不在公开仓库中。

主要产物：

```text
/fault20/runtime/l4_allhit_v1/libfault20_buffer_substitute_abi0.so
/fault20/runtime/l4_allhit_v1/fault20_buffer_ctl
/fault20/runtime/l4_metadata_lifecycle_proxy/libopapi.so
```

## 容器参考

```bash
bash /fault20/source/l4_buffer_substitute_allhit_v1/create_allhit_container_v2.sh
```

该脚本包含实验机专属的模型路径、NPU 设备 2/5、端口 18005 和挂载路径，只应作为已验证配置参考；迁移前必须调整路径、设备和版本门禁。

关键加载顺序：

```bash
export LD_LIBRARY_PATH=/fault20/runtime/l4_metadata_lifecycle_proxy:$LD_LIBRARY_PATH
export LD_PRELOAD=/fault20/runtime/l4_allhit_v1/libfault20_buffer_substitute_abi0.so
```

control v5：

```text
magic   0x4632304255463035
version 5
SHM     /fault20_l4_buffer_substitute_ctl_v5
```

## 控制命令

默认必须保持 disabled。示例中的 device mask `3` 表示已验证的两个逻辑设备 0/1。

```bash
fault20_buffer_ctl status
fault20_buffer_ctl reset
fault20_buffer_ctl disarm

# QMM all-hit
fault20_buffer_ctl arm-all-qmm-lossless 3 1 5120 5120
fault20_buffer_ctl arm-all-qmm-int2     3 1 5120 5120
fault20_buffer_ctl arm-all-qmm-int4     3 1 5120 5120

# MatmulWeightNz all-hit
fault20_buffer_ctl arm-all-matmul-lossless 3 1 12800 25600
fault20_buffer_ctl arm-all-matmul-mantissa 3 1 12800 25600 bf16 3
```

完整 bounded/multi-rank 命令见 `fault20_buffer_ctl` usage 和[能力矩阵](fault20_fast_validation/reports/fault20_capability_matrix_final.md)。

## Fail-closed 边界

以下条件任一失败时，runtime 应原样调用真实 Run：

- exact API/cache/adapter 不匹配；
- device/dtype/format/shape/stride/offset/bytes 不匹配；
- producer address 或 Matmul storage guard 失败；
- ABI/runtime API/分配/拷贝/同步条件不满足；
- primitive 参数非法、元素未实际改变或选择条件不明确。

禁止依赖 private executor layout、长期缓存 activation 地址、创建新 `aclTensor` 填回 cache，或原地修改调用方 address vector。

## 文档

- [当前注入能力统计](fault20_fast_validation/reports/fault20_injection_capability_inventory_20260807.md)
- [最终能力矩阵](fault20_fast_validation/reports/fault20_capability_matrix_final.md)
- [最终架构](fault20_fast_validation/reports/fault20_final_architecture.md)
- [Primitive 设计](fault20_fast_validation/reports/fault20_fault_primitive_design.md)

## 当前不支持

- 任意 ACLNN 算子自动适配；
- FP16 模型级 E2E；
- packed INT2/INT4 或 kernel/指令/硬件故障；
- prefill 大 shape、warmup/cold、non-repeat-cache；
- 并发、soak、异步 buffer pool、性能 SLA；
- 跨 CANN/torch_npu/vLLM 版本和其他架构；
- MoE、GroupedMatmul、MatmulAllReduce；
- 外部故障框架集成。
